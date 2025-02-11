import warnings
from collections import defaultdict
from contextlib import nullcontext
from types import MethodType
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from transformers import Trainer
from trl import GRPOTrainer  # GRPO trainer from TRL
from trl.trainer import disable_dropout_in_model
from typing_extensions import override
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler, get_batch_logps, nested_detach
if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin
    from ...hparams import FinetuningArguments
class CustomGRPOTrainer(GRPOTrainer):
    """
    Custom GRPO trainer that implements GRPO loss using a reference model
    (if provided) for baseline correction. This implementation follows the
    style of our CustomDPOTrainer and adapts the GRPO-specific loss formulation.
    """
    def __init__(
        self,
        model: Union["PreTrainedModel", torch.nn.Module],
        ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]],
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        disable_dropout: bool = True,
        **kwargs,
    ):
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        if disable_dropout:
            disable_dropout_in_model(model)
            if ref_model is not None:
                disable_dropout_in_model(ref_model)
        self.finetuning_args = finetuning_args
        # GRPO-specific hyperparameter (ensure this exists in your FinetuningArguments)
        self.grpo_gamma = getattr(finetuning_args, "grpo_gamma", 1.0)
        self.reference_free = False
        self.label_pad_token_id = IGNORE_INDEX
        self.padding_value = 0
        self.is_encoder_decoder = model.config.is_encoder_decoder
        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        self.ref_model = ref_model
        # Initialize the base Trainer via Trainer.__init__
        Trainer.__init__(self, model=model, **kwargs)
        self.model_accepts_loss_kwargs = False  # overwrite Trainer's default behavior
        if not hasattr(self, "accelerator"):
            raise AttributeError("Please update `transformers`.")
        warnings.simplefilter("ignore")  # suppress warnings on ref_model
        if ref_model is not None:
            if self.is_deepspeed_enabled:
                if not (getattr(ref_model, "is_loaded_in_8bit", False) or getattr(ref_model, "is_loaded_in_4bit", False)):
                    self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
                self.ref_model.eval()
        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))
        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore
            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)
    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()
    @override
    def create_scheduler(self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)
    @override
    def _get_train_sampler(self) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)
        return super()._get_train_sampler()
    @override
    def get_batch_samples(self, epoch_iterator, num_batches):
        """
        Replaces the method of the base Trainer with that of the standard Trainer.
        """
        return Trainer.get_batch_samples(self, epoch_iterator, num_batches)
    @override
    def concatenated_forward(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """
        Performs a forward pass that computes the (possibly averaged) log probabilities.
        This is assumed to operate on paired examples (e.g. chosen vs. rejected).
        """
        if self.finetuning_args.use_ref_model:
            batch = nested_detach(batch, clone=True)  # avoid error
        outputs = model(**batch, return_dict=True, use_cache=False)
        all_logits: "torch.Tensor" = outputs.logits.to(torch.float32)
        all_logps, valid_length = get_batch_logps(logits=all_logits, labels=batch["labels"])
        # For GRPO, we average per valid token
        all_logps = all_logps / valid_length
        batch_size = batch["input_ids"].size(0) // 2
        chosen_logps, rejected_logps = all_logps.split(batch_size, dim=0)
        chosen_logits, rejected_logits = all_logits.split(batch_size, dim=0)
        return chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_logps
    @override
    def compute_reference_log_probs(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    ) -> Tuple[Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        """
        Computes log probabilities from the reference model if one is provided.
        """
        if not self.finetuning_args.use_ref_model:
            return None, None
        if self.ref_model is None:
            ref_model = model
            ref_context = self.accelerator.unwrap_model(model).disable_adapter()
        else:
            ref_model = self.ref_model
            ref_context = nullcontext()
        with torch.no_grad(), ref_context:
            reference_chosen_logps, reference_rejected_logps, *_ = self.concatenated_forward(ref_model, batch)
        return reference_chosen_logps, reference_rejected_logps
    def compute_grpo_loss(
        self,
        policy_chosen_logps: "torch.Tensor",
        policy_rejected_logps: "torch.Tensor",
        reference_chosen_logps: Optional["torch.Tensor"],
        reference_rejected_logps: Optional["torch.Tensor"],
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """
        Computes the GRPO loss for a batch.
        
        If a reference model is used, the loss is computed using a baseline correction.
        Otherwise, it uses the raw log probability differences.
        """
        if self.finetuning_args.use_ref_model and reference_chosen_logps is not None and reference_rejected_logps is not None:
            # Use the reference model for baseline adjustment
            advantage = (policy_chosen_logps - reference_chosen_logps) - (policy_rejected_logps - reference_rejected_logps)
        else:
            advantage = policy_chosen_logps - policy_rejected_logps
        # GRPO loss: scale the advantage and apply negative log-sigmoid.
        grpo_loss = -F.logsigmoid(self.grpo_gamma * advantage)
        chosen_rewards = policy_chosen_logps.detach()
        rejected_rewards = policy_rejected_logps.detach()
        return grpo_loss, chosen_rewards, rejected_rewards
    @override
    def get_batch_loss_metrics(
        self,
        model: "PreTrainedModel",
        batch: Dict[str, "torch.Tensor"],
        train_eval: str = "train",
    ) -> Tuple["torch.Tensor", Dict[str, "torch.Tensor"]]:
        """
        Computes the GRPO loss and related metrics for the given batch.
        """
        metrics = {}
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_chosen_logps_avg,
        ) = self.concatenated_forward(model, batch)
        reference_chosen_logps, reference_rejected_logps = self.compute_reference_log_probs(model, batch)
        loss, chosen_rewards, rejected_rewards = self.compute_grpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        # Optionally add an SFT loss term if required
        sft_loss = -policy_chosen_logps_avg
        if getattr(self.finetuning_args, "sft_loss_weight", 0.0) > 1e-6:
            loss = loss + self.finetuning_args.sft_loss_weight * sft_loss
        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().item()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().item()
        metrics[f"{prefix}rewards/accuracies"] = (chosen_rewards > rejected_rewards).float().mean().item()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().item()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.mean().item()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.mean().item()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.mean().item()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.mean().item()
        if getattr(self.finetuning_args, "sft_loss_weight", 0.0) > 1e-6:
            metrics[f"{prefix}sft_loss"] = sft_loss.mean().item()
        return loss.mean(), metrics
    @override
    def compute_loss(
        self, model: "PreTrainedModel", inputs: Dict[str, "torch.Tensor"], return_outputs: bool = False, **kwargs
    ) -> Union["torch.Tensor", Tuple["torch.Tensor", list]]:
        """
        Computes the loss by delegating to the parent compute_loss method.
        """
        return super().compute_loss(model, inputs, return_outputs=return_outputs)
    @override
    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        """
        Logs metrics by merging any stored metrics from intermediate steps.
        """
        train_eval = "train" if "loss" in logs else "eval"
        key_list, metric_list = [], []
        for key, m_list in self._stored_metrics[train_eval].items():
            key_list.append(key)
            metric_list.append(torch.tensor(m_list, dtype=torch.float).to(self.accelerator.device).mean().item())
        del self._stored_metrics[train_eval]
        # Pad if necessary for distributed reduction
        if len(metric_list) < 10:
            for i in range(10 - len(metric_list)):
                key_list.append(f"dummy_{i}")
                metric_list.append(0.0)
        metric_tensor = torch.tensor(metric_list, dtype=torch.float).to(self.accelerator.device)
        metric_tensor = self.accelerator.reduce(metric_tensor, "mean")
        metric_list = metric_tensor.tolist()
        for key, metric in zip(key_list, metric_list):
            if not key.startswith("dummy_"):
                logs[key] = metric
        return Trainer.log(self, logs, *args, **kwargs)