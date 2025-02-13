# workflow.py
from typing import TYPE_CHECKING, List, Optional

from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset

from ...data import get_dataset, get_template_and_fix_tokenizer  # Assuming these are correct
from ...extras.ploting import plot_loss
from ...hparams import ModelArguments
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push, create_ref_model  # Keep import

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback
    from ...hparams import DataArguments, FinetuningArguments

def run_grpo(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    # Load tokenizer and fix template
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    # Get dataset.  GRPOTrainer expects a dataset with 'prompt' and 'completion' columns.
    dataset = load_dataset("trl-lib/tldr", split="train")  # Replace with your dataset

    # Load model
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)


    # Define GRPOConfig.  GRPOConfig INHERITS from TrainingArguments.
    grpo_config = GRPOConfig(
        # Directly use TrainingArguments fields (no need to duplicate)
        output_dir=training_args.output_dir,
        learning_rate=training_args.learning_rate,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        optim=training_args.optim,
        logging_steps=training_args.logging_steps,
        report_to=training_args.report_to,
        fp16=training_args.fp16,
        bf16=training_args.bf16,
        do_train=training_args.do_train,
        do_eval=training_args.do_eval,
        lr_scheduler_type=training_args.lr_scheduler_type,
        warmup_ratio=training_args.warmup_ratio,
        weight_decay=training_args.weight_decay,
        max_grad_norm=training_args.max_grad_norm,
        save_steps=training_args.save_steps,
        save_strategy="steps",  # Good practice
        max_steps=training_args.max_steps if training_args.max_steps > 0 else -1, #Important
        num_train_epochs=training_args.num_train_epochs, # Use this if max_steps is not set.

        # GRPO-specific arguments (from your FinetuningArguments and DataArguments)
        beta=finetuning_args.pref_beta,  # Corrected from previous responses
        max_prompt_length=data_args.cutoff_len,
        max_completion_length=256,   #  Set a reasonable default.  ADJUST AS NEEDED.
        remove_unused_columns=False,  # CRUCIAL:  GRPOTrainer needs other columns
        use_vllm=False, # Or set based on your needs/availability

        # Optional GRPO arguments (set if present in finetuning_args)
        # If you want to use generate_kwargs, you would NOT set these here.
    )
    if finetuning_args:
        if hasattr(finetuning_args, "temperature"):
            setattr(grpo_config, "temperature", finetuning_args.temperature)
        if hasattr(finetuning_args, "num_generations"):
            setattr(grpo_config, "num_generations", finetuning_args.num_generations)
        if hasattr(finetuning_args, "max_completion_length"):
            setattr(grpo_config, "max_completion_length", finetuning_args.max_completion_length)


    # Define reward functions (example)
    def example_reward_function(prompts, completions, **kwargs):
        # Placeholder - MUST BE REPLACED with your actual reward function(s)
        # This example assumes completions is a list of strings.
        return [len(completion) for completion in completions]

    # Instantiate GRPOTrainer
    trainer = GRPOTrainer(
        model=model,  # Pass the model
        args=grpo_config, # Pass the config
        reward_funcs=[example_reward_function],  # Pass your reward function(s)
        train_dataset=dataset,
        # eval_dataset=eval_dataset,  # Add if you have an eval set
        processing_class=tokenizer,  # Pass the tokenizer as processing_class
        callbacks=callbacks,
    )


    # Training phase
    if training_args.do_train:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss", "rewards/accuracies"])

    # Evaluation phase
    if training_args.do_eval:
        trainer.evaluate()

    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)