import os
import json
import logging
import warnings
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)
import wandb

from adaptive_trainer.config import TokenizerConfig
from adaptive_trainer.collators import CustomDataCollator
from adaptive_trainer.trainers import AdaptiveTrainer
from adaptive_trainer.data_processors import DataProcessor
from adaptive_trainer.utils import upload_model_to_huggingface, setup_environment, cleanup_memory

logger = logging.getLogger(__name__)

def train_adaptively(
    model_name,
    datasets_config,
    output_dir,
    huggingface_repo=None,
    system_prompts=None,
    datasets_kwargs=None,
    dataset_specific_system_prompts=None,
    training_config=None,
    adaptive_loss_config=None,
    upload_config=None
):
    """
    Main training function that coordinates the Adaptive_select loss training process.
    
    Args:
        model_name: Base model name from HuggingFace
        datasets_config: Dictionary with dataset configuration
        output_dir: Directory to save model and logs
        huggingface_repo: HuggingFace repo name to upload the model (optional)
        system_prompts: Dictionary of system prompts keyed by learning style
        training_config: Dictionary with training parameters
        adaptive_loss_config: Dictionary with adaptive loss parameters
        upload_config: Dictionary with model upload parameters
    
    Returns:
        Path to the saved model
    """
    # Initialize defaults
    if training_config is None:
        training_config = {}
    
    if adaptive_loss_config is None:
        adaptive_loss_config = {}
    
    if upload_config is None:
        upload_config = {
            "is_private_repo": False,
            "commit_message": "Upload fine-tuned model with adaptive-select loss"
        }
    
    # Setup environment
    setup_environment()
    warnings.filterwarnings("ignore")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup run name
    run_name = training_config.get("run_name", f"adaptive_select_loss_{model_name.split('/')[-1]}")
    
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load tokenizer configuration based on model
    tokenizer_config, chat_template = TokenizerConfig.from_model_name(model_name, tokenizer)
    tokenizer.chat_template = chat_template

    # Setup padding token
    if tokenizer.pad_token is None:
        if tokenizer.unk_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            new_pad_token = "<|pad|>"
            if new_pad_token not in tokenizer.get_vocab():
                tokenizer.add_special_tokens({'pad_token': new_pad_token})
                logger.info(f"Added new pad token '{new_pad_token}' ({tokenizer.pad_token_id}).")
            else:
                tokenizer.pad_token = new_pad_token
                logger.info(f"Using existing token '{new_pad_token}' ({tokenizer.pad_token_id}) as pad token.")
        else:
            tokenizer.pad_token = tokenizer.unk_token
            logger.info(f"Using <unk> token ({tokenizer.unk_token_id}) as pad token.")

    # Set padding side
    tokenizer.padding_side = training_config.get("padding_side", "left")
    if tokenizer.padding_side != "left":
        print("Warning padding side is not set to left, proceed with caution as this may not be the best practice")
    
    # Get output demarkation tokens for identifying model response sections
    output_demarker = tokenizer_config.assistant_header.strip()
    empty_token_length_for_tokenizer = len(tokenizer(''))
    output_demarkation_ids = tokenizer(output_demarker)['input_ids'][empty_token_length_for_tokenizer:]

    # Setup data processor
    max_length = training_config.get("max_length_token", 4096)
    data_processor = DataProcessor(tokenizer, tokenizer_config, max_length)
    
    # Prepare datasets
    if system_prompts:
        datasets_config["system_prompts"] = system_prompts

    if datasets_kwargs:
        datasets_config["datasets_kwargs"] = datasets_kwargs

    if dataset_specific_system_prompts:
        datasets_config["dataset_specific_system_prompts"] = dataset_specific_system_prompts

    train_dataset, eval_dataset = data_processor.combine_datasets(datasets_config)
    
    # Load model
    attn_impl = training_config.get("attn_impl", None)
    logger.info(f"Loading model {model_name} with attention implementation: {attn_impl}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=attn_impl
    )
    
    # Set pad token ID and resize token embeddings if needed
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    # Make sure all parameters are trainable
    for param in model.parameters():
        param.requires_grad = True
    
    # Use a data collator for language modeling
    data_collator = CustomDataCollator(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_config.get("num_epochs", 3),
        per_device_train_batch_size=training_config.get("batch_size", 4),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 16),
        per_device_eval_batch_size=training_config.get("eval_batch_size", training_config.get("batch_size", 4)),
        eval_strategy=training_config.get("eval_strategy", 'steps'),
        eval_steps=training_config.get("eval_steps", 200 if training_config.get("save_strategy", 'steps')=='steps' else None),
        logging_dir=f"{output_dir}/logs",
        logging_steps=training_config.get("logging_steps", 10),
        learning_rate=training_config.get("learning_rate", 2e-5),
        weight_decay=training_config.get("weight_decay", 0.01),
        warmup_steps=training_config.get("warmup_steps", 10),
        save_strategy=training_config.get("save_strategy", 'steps'),
        save_steps=training_config.get("save_steps", 300 if training_config.get("save_strategy", 'steps')=='steps' else None),
        fp16=training_config.get("fp16", True),
        report_to="wandb" if training_config.get("use_wandb", True) else "none",
        local_rank=training_config.get("local_rank", -1),
        run_name=run_name,
        save_total_limit=training_config.get("save_total_limit", 5),
        label_names=['input_ids', 'attention_mask', 'labels', 'learning_style'],
        gradient_checkpointing=training_config.get("gradient_checkpointing", True),
        optim=training_config.get("optimizer", "paged_adamw_8bit"),
        lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
        eval_on_start=training_config.get("eval_on_start", False),
        use_liger_kernel=training_config.get("use_liger_kernel", False)
    )

    # Initialize wandb if requested
    if training_config.get("use_wandb", True):
        wandb_project = training_config.get("wandb_project", "selective-loss-training")
        wandb_entity = training_config.get("wandb_entity", None)
        wandb.init(project=wandb_project, entity=wandb_entity, name=run_name)
    
    # Create trainer with selective loss
    trainer = AdaptiveTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        output_demarkation_token_ids=output_demarkation_ids,
        top_k=adaptive_loss_config.get("top_k", 10),
        initial_confidence_threshold=adaptive_loss_config.get("initial_confidence", 0.7),
        final_confidence_threshold=adaptive_loss_config.get("final_confidence", 0.9),
        max_kl_divergence=adaptive_loss_config.get("max_kl_divergence", 0.2),
        curriculum_epochs=adaptive_loss_config.get("curriculum_epochs", 1.0),
        stats_save_path=f"{output_dir}/selective_loss_stats.json",
        log_metric_steps=adaptive_loss_config.get("adaptive_log_steps", 100)
    )
    
    # Start training
    trainer.train()
    
    # Save final model
    final_model_path = os.path.join(output_dir, "final_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    # Record upload info
    if huggingface_repo:
        with open(os.path.join(output_dir, 'upload_hf.txt'), 'w') as f:
            upload_info = {
                'model_path': final_model_path,
                'hf_model_name': huggingface_repo,
                'token': os.environ.get('HF_TOKEN', ''),
                'private': upload_config.get("is_private_repo", False)
            }
            f.write(f'{json.dumps(upload_info)}\n\n')
        
        # Upload model to HuggingFace if requested
        if os.environ.get('HF_TOKEN'):
            logger.info(f"Uploading model to HuggingFace: {huggingface_repo}")
            upload_model_to_huggingface(
                model_path=final_model_path,
                hf_model_name=huggingface_repo,
                model_class=AutoModelForCausalLM,
                token=os.environ.get('HF_TOKEN'),
                private=upload_config.get("is_private_repo", False),
                commit_message=upload_config.get("commit_message", "Upload fine-tuned model"),
                base_model_name=model_name
            )
    
    # Clean up
    cleanup_memory()
    
    return final_model_path
