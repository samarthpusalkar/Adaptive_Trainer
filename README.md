# Adaptive Trainer

A Hugging Face `Trainer` extension designed for more efficient and effective fine-tuning of language models using adaptive loss mechanisms and curriculum learning principles.

## Features

*   **Adaptive Loss Calculation:** Focuses training on tokens where the model is less confident or incorrect, rather than all tokens equally.
*   **Ideas Learning:** Allows model to learn concepts and behaviour from training data rather than actual memorisation much like student learning.
*   **Learning Style Differentiation:** Also allows specific datasets to be learnt with attention and concepts with ideas.Allows for different training objectives (e.g., "ideas" vs. "attention") based on dataset characteristics.
*   **Integration with Hugging Face:** Built on top of the `transformers` library and `Trainer` API.
*   **More Features** like parameter control and hyperparameter optimisation of loss for user comming soon


## Not Implemented yet:
Custom dataset processing as user_input..

## Installation

You can install `adaptive-trainer` using pip:

```bash
pip install adaptive-trainer
```
*(Once you upload it to PyPI. For local development:)*

```bash
pip install .
```
*Or, if you want to install with optional dependencies like flash-attention:*
```bash
pip install .[flash_attn]
```

## Quick Start

```python
from adaptive_trainer import train_adaptively, AdaptiveTrainer # Assuming you export AdaptiveTrainer too

# Define system prompts
system_prompts = {
    'both': "Your role as an assistant is ...",
    'ideas': "Your role as an assistant is ...",
    'attention': "Your role as an assistant is ..."
}

# Configure datasets
datasets_config = {
    'ideas': ["user/my-ideas-dataset:|:K"],
    'attention': ["user/my-attention-dataset"],
    'both': [],
    'misc': []
}
# The above will trim the `user/my-ideas-dataset` till top K values and similar thing can be done for attention datasets too just add :|:K at the end of dataset name and it will only use top K rows of the dataset..

dataset_kwargs = {
    'user/my-ideas-dataset:|:K': {'data_dir'=None},
    'user/my-ideas-dataset': {}, # Both keys are valid and should work
    'user/my-attention-dataset': {}
}
# If you want to added custom arguments to `load_dataset` function from `datasets` library
# optional can be `None`

dataset_specific_system_prompts = {
    'user/my-ideas-dataset:|:K': 'Dataset_specific_prompt',
    'user/my-ideas-dataset': 'Dataset_specific_prompt', # Both keys are valid and should work
    'user/my-attention-dataset': 'Dataset_specific_prompt'
}
# dataset specific system prompt will get appended to master system prompt: system_prompt+dataset_specific_prompt
# optional can be `None`

# Configure training parameters
# Below are default values for all currently supported user_inputs:
training_config = {
    'run_name': 'my_adaptive_run',
    'wandb_project': 'my-adaptive-experiments',
    'wandb_entity': None
    'max_length_token': 4096,
    'padding_side': 'left',
    'batch_size': 4,
    'learning_rate': 2e-5,
    'gradient_accumulation_steps': 16,
    'num_epochs': 3,
    'attn_impl': 'flash_attention_2',
    'save_total_limit': 5,
    'fp16': True,
    'gradient_checkpointing': True,
    'eval_strategy': 'steps',
    'local_rank': -1,
    'logging_steps':10,
    'weight_decay': 0.01,
    'warmup_steps':10,
    'eval_steps': 200,
    'save_steps': 300,
    'eval_batch_size': 4, # defaults to batch_size
    'optimizer': 'paged_adamw_8bit',
    'use_liger_kernel': False,
    'eval_on_start': False
}

# Configure adaptive loss parameters
adaptive_loss_config = {
    'top_k': 8 # Currently only top_k parameter is allowed for user_control in training
}
# It is expected that a lower value of top_k will lead to stricter learning possibly leading previous learned behaviour
# being forgotten and a higher value of top_k will allow more lienient learning
# top_k=1 can cause ideas learning to behave the same as attention learning

# Run training
model_path = train_adaptively(
    model_name="model_to_train",
    datasets_config=datasets_config,
    datasets_kwargs =datasets_kwargs,
    output_dir="./my_finetuned_model_adaptive",
    huggingface_repo="your_username/my_finetuned_model_adaptive", # Optional
    system_prompts=system_prompts,
    training_config=training_config,
    adaptive_loss_config=adaptive_loss_config
)

print(f"Training complete. Model saved to {model_path}")
```

## Contributing
... (Details on how others can contribute) not yet revealed ...
Open issues as of now

## License
This project is licensed under <License yet to be announced>
