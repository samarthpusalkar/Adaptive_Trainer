# Adaptive Trainer

A Hugging Face `Trainer` extension designed for more efficient and effective fine-tuning of language models using adaptive loss mechanisms and curriculum learning principles.

## Features

*   **Adaptive Loss Calculation:** Focuses training on tokens where the model is less confident or incorrect, rather than all tokens equally.
*   **Ideas Learning:** Allows model to learn concepts and behaviour from training data rather than actual memorisation much like student learning.
*   **Learning Style Differentiation:** Also allows specific datasets to be learnt with attention and concepts with ideas.Allows for different training objectives (e.g., "ideas" vs. "attention") based on dataset characteristics.
*   **Integration with Hugging Face:** Built on top of the `transformers` library and `Trainer` API.
*   **More Features** like parameter control and hyperparameter optimisation of loss for user comming soon


## Not Implemented yet: 
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
    'ideas': ["user/my-ideas-dataset"],
    'attention': ["user/my-attention-dataset"],
}

# Configure training parameters
training_config = {
    'max_length_token': 2048,
    'batch_size': 2,
    'gradient_accumulation_steps': 8,
    'learning_rate': 5e-6,
    'num_epochs': 3,
    'attn_impl': 'sdpa', # Scaled Dot Product Attention (PyTorch 2.0+) or 'flash_attention_2' if installed
    'run_name': 'my_adaptive_run',
    'wandb_project': 'my-adaptive-experiments'
}

# Configure adaptive loss parameters
adaptive_loss_config = {
    'top_k': 8,
    'initial_confidence': 0.65,
    'final_confidence': 0.85,
    'curriculum_epochs': 1.0
}

# Run training
model_path = train_adaptively(
    model_name="mistralai/Mistral-7B-Instruct-v0.1",
    datasets_config=datasets_config,
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

## License
This project is licensed under <License yet to be announced>

