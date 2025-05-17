import os
import torch
import logging
from huggingface_hub import login, HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

def upload_model_to_huggingface(
    model_path,
    hf_model_name,
    model_class=AutoModelForCausalLM,
    token=None,
    private=False,
    commit_message="Upload fine-tuned model",
    base_model_name=None
):
    """
    Load a local fine-tuned model and upload it to Hugging Face Hub
    
    Args:
        model_path: Path to the fine-tuned model directory
        hf_model_name: Name for the model on Hugging Face Hub (username/model_name)
        model_class: Transformers model class to use for loading
        token: Hugging Face API token
        private: Whether the model should be private
        commit_message: Commit message for the upload
        base_model_name: Name of the base model (for tokenizer if needed)
    
    Returns:
        URL of the uploaded model on Hugging Face Hub
    """
    # Authenticate with Hugging Face
    if token:
        login(token)
    else:
        logger.info("No token provided. Make sure you're logged in with `huggingface-cli login`")
    
    # Use tokenizer from base model if specified
    if base_model_name:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        # Save tokenizer to model path (will be uploaded with model)
        tokenizer.save_pretrained(model_path, push_to_hub=True, repo_id=hf_model_name)
    
    # Upload to Hugging Face Hub
    logger.info(f"Uploading model to Hugging Face Hub as {hf_model_name}")
    api = HfApi()
    api.create_repo(
        repo_id=hf_model_name,
        private=private,
        exist_ok=True,
    )
    
    api.upload_folder(
        folder_path=model_path,
        repo_id=hf_model_name,
        commit_message=commit_message,
    )
    
    logger.info(f"Model successfully uploaded to https://huggingface.co/{hf_model_name}")
    return f"https://huggingface.co/{hf_model_name}"

def setup_environment():
    """
    Configure the environment for training (logging, CUDA, etc.)
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Check for CUDA
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        logger.info(f"Using {n_gpus} GPU(s)")
        
        # Log GPU info
        for i in range(n_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            logger.info(f"  GPU {i}: {gpu_name} with {gpu_mem:.2f} GB memory")
    else:
        logger.info("CUDA not available. Using CPU.")

def cleanup_memory():
    """
    Clean up memory after training
    """
    import gc
    gc.collect()
    if torch.cuda.is_available():
        logger.info("Emptying CUDA cache...")
        torch.cuda.empty_cache()
