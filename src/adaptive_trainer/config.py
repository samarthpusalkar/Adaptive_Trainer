class TokenizerConfig:
    """Configuration for model-specific tokenization."""
    
    def __init__(self):
        self.begin_text_token = ""
        self.system_header = "<|start_header_id|>system<|end_header_id|>"
        self.user_header = "<|start_header_id|>user<|end_header_id|>"
        self.assistant_header = "<|start_header_id|>assistant<|end_header_id|>"
        self.end_turn_token = "<|eot_id|>"
        self.end_text_token = ""
    
    def load_llama_config(self):
        """Load LLaMA-specific tokenizer configuration."""
        self.begin_text_token = "<|begin_of_text|>"
        self.system_header = "<|start_header_id|>system<|end_header_id|>"
        self.user_header = "<|start_header_id|>user<|end_header_id|>"
        self.assistant_header = "<|start_header_id|>assistant<|end_header_id|>"
        self.end_turn_token = "<|eot_id|>"
        self.end_text_token = ""
        return self
    
    def load_phi_config(self):
        """Load Phi-specific tokenizer configuration."""
        self.begin_text_token = ""
        self.system_header = "<|system|>\n"
        self.user_header = "\n<|user|>\n"
        self.assistant_header = "\n<|assistant|>\n"
        self.end_turn_token = "<|end|>"
        self.end_text_token = ""
        return self
    
    def load_qwen_config(self):
        """Load Qwen-specific tokenizer configuration."""
        self.begin_text_token = ""
        self.system_header = "<|im_start|>system\n"
        self.user_header = "<|im_start|>user\n"
        self.assistant_header = "<|im_start|>assistant\n"
        self.end_turn_token = "<|im_end|>\n"
        self.end_text_token = ""
        return self
    
    def load_smol_config(self):
        """Load SmolLM-specific tokenizer configuration."""
        self.begin_text_token = ""
        self.system_header = "<|im_start|>system\n"
        self.user_header = "<|im_start|>user\n"
        self.assistant_header = "<|im_start|>assistant\n"
        self.end_turn_token = "<|im_end|>\n"
        self.end_text_token = ""
        return self

    @classmethod
    def from_model_name(cls, model_name):
        """Create configuration based on model name."""
        config = cls()
        model_name_lower = model_name.lower()
        
        if "llama" in model_name_lower:
            return config.load_llama_config()
        elif "phi" in model_name_lower:
            return config.load_phi_config()
        elif "qwen" in model_name_lower:
            return config.load_qwen_config()
        elif "smollm" in model_name_lower:
            return config.load_smol_config()
        
        # Default to LLaMA format if no match
        return config
