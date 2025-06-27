class TokenizerConfig:
    """Configuration for model-specific tokenization."""
    
    def __init__(self):
        self.begin_text_token = ""
        self.system_header = "<|start_header_id|>system<|end_header_id|>"
        self.user_header = "<|start_header_id|>user<|end_header_id|>"
        self.assistant_header = "<|start_header_id|>assistant<|end_header_id|>"
        self.end_turn_token = "<|eot_id|>"
        self.end_text_token = ""
        self.chat_template = ""
    
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
    def from_model_name(cls, model_name, tokenizer):
        """Create configuration based on model name."""
        config = cls()
        model_name_lower = model_name.lower()
        
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
            pass
        elif "llama" in model_name_lower:
            config = config.load_llama_config()
        elif "phi" in model_name_lower:
            config = config.load_phi_config()
        elif "qwen" in model_name_lower:
            config = config.load_qwen_config()
        elif "smollm" in model_name_lower:
            config = config.load_smol_config()
        
        if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
            config.chat_template = """{{%- if messages[0]['role'] == 'system' %}}
    {{%- set system_message = messages[0]['content'] %}}
    {{%- set loop_messages = messages[1:] %}}
{{%- else %}}
    {{%- set loop_messages = messages %}}
{{%- endif %}}

{{{{- bos_token }}}}
{{%- for message in loop_messages %}}
    {{%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}}
        {{{{- raise_exception('After the optional system message, conversation roles must alternate user/assistant/user/assistant/...') }}}}
    {{%- endif %}}
    {{%- if message['role'] == 'user' %}}
        {{%- if loop.first and system_message is defined %}}
            {{{{- '{system_header}' + system_message + '{end_turn_token}\n\n{user_header}' + message['content'] + '{end_turn_token}' }}}}
        {{%- else %}}
            {{{{- '{user_header}' + message['content'] + '{end_turn_token}' }}}}
        {{%- endif %}}
    {{%- elif message['role'] == 'assistant' %}}
        {{{{- '{assistant_header}' + message['content'] + '{end_turn_token}' + eos_token}}}}
    {{%- else %}}
        {{{{- raise_exception('Only user and assistant roles are supported, with the exception of an initial optional system message!') }}}}
    {{%- endif %}}
{{%- endfor %}}
""".format(
            system_header=config.system_header,
            user_header=config.user_header,
            assistant_header=config.assistant_header,
            end_turn_token=config.end_turn_token
        )
        else:
            config.chat_template = tokenizer.chat_template
            config.system_header = tokenizer.apply_chat_template([{'role':'system', 'content':'|SEP|'}], tokenize=False).split('|SEP|')[0].replace(tokenizer.bos_token, '').replace(tokenizer.eos_token, '')
            config.user_header = tokenizer.apply_chat_template([{'role':'user', 'content':'|SEP|'}], tokenize=False).split('|SEP|')[0].replace(tokenizer.bos_token, '').replace(tokenizer.eos_token, '')
            user, assistant = tokenizer.apply_chat_template([{'role':'user', 'content':'|USER|'}, {'role':'assistant', 'content':'|SEP|'}], tokenize=False).replace(tokenizer.bos_token, '').replace(tokenizer.eos_token, '').split('|USER|')
            assistant_header, eot_token = assistant.split('|SEP|')
            config.assistant_header = assistant_header[len(eot_token):]
            config.end_turn_token = eot_token
            config.begin_text_token = tokenizer.bos_token
            config.end_text_token = tokenizer.eos_token

        return config, config.chat_template
