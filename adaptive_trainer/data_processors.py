from datasets import load_dataset, concatenate_datasets
import torch
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Handles dataset preparation, tokenization, and processing for training.
    """
    
    def __init__(self, tokenizer, config, max_length=4096, context_mode=True):
        """
        Initialize the data processor.
        
        Args:
            tokenizer: HuggingFace tokenizer
            config: TokenizerConfig instance with model-specific token settings
            max_length: Maximum sequence length
            context_mode: Whether to use context mode for certain datasets
        """
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = max_length
        self.context_mode = context_mode
        
    def prepare_text_for_training(self, sample, system_prompt):
        """
        Format a sample for training based on its structure.
        
        Args:
            sample: Dataset sample
            system_prompt: System prompt to use
            
        Returns:
            Formatted text string ready for tokenization
        """
        if 'conversations' in sample:
            conversations = sample['conversations']
            user0 = ''
            assistant = ''
            for i in conversations:
                if i['from'] == 'user' and user0 == '':
                    user0 = i['value']
                if i['from'] == 'assistant' and assistant == '':
                    assistant = i['value']
            assistant = assistant.replace('<|begin_of_thought|>','<think>')
            assistant = assistant.replace('<|end_of_thought|>','</think>')
            assistant = assistant.replace('<|begin_of_solution|>','')
            assistant = assistant.replace('<|end_of_solution|>','')
        elif 'query' in sample:
            user0 = sample['query']
            assistant = sample['response']
        elif 'problem' in sample and 'solution' in sample:
            user0 = sample['problem']
            assistant = sample['solution']
        elif 'question' in sample and 'answer' in sample and 'question_with_context' in sample:
            user0 = sample['question_with_context'] if self.context_mode else sample['question']
            assistant = sample['answer']
        elif 'question' in sample and 'answer' in sample:
            user0 = sample['question']
            assistant = sample['answer']
        elif 'prompt' in sample and 'response' in sample:
            user0 = sample['prompt']
            assistant = sample['response'].replace('<|thinking|>', '<think>').replace('</|thinking|>', '</think>').replace('</|actual_response|>', '').replace('<|actual_response|>', '')
        else:
            # Handle other dataset formats or log warning
            logger.warning(f"Unrecognized sample format: {sample.keys()}")
            if 'input' in sample and 'output' in sample:
                user0 = sample['input']
                assistant = sample['output']
            else:
                user0 = str(sample)
                assistant = "Unable to process this example"
        
        return f"{self.config.begin_text_token}{self.config.system_header}{system_prompt}{self.config.end_turn_token}{self.config.user_header}{user0}{self.config.end_turn_token}{self.config.assistant_header}{assistant}{self.config.end_turn_token}{self.config.end_text_token}"
    
    def prepare_dataset(self, dataset_name, learning_style='both', system_prompts=None):
        """
        Load, format, and tokenize a dataset.
        
        Args:
            dataset_name: HuggingFace dataset name, if only k number of items from the dataset are to be used for training you can append :|:k at the end of dataset_name
            learning_style: Training style ('both', 'ideas', 'attention', etc.)
            system_prompts: Dictionary of system prompts by learning style
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        if ':|:' in dataset_name:
            try:
                N = dataset_name.split(':|:')[-1]
                N = int(N)
                dataset_name = ':|:'.join(dataset_name.split(':|:')[:-1])
            except:
                N = 10000000
                pass 
        if system_prompts is None:
            # Default system prompt
            system_prompt = "Please follow the user instructions faithfully."
        else:
            system_prompt = system_prompts.get(learning_style, system_prompts.get('both', 
                            "Please follow the user instructions faithfully."))
        
        logger.info(f"Loading dataset: {dataset_name}")
        dataset = load_dataset(dataset_name)
        
        # Print available splits for debugging
        available_splits = list(dataset.keys())
        logger.info(f"Available splits in the dataset: {available_splits}")
        
        # Choose appropriate splits
        train_split = "train" if "train" in available_splits else available_splits[0]
        val_split = "validation" if "validation" in available_splits else (
            "test" if "test" in available_splits else train_split
        )

        # If validation split is the same as train split, create a separate validation set
        if val_split == train_split:
            split_dataset = dataset[train_split].train_test_split(test_size=0.05, seed=42)
            train_dataset = split_dataset["train"].shuffle(seed=42).select(range(min(N, len(split_dataset["train"]))))
            val_dataset = split_dataset["test"].shuffle(seed=42).select(range(min(round(N*2/3), len(split_dataset["test"]))))
        else:
            train_dataset = dataset[train_split].shuffle(seed=42).select(range(min(N, len(dataset[train_split])))) 
            val_dataset = dataset[val_split].shuffle(seed=42).select(range(min(round(N*2/3), len(dataset[val_split]))))
            
        # Format the prompts
        train_dataset = train_dataset.map(
            lambda sample: {"formatted_text": self.prepare_text_for_training(sample, system_prompt)},
            remove_columns=dataset[train_split].column_names
        )
        
        val_dataset = val_dataset.map(
            lambda sample: {"formatted_text": self.prepare_text_for_training(sample, system_prompt)},
            remove_columns=dataset[val_split].column_names
        )
        
        # Tokenize the datasets
        logger.info("Tokenizing dataset...")
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["formatted_text"],
                truncation=False,
                max_length=self.max_length,
                padding="max_length"
            )

        tokenized_train = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["formatted_text"]
        )
        
        tokenized_val = val_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["formatted_text"]
        )

        # Filter out examples that exceed max_length
        tokenized_train = tokenized_train.filter(lambda example: len(example['input_ids']) <= self.max_length)
        tokenized_val = tokenized_val.filter(lambda example: len(example['input_ids']) <= self.max_length)

        # Add learning style column
        def learning_style_column(example):
            example["learning_style"] = learning_style
            return example

        tokenized_train = tokenized_train.map(learning_style_column)
        tokenized_val = tokenized_val.map(learning_style_column)

        # Set format for PyTorch
        tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "learning_style"])
        tokenized_val.set_format("torch", columns=["input_ids", "attention_mask", "learning_style"])

        return tokenized_train, tokenized_val
    
    def combine_datasets(self, dataset_dict):
        """
        Combine multiple datasets with their corresponding learning styles.
        
        Args:
            dataset_dict: Dictionary mapping dataset names to learning styles and system prompts
                Format: {
                    'ideas': [dataset_names],
                    'attention': [dataset_names],
                    'both': [dataset_names],
                    'misc': [dataset_names],
                    'system_prompts': {
                        'ideas': '...',
                        'attention': '...',
                        'both': '...',
                        'misc': '...'
                    }
                }
                
        Returns:
            Tuple of (combined_train_dataset, combined_val_dataset)
        """
        training_datasets = []
        eval_datasets = []
        system_prompts = dataset_dict.get('system_prompts', {})
        
        # Process each dataset category
        for style, datasets in dataset_dict.items():
            if style == 'system_prompts':
                continue
                
            for dataset_name in datasets:
                train_dataset, eval_dataset = self.prepare_dataset(
                    dataset_name, 
                    learning_style=style,
                    system_prompts=system_prompts
                )
                training_datasets.append(train_dataset)
                eval_datasets.append(eval_dataset)
        
        if not training_datasets:
            raise ValueError("No datasets were successfully processed")
            
        train_dataset = concatenate_datasets(training_datasets).shuffle(seed=42)
        eval_dataset = concatenate_datasets(eval_datasets).shuffle(seed=42)
        
        return train_dataset, eval_dataset
