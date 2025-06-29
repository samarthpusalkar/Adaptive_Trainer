from datasets import load_dataset, concatenate_datasets
import torch
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Handles dataset preparation, tokenization, and processing for training.
    """
    
    def __init__(self, tokenizer, config, max_length=4096):
        """
        Initialize the data processor.
        
        Args:
            tokenizer: HuggingFace tokenizer
            config: TokenizerConfig instance with model-specific token settings
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = max_length

    def prepare_text_for_training(self, sample, system_prompt, context_mode=False, dataset_preprocessing_function=None):
        """
        Format a sample for training based on its structure.
        
        Args:
            sample: Dataset sample
            system_prompt: System prompt to use
            
        Returns:
            Formatted text string ready for tokenization
        """
        if dataset_preprocessing_function is not None:
            user0, assistant = dataset_preprocessing_function(sample, context_mode=context_mode)
        elif 'conversations' in sample:
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
        elif 'prompt' in sample and 'target_price_continuation' in sample:
            user0 = sample['prompt']
            assistant = sample['target_price_continuation']
        elif 'problem' in sample and 'solution' in sample:
            user0 = sample['problem']
            assistant = sample['solution']
        elif 'question' in sample and 'answer' in sample and 'question_with_context' in sample:
            user0 = sample['question_with_context'] if context_mode else sample['question']
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
                raise 'Unable to autoprocess dataset please pass the data_processing_function in datasets_config.'

        return f"{self.config.begin_text_token}{self.config.system_header}{system_prompt}{self.config.end_turn_token}{self.config.user_header}{user0}{self.config.end_turn_token}{self.config.assistant_header}{assistant}{self.config.end_turn_token}{self.config.end_text_token}"

    def prepare_dataset(self, dataset_name, dataset_kwargs, dataset_specific_prompt, learning_style='both', dataset_preprocessing_function=None, system_prompts=None):
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
        else:
            N = 10000000
            
        if system_prompts is None:
            # Default system prompt
            system_prompt = "" +  dataset_specific_prompt
        else:
            system_prompt = system_prompts.get(learning_style, '') + dataset_specific_prompt

        context_mode = dataset_kwargs.pop("context_mode", False) if dataset_kwargs is not None else False
        train_split = dataset_kwargs.pop("train_split", "train")    if dataset_kwargs is not None else "train"
        val_split   = dataset_kwargs.pop("val_split", "validation") if dataset_kwargs is not None else "validation"
        train_split 
        if context_mode in ["attention", "ideas"]:
            context_mode = (learning_style == context_mode)
        logger.info(f"Loading dataset: {dataset_name}")
        if dataset_kwargs is not None and type(dataset_kwargs)==dict:
            dataset = load_dataset(dataset_name, **dataset_kwargs)
        else:
            dataset = load_dataset(dataset_name)

        # Print available splits for debugging
        available_splits = list(dataset.keys())
        logger.info(f"Available splits in the dataset: {available_splits}")
        
        # Choose appropriate splits
        train_split = train_split if train_split in available_splits else available_splits[0]
        val_split = val_split if val_split in available_splits else (
            "test" if "test" in available_splits else train_split
        )

        # If validation split is the same as train split, create a separate validation set
        if val_split == train_split:
            split_dataset = dataset[train_split].train_test_split(test_size=0.05, seed=42)
            train_dataset = split_dataset["train"].shuffle(seed=42).select(range(min(N, len(split_dataset["train"]))))
            val_dataset = split_dataset["test"].shuffle(seed=42).select(   range(min(N, len(split_dataset["test"]))))
        else:
            train_dataset = dataset[train_split].shuffle(seed=42).select(range(min(N, len(dataset[train_split])))) 
            val_dataset   = dataset[val_split].shuffle(seed=42).select(  range(min(N, len(dataset[val_split]))))

        # Format the prompts
        train_dataset = train_dataset.map(
            lambda sample: {"formatted_text": self.prepare_text_for_training(sample, system_prompt, context_mode, dataset_preprocessing_function)},
            remove_columns=dataset[train_split].column_names
        )

        val_dataset = val_dataset.map(
            lambda sample: {"formatted_text": self.prepare_text_for_training(sample, system_prompt, context_mode, dataset_preprocessing_function)},
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

        len_val_set = max(round((len(tokenized_val)/len(tokenized_train))*N), 10) if len(tokenized_val)>10 else len(tokenized_val)
        tokenized_train = tokenized_train.map(learning_style_column).select(range(N))
        tokenized_val = tokenized_val.map(learning_style_column).select(range(len_val_set))

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
        datasets_kwargs = dataset_dict.get('datasets_kwargs', {})
        dataset_specific_system_prompts = dataset_dict.get('dataset_specific_system_prompts', {})

        # Process each dataset category
        for style, datasets in dataset_dict.items():
            if style not in ['ideas', 'attention', 'both', 'misc']:
                continue

            for dataset_name in datasets:
                dataset_preprocessing_function = dataset_dict.get(f'data_processing_function_{dataset_name}',dataset_dict.get(f'data_processing_function_{style}', dataset_dict.get('data_processing_function', None)))
                dataset_kwargs = datasets_kwargs.get(dataset_name, None)
                dataset_specific_prompt = dataset_specific_system_prompts.get(dataset_name, '')
                if (dataset_kwargs is None) and (":|:" in dataset_name):
                    dataset_kwargs = datasets_kwargs.get(":|:".join(dataset_name.split(":|:")[:-1]), None)
                if (dataset_specific_prompt=='') and (":|:" in dataset_name):
                    dataset_specific_prompt = dataset_specific_system_prompts.get(":|:".join(dataset_name.split(":|:")[:-1]), '')
                    dataset_preprocessing_function = dataset_dict.get(f'data_processing_function_{dataset_name}',dataset_dict.get(f'data_processing_function_{style}', dataset_dict.get('data_processing_function', None))) 
                train_dataset, eval_dataset = self.prepare_dataset(
                    dataset_name,
                    dataset_kwargs.copy() if dataset_kwargs is not None else None,
                    dataset_specific_prompt,
                    learning_style=style,
                    dataset_preprocessing_function=dataset_preprocessing_function,
                    system_prompts=system_prompts
                )
                training_datasets.append(train_dataset)
                eval_datasets.append(eval_dataset)
        
        if not training_datasets:
            raise ValueError("No datasets were successfully processed")
            
        train_dataset = concatenate_datasets(training_datasets).shuffle(seed=42)
        eval_dataset = concatenate_datasets(eval_datasets).shuffle(seed=42)
        
        return train_dataset, eval_dataset
