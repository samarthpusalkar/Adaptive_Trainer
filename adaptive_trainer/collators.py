from transformers import DataCollatorForLanguageModeling

class CustomDataCollator(DataCollatorForLanguageModeling):
    """
    Custom data collator that handles learning style information in addition
    to the standard language modeling collation.
    """
    
    def __call__(self, features, return_tensors=None):
        # Extract learning styles before passing to parent class
        learning_styles = [feature.get('learning_style', 'both') for feature in features]
        
        # Clean features for parent class
        cleaned_features = [
            {'input_ids': feature['input_ids'], 'attention_mask': feature['attention_mask']} 
            for feature in features
        ]
        
        # Call parent class implementation
        batch = super().__call__(cleaned_features, return_tensors=return_tensors)
        
        # Add learning styles back to batch
        batch['learning_style'] = learning_styles
        return batch
