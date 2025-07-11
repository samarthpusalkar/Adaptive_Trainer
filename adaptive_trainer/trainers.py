import torch
import json
from collections import defaultdict
import logging
from transformers import Trainer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdaptiveTrainer(Trainer):
    """
    Custom trainer that implements adaptive loss for fine-tuning.
    
    This trainer applies different training approaches based on token prediction
    confidence and learning styles, enabling curriculum learning and adaptive-selective
    training on different parts of the input.
    """
    
    def __init__(
        self, 
        *args, 
        top_k=10,
        initial_confidence_threshold=0.7,
        final_confidence_threshold=0.9,
        max_kl_divergence=0.2,
        curriculum_epochs=1.0,
        stats_save_path="./selective_loss_stats.json",
        skip_init_tokens=0,
        output_demarkation_token_ids=None,
        log_metric_steps=100,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.top_k = top_k
        self.initial_confidence_threshold = initial_confidence_threshold
        self.final_confidence_threshold = final_confidence_threshold
        self.max_kl_divergence = max_kl_divergence
        self.curriculum_epochs = curriculum_epochs
        self.stats_save_path = stats_save_path
        self.skip_init_tokens = skip_init_tokens
        self.OUTPUT_DEMARKATION_TOKEN_IDs = torch.tensor(output_demarkation_token_ids) if (output_demarkation_token_ids is not None) and (len(output_demarkation_token_ids) > 0) else None
        self.log_metric_steps = log_metric_steps
        # Statistics tracking
        self.stats = {
            "total_tokens": 0,
            "already_correct": 0,
            "in_top_k": 0,
            "outside_top_k": 0,
            "training_mask": 0,
            "tokens_by_position": defaultdict(lambda: defaultdict(int)),
            "step_count": 0
        }

    def find_sequence_vectorized_2d(self, main_tensor: torch.Tensor, sequence: torch.Tensor) -> torch.Tensor:
        """
        Finds the starting index of the last occurrence of a 1D sequence
        within each row of a 2D main tensor using vectorized operations.

        Args:
            main_tensor: The 2D tensor (shape BxN) to search within, where B is
                        the number of rows (batch size) and N is the length
                        of each row.
            sequence: The 1D sequence tensor (shape M) to find.

        Returns:
            A 1D tensor (shape B) of dtype long, where each element is the
            starting index of the last occurrence of the sequence in the
            corresponding row of main_tensor, or -1 if the sequence is not
            found in that row.
        """
        # Input validation
        if main_tensor.dim() != 2:
            raise ValueError(
                f"main_tensor must be 2D, but got {main_tensor.dim()} dimensions."
            )
        if sequence.dim() != 1:
            raise ValueError(
                f"sequence must be 1D, but got {sequence.dim()} dimensions."
            )

        B, N = main_tensor.shape # Batch size, Row length
        M = sequence.shape[0]   # Sequence length

        # --- Handle Edge Cases ---
        # Case 1: Empty sequence
        if M == 0:
            # An empty sequence is typically considered found at index 0
            return torch.zeros(
                B, dtype=torch.long, device=main_tensor.device
            )

        # Case 2: Sequence is longer than the rows it's searched in
        if M > N:
            # Sequence cannot be found in any row
            return torch.full(
                (B,), -1, dtype=torch.long, device=main_tensor.device
            )

        # --- Main Logic ---
        # Ensure sequence is on the same device as main_tensor
        sequence = sequence.to(main_tensor.device)

        # Create sliding windows for *each row* along dimension 1 (the row length dim)
        # unfold(dimension, size, step)
        # Shape: (B, N - M + 1, M)
        unfolded_windows = main_tensor.unfold(dimension=1, size=M, step=1)

        # Compare each window with the sequence using broadcasting.
        # sequence (M,) broadcasts against the last dim of unfolded_windows (B, N-M+1, M)
        # Shape: (B, N - M + 1, M)
        comparison = unfolded_windows == sequence

        # Check if *all* elements within each window match the sequence.
        # We check along the last dimension (dim=2), which corresponds to the sequence length M.
        # Shape: (B, N - M + 1)
        # row_matches[i, j] is True if row 'i' has a match starting at index 'j'
        row_matches = torch.all(comparison, dim=2)

        # Find the index of the *last* match in each row.
        # We use argmax. Since True casts to 1 and False to 0, argmax finds the
        # index of the last True. If a row has no True values, argmax returns 0.
        # Shape: (B,)
        row_matches_reversed = torch.flip(row_matches, dims=[1])
        # 2. Reverse each row
        last_match_indices_from_right = torch.argmax(row_matches_reversed.int(), dim=1)
        last_match_indices = row_matches_reversed.shape[1] - last_match_indices_from_right - 1

        # Distinguish rows with no matches at all from rows where the match starts at index 0.
        # Check if *any* match occurred in each row.
        # Shape: (B,)
        any_match_in_row = torch.any(row_matches, dim=1)

        # Create the final result tensor.
        # If any_match_in_row is True, use the index found by argmax.
        # Otherwise (no match in the row), set the index to -1.
        result = torch.where(
            any_match_in_row,
            last_match_indices,
            torch.tensor(-1, device=main_tensor.device, dtype=torch.long),
        )

        return result

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        """
        Override compute_loss to implement selective loss
        """
        # Get current confidence threshold based on curriculum learning
        progress = min(1.0, self.state.epoch / self.curriculum_epochs)
        current_threshold = self.initial_confidence_threshold + progress * (
            self.final_confidence_threshold - self.initial_confidence_threshold
        )
        learning_styles = inputs.pop("learning_style", ["both"] * len(inputs.get("input_ids", [])))
        
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Get labels
        labels = inputs.get("labels")
        
        # Create tensor for gathering statistics by position
        positions = None#torch.arange(labels.size(1), device=labels.device).expand_as(labels)
        
        # Apply selective loss
        loss, loss_stats = self._compute_adaptive_loss(
            logits, 
            labels,
            positions,
            top_k=self.top_k,
            confidence_threshold=current_threshold,
            learning_styles=learning_styles
        )
        
        # Update statistics
        self._update_stats(loss_stats)
        
        # Save statistics periodically
        self.stats["step_count"] += 1
        if self.stats["step_count"] % self.log_metric_steps == 0:
            self._save_stats()
        
        return (loss, outputs) if return_outputs else loss

    def _compute_adaptive_loss(
        self, 
        logits, 
        labels, 
        positions,
        top_k=10, 
        confidence_threshold=0.8,
        learning_styles=None
    ):
        """
        Modified version of selective loss computation.
        Splits tokens into:
        - Case 1: Tokens where expected label is in top-k (skip training)
        - Case 2: Tokens where expected label is outside top-k (train these tokens)
        """
        device = logits.device
        batch_size, seq_len, vocab_size = logits.shape

        # Compute top-k indices along the vocab dimension.
        labels[:,:-1] = labels[:,1:].clone()
        labels[:,-1] = -100
        if self.OUTPUT_DEMARKATION_TOKEN_IDs is not None and len(self.OUTPUT_DEMARKATION_TOKEN_IDs):
            x = self.find_sequence_vectorized_2d(labels, self.OUTPUT_DEMARKATION_TOKEN_IDs)
            for idx, sample_idx in enumerate(x):
                if sample_idx >= 0:  # Only if sequence was found
                    labels[idx, :sample_idx] = -100

        # Valid tokens mask (labels != -100)
        valid_mask = labels != -100
        # Compute predictions
        pred_tokens = logits.argmax(dim=-1)  # shape: (B, S)
        
        # Check if expected token is in the top-k predictions
        _, top_k_indices = torch.topk(logits, top_k, dim=-1)
        expected_in_top_k = (labels.unsqueeze(-1) == top_k_indices).any(dim=-1)
        # Identify tokens NOT in top-k (to be trained)
        # Train for meta concepts
        training_mask = (~expected_in_top_k) & valid_mask & (~(pred_tokens == labels))
        _, top_1_indices = torch.topk(logits, 1, dim=-1)
        expected_in_top_1 = (labels.unsqueeze(-1) == top_1_indices).any(dim=-1)
        # Train for language alignment
        training_mask_coherence = (~expected_in_top_1) & valid_mask & (~(pred_tokens == labels))
        training_mask_correct_prediction = valid_mask & (pred_tokens == labels)
        
        # Initialize loss mask
        loss_mask_ideas = torch.zeros_like(labels, dtype=torch.bool)
        loss_mask_ideas |= training_mask
        loss_mask_coherence = torch.zeros_like(labels, dtype=torch.bool)
        loss_mask_coherence |= training_mask_coherence
        loss_mask_correct_prediction = torch.zeros_like(labels, dtype=torch.bool)
        loss_mask_correct_prediction |= training_mask_correct_prediction

        # Initialize statistics
        stats = {
            "total_tokens": valid_mask.sum().item(),
            "already_correct": ((pred_tokens == labels) & valid_mask).sum().item(),
            "in_top_k": (expected_in_top_k & valid_mask).sum().item(),
            "outside_top_k": ((~expected_in_top_k) & valid_mask).sum().item(),
            "training_mask": training_mask.sum().item(),
            "tokens_by_position": defaultdict(lambda: defaultdict(int))
        }

        # Compute cross-entropy loss only for tokens selected by loss_mask
        loss_fct = torch.nn.CrossEntropyLoss()
        
        flat_logits = logits.view(-1, vocab_size)

        flat_labels = labels.view(-1).clone()
        flat_labels_ideas = labels.view(-1).clone()
        flat_labels_coherence = labels.view(-1).clone()
        flat_valid_mask = flat_labels != -100

        if learning_styles is None:
            learning_styles = ["both"]*len(labels)
        if learning_styles is not None:
            device = labels.device  # Get the device from an existing tensor like labels or logits
            ones = torch.ones_like(labels[0:1], dtype=torch.bool, device=device)
            style_is_both_1d = torch.tensor([s == 'both' for s in learning_styles], dtype=torch.bool, device=device)
            style_is_ideas_1d = torch.tensor([s == 'ideas' for s in learning_styles], dtype=torch.bool, device=device)
            style_is_attention_1d = torch.tensor([s == 'attention' for s in learning_styles], dtype=torch.bool, device=device)
            learn_style_mask_both = style_is_both_1d.unsqueeze(-1) * ones
            learn_style_mask_ideas = style_is_ideas_1d.unsqueeze(-1) * ones
            learn_style_mask_attention = style_is_attention_1d.unsqueeze(-1) * ones
        else:
            loss = loss_fct(flat_logits, flat_labels)
            return loss, stats

        flat_learn_style_mask_both = learn_style_mask_both.view(-1)
        flat_learn_style_mask_ideas = learn_style_mask_ideas.view(-1)
        flat_learn_style_mask_attention = learn_style_mask_attention.view(-1)
        flat_learn_style_mask_ideas = flat_learn_style_mask_ideas | flat_learn_style_mask_both
        flat_learn_style_mask_attention = flat_learn_style_mask_attention | flat_learn_style_mask_both

        flat_pred_tokens = pred_tokens.view(-1).clone()
        flat_pred_tokens_correct_prediction = pred_tokens.view(-1).clone()
        
        flat_loss_mask_ideas = loss_mask_ideas.view(-1)
        flat_loss_mask_coherence = loss_mask_coherence.view(-1)
        flat_loss_mask_correct_prediction = loss_mask_correct_prediction.view(-1)

        valid_loss_mask_ideas = (flat_labels != -100) & flat_loss_mask_ideas
        valid_loss_mask_coherence = (flat_labels != -100) & flat_loss_mask_coherence
        valid_loss_mask_correct_prediction = (flat_labels != -100) & flat_loss_mask_correct_prediction

        flat_labels_ideas[~(valid_loss_mask_ideas)] = -100
        flat_labels_coherence[~(valid_loss_mask_coherence)] = -100

        flat_pred_tokens[~(flat_labels != -100)] = -100
        flat_pred_tokens_correct_prediction[~(valid_loss_mask_correct_prediction)] = -100

        self_confidence_loss = loss_fct(flat_logits, flat_pred_tokens_correct_prediction)

        common_language_continuation_loss = loss_fct(flat_logits, flat_labels) # this loss (`should`) handle deviation from language continuation and prompted task

        if valid_loss_mask_ideas.sum().item() == 0:
            flat_pred_tokens_temp = flat_pred_tokens.clone()
            flat_pred_tokens_temp[~(flat_learn_style_mask_ideas)] = -100
            ideas_learning_loss = loss_fct(flat_logits, flat_pred_tokens_temp) + self_confidence_loss/4
        else:
            flat_labels_ideas_temp = flat_labels_ideas.clone()
            if (~(flat_learn_style_mask_ideas)).all():
                ideas_learning_loss = self_confidence_loss/4
            else:
                flat_labels_ideas_temp[~(flat_learn_style_mask_ideas)] = -100
                ideas_learning_loss = loss_fct(flat_logits, flat_labels_ideas_temp) + self_confidence_loss/4

        if valid_loss_mask_coherence.sum().item() == 0:
            flat_pred_tokens_temp = flat_pred_tokens.clone()
            flat_pred_tokens_temp[~(flat_learn_style_mask_attention)] = -100
            attention_learning_loss = loss_fct(flat_logits, flat_pred_tokens_temp) + self_confidence_loss/4
        else:
            flat_labels_coherence_temp = flat_labels_coherence.clone()
            if (~(flat_learn_style_mask_attention)).all():
                attention_learning_loss = self_confidence_loss/4
            else:
                flat_labels_coherence_temp[~(flat_learn_style_mask_attention)] = -100
                attention_learning_loss = loss_fct(flat_logits, flat_labels_coherence_temp) + self_confidence_loss/4

        # Recalculating proper training tokens
        stats['training_mask'] = ((flat_learn_style_mask_attention & valid_loss_mask_coherence) & flat_valid_mask).sum().item() + ((flat_learn_style_mask_ideas & valid_loss_mask_ideas) & flat_valid_mask).sum().item() - (((flat_learn_style_mask_both & valid_loss_mask_coherence) & valid_loss_mask_ideas) & flat_valid_mask).sum().item()
        alpha_attention_bias = torch.tensor(1.0, requires_grad=True)
        loss = (attention_learning_loss*alpha_attention_bias + ideas_learning_loss*(2-alpha_attention_bias) + common_language_continuation_loss*(alpha_attention_bias+0.5)/2)**2

        return loss, stats

    def _update_stats(self, batch_stats):
        """Update global statistics with batch statistics"""
        for key in ['total_tokens', 'already_correct', 'in_top_k', 
                'outside_top_k', 'training_mask']:
            self.stats[key] += batch_stats[key]
        
        # Update position-specific stats
        for pos, pos_stats in batch_stats["tokens_by_position"].items():
            for stat_name, count in pos_stats.items():
                self.stats["tokens_by_position"][pos][stat_name] += count

    def _save_stats(self):
        """Save statistics to file"""
        # Convert defaultdict to regular dict for JSON serialization
        serializable_stats = {
            k: (dict(v) if isinstance(v, defaultdict) else v)
            for k, v in self.stats.items()
        }
        serializable_stats["tokens_by_position"] = {
            k: dict(v) for k, v in serializable_stats["tokens_by_position"].items()
        }
        
        with open(self.stats_save_path, 'a') as f:
            f.write(',\n')
            json.dump(serializable_stats, f, indent=2)
        
        # Log summary to console
        total = self.stats["total_tokens"]
        if total > 0:
            logger.info(f"Adaptive Loss Statistics (after {self.stats['step_count']} steps):")
            logger.info(f"  Already correct:       \t{self.stats['already_correct'] / total:.2%}")
            logger.info(f"  From top-k:            \t{self.stats['in_top_k'] / total:.2%}")
            logger.info(f"  Outside top-k:         \t{self.stats['outside_top_k'] / total:.2%}")
            logger.info(f"  Total Trained in loop: \t{self.stats['training_mask'] / total:.2%}")
            for key in ['total_tokens', 'already_correct', 'in_top_k', 
                'outside_top_k', 'training_mask']:
                self.stats[key] = 0
            
            # Reseting stats for better results tracking.
            self.stats = {
                "total_tokens": 0,
                "already_correct": 0,
                "in_top_k": 0,
                "outside_top_k": 0,
                "training_mask": 0,
                "tokens_by_position": defaultdict(lambda: defaultdict(int)),
                "step_count": self.stats['step_count']
            }
