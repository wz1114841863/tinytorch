import numpy as np
import copy
from typing import List, Dict, Any, Tuple, Optional
import time

from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear, Sequential
from tinytorch.core.activations import ReLU

BYTES_PER_FLOAT32 = 4  # Standard float32 size in bytes
MB_TO_BYTES = 1024 * 1024  # Megabytes to bytes conversion


def measure_sparsity(model):
    """Calculate the sparsity of the model's parameters."""
    total_params = 0
    zero_params = 0

    for param in model.parameters():
        # Only count weight matrices (2D), not biases (1D)
        # Biases are often initialized to zero, which would skew sparsity
        if len(param.shape) > 1:
            total_params += param.size
            zero_params += np.sum(param.data == 0)

    if total_params == 0:
        return 0.0

    return (zero_params / total_params) * 100.0


def magnitude_prune(model, sparsity=0.9):
    """Remove weights with the smallest magnitudes to achieve the desired sparsity."""
    all_weights = []
    weight_params = []

    for param in model.parameters():
        if len(param.shape) > 1:  # Only consider weight matrices
            all_weights.extend(np.abs(param.data).flatten())
            weight_params.append(param)

    if not all_weights:
        return model

    magnitudes = np.abs(all_weights)
    threshold = np.percentile(magnitudes, sparsity * 100)

    for param in weight_params:
        mask = np.abs(param.data) >= threshold
        param.data = param.data * mask  # Zero out weights below the threshold

    return model


def structured_prune(model, prune_ratio=0.5):
    """Remove entire channels based on L2 norm importance."""
    for layer in model.layers:
        if isinstance(layer, Linear):
            weight = layer.weight.data
            # Channel Importance Metrics:L2 norm of each output channel
            channel_norms = np.linalg.norm(weight, axis=0)
            num_channels = weight.shape[1]
            num_to_prune = int(num_channels * prune_ratio)

            if num_to_prune > 0:
                # Get indices of channels to prune (smallest norms)
                prune_indices = np.argpartition(channel_norms, num_to_prune)[
                    :num_to_prune
                ]
                weight[:, prune_indices] = 0  # Zero out entire channels
                if layer.bias is not None:
                    layer.bias.data[prune_indices] = 0  # Zero out corresponding biases

    return model


def low_rank_approximate(weight_matrix, rank_ratio=0.5):
    """Approximate the weight matrix with a lower rank using SVD."""
    m, n = weight_matrix.shape
    U, S, V = np.linalg.svd(weight_matrix, full_matrices=False)
    max_rank = min(m, n)
    target_rank = max(1, int(rank_ratio * max_rank))

    U_truncated = U[:, :target_rank]
    S_truncated = S[:target_rank]
    V_truncated = V[:target_rank, :]

    return U_truncated, S_truncated, V_truncated


class KnowledgeDistillation:
    """Knowledge distillation for model compression."""

    def __init__(self, teacher_model, student_model, temperature=3.0, alpha=0.5):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha

    def distillation_loss(self, student_logits, teacher_logits, true_labels):
        """Calculate the distillation loss."""
        # Soft targets from teacher
        student_logits = student_logits.data
        teacher_logits = teacher_logits.data

        # true_labels might be numpy array or Tensor
        if isinstance(true_labels, Tensor):
            true_labels = true_labels.data

        # Soften distributions with temperature
        student_soft = self._softmax(student_logits / self.temperature)
        teacher_soft = self._softmax(teacher_logits / self.temperature)

        # Soft target loss (KL divergence)
        soft_loss = self._kl_divergence(student_soft, teacher_soft)

        # Hard target loss (cross-entropy)
        student_hard = self._softmax(student_logits)
        hard_loss = self._cross_entropy(student_hard, true_labels)

        # Combined loss
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss

        return total_loss

    def _softmax(self, logits):
        """Compute softmax with numerical stability."""
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    def _kl_divergence(self, p, q):
        """Compute KL divergence between distributions."""
        return np.sum(p * np.log(p / (q + 1e-8) + 1e-8))

    def _cross_entropy(self, predictions, labels):
        """Compute cross-entropy loss."""
        # Simple implementation for integer labels
        if labels.ndim == 1:
            return -np.mean(np.log(predictions[np.arange(len(labels)), labels] + 1e-8))
        else:
            return -np.mean(np.sum(labels * np.log(predictions + 1e-8), axis=1))


def compress_model(model, compression_config):
    """Apply comprehensive model compression based on configuration."""
    original_params = sum(p.size for p in model.parameters())
    original_sparsity = measure_sparsity(model)

    stats = {
        "original_params": original_params,
        "original_sparsity": original_sparsity,
        "applied_techniques": [],
    }

    # Apply magnitude pruning
    if "magnitude_prune" in compression_config:
        sparsity = compression_config["magnitude_prune"]
        magnitude_prune(model, sparsity=sparsity)
        stats["applied_techniques"].append(f"magnitude_prune_{sparsity}")

    # Apply structured pruning
    if "structured_prune" in compression_config:
        ratio = compression_config["structured_prune"]
        structured_prune(model, prune_ratio=ratio)
        stats["applied_techniques"].append(f"structured_prune_{ratio}")

    # Apply low-rank approximation (conceptually - would need architecture changes)
    if "low_rank" in compression_config:
        ratio = compression_config["low_rank"]
        # For demo, we'll just record that it would be applied
        stats["applied_techniques"].append(f"low_rank_{ratio}")

    # Final measurements
    final_sparsity = measure_sparsity(model)
    stats["final_sparsity"] = final_sparsity
    stats["sparsity_increase"] = final_sparsity - original_sparsity

    return stats


# | export
class Compressor:
    """Complete compression system for milestone use."""

    @staticmethod
    def measure_sparsity(model) -> float:
        """Measure the sparsity of a model (returns fraction 0-1)."""
        return measure_sparsity(model) / 100.0

    @staticmethod
    def magnitude_prune(model, sparsity=0.5):
        """Prune model weights by magnitude. Delegates to standalone function."""
        return magnitude_prune(model, sparsity)

    @staticmethod
    def structured_prune(model, prune_ratio=0.5):
        """Prune entire neurons/channels. Delegates to standalone function."""
        return structured_prune(model, prune_ratio)

    @staticmethod
    def compress_model(model, compression_config: Dict[str, Any]):
        """
        Apply complete compression pipeline to a model.

        Args:
            model: Model to compress
            compression_config: Dictionary with compression settings
                - 'magnitude_sparsity': float (0-1)
                - 'structured_prune_ratio': float (0-1)

        Returns:
            Compressed model with sparsity stats (fractions 0-1)
        """
        stats = {"original_sparsity": Compressor.measure_sparsity(model)}

        # Apply magnitude pruning
        if "magnitude_sparsity" in compression_config:
            model = Compressor.magnitude_prune(
                model, compression_config["magnitude_sparsity"]
            )

        # Apply structured pruning
        if "structured_prune_ratio" in compression_config:
            model = Compressor.structured_prune(
                model, compression_config["structured_prune_ratio"]
            )

        stats["final_sparsity"] = Compressor.measure_sparsity(model)
        stats["compression_ratio"] = (
            1.0 / (1.0 - stats["final_sparsity"])
            if stats["final_sparsity"] < 1.0
            else float("inf")
        )

        return model, stats
