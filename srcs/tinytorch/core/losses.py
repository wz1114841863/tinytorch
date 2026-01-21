import numpy as np

from typing import Optional
from .tensor import Tensor
from .activations import ReLU
from .layers import Linear

__all__ = [
    "EPSILON",
    "log_softmax",
    "MSELoss",
    "CrossEntropyLoss",
    "BinaryCrossEntropyLoss",
]

# Constants for numerical stability
EPSILON = 1e-7  # Small value to prevent log(0) and numerical instability


def log_softmax(x, dim: int = -1):
    """Compute log-softmax with numerical stability.

    EXAMPLE:
    >>> logits = Tensor([[1.0, 2.0, 3.0], [0.1, 0.2, 0.9]])
    >>> result = log_softmax(logits, dim=-1)
    >>> print(result.shape)
    (2, 3)
    """
    max_vals = np.max(x.data, axis=dim, keepdims=True)
    shifted = x.data - max_vals
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=dim, keepdims=True))
    result = x.data - max_vals - log_sum_exp
    return Tensor(result)


class MSELoss:
    """Mean Squared Error loss for regression tasks."""

    def __init__(self):
        "Initialize MSE loss function."
        pass

    def forward(self, predictions, targets):
        """Compute mean squared error between pred and targets.

        >>> loss_fn = MSELoss()
        >>> predictions = Tensor([1.0, 2.0, 3.0])
        >>> targets = Tensor([1.5, 2.5, 2.8])
        >>> loss = loss_fn(predictions, targets)
        >>> print(f"MSE Loss: {loss.data:.4f}")
        MSE Loss: 0.1467
        """
        diff = predictions.data - targets.data
        squared_diff = diff**2
        mse = np.mean(squared_diff)
        return Tensor(mse)

    def __call__(self, predictions, targets):
        """Allows the loss function to be called like a function."""
        return self.forward(predictions, targets)

    def backward(self):
        """Compute gradients"""
        pass


class CrossEntropyLoss:
    """Cross-entropy loss for multi-class classification."""

    def __init__(self):
        """Initialize cross-entropy loss function."""
        pass

    def forward(self, logits, targets):
        """Compute cross-entropy loss between logits and target class indices.

        EXAMPLE:
        >>> loss_fn = CrossEntropyLoss()
        # 2 samples, 3 classes
        >>> logits = Tensor([[2.0, 1.0, 0.1], [0.5, 1.5, 0.8]])
        # First sample is class 0, second is class 1
        >>> targets = Tensor([0, 1])
        >>> loss = loss_fn(logits, targets)
        >>> print(f"Cross-Entropy Loss: {loss.data:.4f}")
        """
        log_probs = log_softmax(logits, dim=-1)
        batch_size = logits.shape[0]
        target_indices = targets.data.astype(int)
        selected_log_probs = log_probs.data[np.arange(batch_size), target_indices]
        cross_entropy = -np.mean(selected_log_probs)

        return Tensor(cross_entropy)

    def __call__(self, logits, targets):
        """Allows the loss function to be called like a function."""
        return self.forward(logits, targets)

    def backward(self):
        """Compute gradients"""
        pass


class BinaryCrossEntropyLoss:
    """Binary cross-entropy loss for binary classification."""

    def __init__(self):
        """Initialize binary cross-entropy loss function."""
        pass

    def forward(self, predictions, targets):
        """Compute binary cross-entropy loss.

        EXAMPLE:
        >>> loss_fn = BinaryCrossEntropyLoss()
        >>> predictions = Tensor([0.9, 0.1, 0.7, 0.3])  # Probabilities between 0 and 1
        >>> targets = Tensor([1.0, 0.0, 1.0, 0.0])      # Binary labels
        >>> loss = loss_fn(predictions, targets)
        >>> print(f"Binary Cross-Entropy Loss: {loss.data:.4f}")
        """
        eps = EPSILON
        clamped_preds = np.clip(predictions.data, eps, 1 - eps)

        log_preds = np.log(clamped_preds)
        log_one_minus_preds = np.log(1 - clamped_preds)

        bce_per_sample = -(
            targets.data * log_preds + (1 - targets.data) * log_one_minus_preds
        )

        bce_loss = np.mean(bce_per_sample)
        return Tensor(bce_loss)

    def __call__(self, predictions, targets):
        """Allows the loss function to be called like a function."""
        return self.forward(predictions, targets)

    def backward(self):
        """Compute gradients"""
        pass
