import numpy as np
import pickle
import time
import sys
import os
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path

from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear
from tinytorch.core.losses import MSELoss, CrossEntropyLoss
from tinytorch.core.optimizers import SGD, AdamW
from tinytorch.core.autograd import enable_autograd

enable_autograd()

# Constants for learning rate scheduling defaults
DEFAULT_MAX_LR = 0.1  # Default maximum learning rate for cosine schedule
DEFAULT_MIN_LR = 0.01  # Default minimum learning rate for cosine schedule
DEFAULT_TOTAL_EPOCHS = 100  # Default total epochs for learning rate schedule


class CosineSchedule:
    """Implements a cosine learning rate schedule.

    EXAMPLE:
    >>> schedule = CosineSchedule(max_lr=0.1, min_lr=0.01, total_epochs=100)
    >>> print(schedule.get_lr(0))    # Start: 0.1
    >>> print(schedule.get_lr(50))   # Middle: ~0.055
    >>> print(schedule.get_lr(100))  # End: 0.01
    """

    def __init__(
        self,
        max_lr: float = DEFAULT_MAX_LR,
        min_lr: float = DEFAULT_MIN_LR,
        total_epochs: int = DEFAULT_TOTAL_EPOCHS,
    ):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.total_epochs = total_epochs

    def get_lr(self, epoch: int) -> float:
        """Get learning rate for current epoch."""
        if epoch >= self.total_epochs:
            return self.min_lr

        # Cosine annealing formula
        cosine_factor = (1 + np.cos(np.pi * epoch / self.total_epochs)) / 2
        return self.min_lr + (self.max_lr - self.min_lr) * cosine_factor


def clip_grad_norm(parameters, max_morm: float = 1.0):
    """Clips gradient norm of an iterable of parameters.

     EXAMPLE:
    >>> params = [Tensor([1, 2, 3], requires_grad=True)]
    >>> params[0].grad = Tensor([10, 20, 30])  # Large gradients
    >>> original_norm = clip_grad_norm(params, max_norm=1.0)
    """
    if not parameters:
        return 0.0

    total_norm = 0.0
    for param in parameters:
        if param.grad is not None:
            if isinstance(param.grad, np.ndarray):
                grad_data = param.grad
            else:
                grad_data = param.grad.data
            total_norm += np.sum(grad_data**2)
    total_norm = np.sqrt(total_norm)

    # clip if necessary
    if total_norm > max_morm:
        clip_coef = max_morm / (total_norm + 1e-6)
        for param in parameters:
            if param.grad is not None:
                if isinstance(param.grad, np.ndarray):
                    param.grad *= clip_coef
                else:
                    param.grad.data *= clip_coef

    return float(total_norm)


class Trainer:
    """Component responsible for training a model."""

    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        scheduler=None,
        grad_clip_norm=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.grad_clip_norm = grad_clip_norm

        # Training state
        self.epoch = 0
        self.step = 0
        self.training_mode = True

        # History tracking
        self.history = {"train_loss": [], "eval_loss": [], "learning_rates": []}

    def _get_model_state(self):
        """Extract model parameters for checkpointing."""
        return {i: param.data.copy() for i, param in enumerate(self.model.parameters())}

    def _set_model_state(self, state):
        """Restore model parameters from checkpoint."""
        for i, param in enumerate(self.model.parameters()):
            if i in state:
                param.data = state[i].copy()

    def _get_optimizer_state(self):
        """Extract optimizer state for checkpointing."""
        state = {}
        state["lr"] = self.optimizer.lr
        if hasattr(self.optimizer, "has_momentum") and self.optimizer.has_momentum():
            momentum_state = self.optimizer.get_momentum_state()
            if momentum_state is not None:
                state["momentum_buffers"] = momentum_state
        return state

    def _set_optimizer_state(self, state):
        """Restore optimizer state from checkpoint."""
        if "lr" in state:
            self.optimizer.lr = state["lr"]
        if "momentum_buffers" in state:
            if (
                hasattr(self.optimizer, "has_momentum")
                and self.optimizer.has_momentum()
            ):
                self.optimizer.set_momentum_state(state["momentum_buffers"])

    def _get_scheduler_state(self):
        """Extract scheduler state for checkpointing."""
        if self.scheduler is None:
            return None
        return {
            "max_lr": getattr(self.scheduler, "max_lr", None),
            "min_lr": getattr(self.scheduler, "min_lr", None),
            "total_epochs": getattr(self.scheduler, "total_epochs", None),
        }

    def _set_scheduler_state(self, state):
        """Restore scheduler state from checkpoint."""
        if state is None or self.scheduler is None:
            return
        for key, value in state.items():
            if hasattr(self.scheduler, key):
                setattr(self.scheduler, key, value)

    def _process_batch(self, inputs, targets, accumulation_steps):
        """Process a single batch of data."""
        outputs = self.model.forward(inputs)
        loss = self.loss_fn.forward(outputs, targets)
        # gradient scaling for accumulation
        scaled_loss = loss.data / accumulation_steps
        scaled_gradient = np.ones_like(loss.data) / accumulation_steps
        loss.backward(scaled_gradient)
        return float(scaled_loss)

    def _optimizer_update(self):
        """Apply optimizer step and scheduler update."""
        if self.grad_clip_norm is not None:
            params = self.model.parameters()
            clip_grad_norm(params, self.grad_clip_norm)

        self.optimizer.step()
        self.optimizer.zero_grad()

    def train_epoch(self, dataloader, accumulation_steps=1):
        """Train for one epoch through the dataset."""
        self.model.training = True
        self.training_mode = True

        total_loss = 0.0
        num_batches = 0
        accumulated_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            accumulated_loss += self._process_batch(inputs, targets, accumulation_steps)

            if (batch_idx + 1) % accumulation_steps == 0:
                self._optimizer_update()
                total_loss += accumulated_loss
                accumulated_loss = 0.0
                num_batches += 1
                self.step += 1

        if accumulated_loss > 0.0:
            self._optimizer_update()
            total_loss += accumulated_loss
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        self.history["train_loss"].append(avg_loss)

        if self.scheduler is not None:
            current_lr = self.scheduler.get_lr(self.epoch)
            self.optimizer.lr = current_lr
            self.history["learning_rates"].append(current_lr)

        self.epoch += 1
        return avg_loss

    def evaluate(self, dataloader):
        """Evaluate model on validation/test dataset."""
        self.model.training = False
        self.training_mode = False

        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0

        for inputs, targets in dataloader:
            # Forward pass only
            outputs = self.model.forward(inputs)
            loss = self.loss_fn.forward(outputs, targets)

            total_loss += loss.data
            num_batches += 1

            # Calculate accuracy (for classification)
            if len(outputs.data.shape) > 1:  # Multi-class
                predictions = np.argmax(outputs.data, axis=1)
                if len(targets.data.shape) == 1:  # Integer targets
                    correct += np.sum(predictions == targets.data)
                else:  # One-hot targets
                    correct += np.sum(predictions == np.argmax(targets.data, axis=1))
                total += len(predictions)

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        self.history["eval_loss"].append(avg_loss)

        return avg_loss, accuracy

    def save_checkpoint(self, path):
        """Save model and optimizer state to a checkpoint file."""
        checkpoint = {
            "epoch": self.epoch,
            "step": self.step,
            "model_state": self._get_model_state(),
            "optimizer_state": self._get_optimizer_state(),
            "scheduler_state": self._get_scheduler_state(),
            "history": self.history,
            "training_mode": self.training_mode,
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(checkpoint, f)

    def load_checkpoint(self, path):
        with open(path, "rb") as f:
            checkpoint = pickle.load(f)

        self.epoch = checkpoint["epoch"]
        self.step = checkpoint["step"]
        self.history = checkpoint["history"]
        self.training_mode = checkpoint["training_mode"]

        # Restore states
        if "model_state" in checkpoint:
            self._set_model_state(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            self._set_optimizer_state(checkpoint["optimizer_state"])
        if "scheduler_state" in checkpoint:
            self._set_scheduler_state(checkpoint["scheduler_state"])
