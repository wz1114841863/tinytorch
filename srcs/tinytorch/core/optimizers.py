import numpy as np
from typing import List, Union, Optional, Dict, Any
from tinytorch.core.tensor import Tensor
from tinytorch.core.autograd import enable_autograd

enable_autograd()

# Constants for optimizer defaults
DEFAULT_LEARNING_RATE_SGD = 0.01  # Default learning rate for SGD
DEFAULT_LEARNING_RATE_ADAM = 0.001  # Default learning rate for Adam/AdamW
DEFAULT_MOMENTUM = 0.9  # Default momentum for SGD
DEFAULT_BETA1 = 0.9  # First moment decay rate for Adam
DEFAULT_BETA2 = 0.999  # Second moment decay rate for Adam
DEFAULT_EPS = 1e-8  # Small epsilon for numerical stability in Adam
DEFAULT_WEIGHT_DECAY_ADAMW = 0.01  # Default weight decay for AdamW


class Optimizer:
    """Base class for all optimizers.

    This class defines the common interface that all optimizers must implement:
        - zero_grad(): Clear gradients from parameters
        - step(): Update parameters based on gradients
    """

    def __init__(self, params: List[Tensor]):
        """Initialize optimizer with parameters."""
        # validate that params is a list of Tensors
        if not isinstance(params, list):
            params = list(params)

        # store parameters and initialize state
        self.params = params

        # Ensure all parameters are Tensors and require gradients
        for param in self.params:
            if isinstance(param, Tensor):
                param.requires_grad = True
                param.grad = None

        self.step_count = 0  # Initialize step count for optimizers that require it

    def zero_grad(self):
        """Clear gradients from all parameters."""
        for param in self.params:
            if param.grad is not None:
                param.grad = None

    def step(self):
        """
        Update parameters based on gradients.

        This is abstract - each optimizer implements its own update rule.
        """
        raise NotImplementedError(
            f"Abstract method step() not implemented\n"
            f"  ❌ {self.__class__.__name__} inherits from Optimizer but doesn't define step()\n"
            f"  💡 Each optimizer must implement its own update rule (SGD, Adam, etc.)\n"
            f"  🔧 Override step() in your optimizer subclass:\n"
            f"      def step(self):\n"
            f"          for param in self.params:\n"
            f"              if param.grad is not None:\n"
            f"                  param.data -= self.lr * param.grad.data"
        )


class _ExtractGradientMixin:
    """Mixin added to optimizers to extract gradients from parameters."""

    def _extract_gradient(self, param: Tensor) -> np.ndarray:
        """Extract gradients from a parameter tensor.

        EXAMPLE:
        >>> param = Tensor([1.0, 2.0], requires_grad=True)
        >>> param.grad = Tensor([0.1, 0.2])
        >>> optimizer._extract_gradient(param)
        array([0.1, 0.2])
        """
        grad = param.grad
        if isinstance(grad, Tensor):
            return grad.data
        else:
            return grad


# Attach _extract_gradient to Optimizer so all subclasses inherit it
Optimizer._extract_gradient = _ExtractGradientMixin._extract_gradient


class SGD(Optimizer):
    """Stochastic Gradient Descent (SGD) optimizer.

    This optimizer updates parameters in the direction of the negative gradient.
    It can also include momentum to accelerate convergence.
    """

    def __init__(
        self,
        params: List[Tensor],
        lr: float = DEFAULT_LEARNING_RATE_SGD,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ):
        """Initialize SGD optimizer.

        Args:
            params: List of parameters to optimize
            lr: Learning rate (default: 0.01)
            momentum: Momentum factor (default: 0.0)
            weight_decay: Weight decay factor (default: 0.0)
        """
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        # Initialize momentum buffers (created lazily)
        self.momentum_buffers = [None for _ in self.params]

    def has_momentum(self) -> bool:
        """Check if momentum is enabled."""
        return self.momentum > 0.0

    def get_momentum_state(self):
        """Get the current momentum state for all parameters."""
        if not self.has_momentum():
            return None
        return [
            buf.copy() if buf is not None else None for buf in self.momentum_buffers
        ]

    def set_momentum_state(self, state):
        """Restore momentum buffers from checkpoint."""
        if state is None or not self.has_momentum():
            return

        if len(state) != len(self.momentum_buffers):
            raise ValueError(
                f"Momentum state length {len(state)} does not match number of parameters {len(self.momentum_buffers)}"
            )

        for i, buf in enumerate(state):
            if buf is not None:
                self.momentum_buffers[i] = buf.copy()

    def step(self):
        """Perform SGD update step with momentum."""
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            grad_data = self._extract_gradient(param)

            # Apply weight decay if specified
            if self.weight_decay != 0.0:
                grad_data = grad_data + self.weight_decay * param.data

            # update momentum buffer
            if self.momentum != 0:
                if self.momentum_buffers[i] is None:
                    self.momentum_buffers[i] = np.zeros_like(param.data)
                self.momentum_buffers[i] = (
                    self.momentum * self.momentum_buffers[i] + grad_data
                )
                grad_data = self.momentum_buffers[i]

            # Update parameters
            param.data = param.data - self.lr * grad_data

        self.step_count += 1  # Increment step count after update


class Adam(Optimizer):
    """Adam optimizer with adaptive learning rates."""

    def __init__(
        self,
        params: List[Tensor],
        lr: float = DEFAULT_LEARNING_RATE_ADAM,
        betas: tuple = (DEFAULT_BETA1, DEFAULT_BETA2),
        eps: float = DEFAULT_EPS,
        weight_decay: float = 0.0,
    ):
        """Initialize Adam optimizer.

        Args:
            params: List of parameters to optimize
            lr: Learning rate (default: 0.001)
            betas: Coefficients for computing running averages of gradient and its square (default: (0.9, 0.999))
            eps: Term added to denominator for numerical stability (default: 1e-8)
            weight_decay: Weight decay factor (default: 0.0)
        """
        super().__init__(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # Initialize moment buffers (created lazily)
        self.m_buffers = [None for _ in self.params]  # First moment (mean)
        self.v_buffers = [None for _ in self.params]  # Second moment (variance)


class _AdamUpdateMomentsMixin:
    """Mixin added to Adam for moment updates."""

    def _update_moments(self, i: int, grad_data: np.ndarray) -> tuple:
        """Update first and second moment estimates with bias correction.

        EXAMPLE:
        >>> m_hat, v_hat = self._update_moments(0, np.array([0.1, 0.2]))
        >>> # m_hat ≈ grad (after bias correction at step 1)
        >>> # v_hat ≈ grad^2 (after bias correction at step 1)

        """
        # Initialize buffers if needed
        if self.m_buffers[i] is None:
            self.m_buffers[i] = np.zeros_like(grad_data)
            self.v_buffers[i] = np.zeros_like(grad_data)

        # Update biased first moment estimate
        self.m_buffers[i] = (
            self.beta1 * self.m_buffers[i] + (1 - self.beta1) * grad_data
        )

        # Update biased second moment estimate
        self.v_buffers[i] = self.beta2 * self.v_buffers[i] + (1 - self.beta2) * (
            grad_data**2
        )

        # Compute bias correction
        bias_correction1 = 1 - self.beta1**self.step_count
        bias_correction2 = 1 - self.beta2**self.step_count

        # Compute bias-corrected moments
        m_hat = self.m_buffers[i] / bias_correction1
        v_hat = self.v_buffers[i] / bias_correction2

        return m_hat, v_hat


# Attach _update_moments to Adam
Adam._update_moments = _AdamUpdateMomentsMixin._update_moments


class _AdamStepMixin:
    """Mixin added to Adam for step method."""

    def step(self):
        """Perform Adam update step."""
        self.step_count += 1  # Increment step count first for correct bias correction

        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            grad_data = self._extract_gradient(param)

            # Apply weight decay if specified (AdamW style)
            if self.weight_decay != 0.0:
                grad_data = grad_data + self.weight_decay * param.data

            # Update moments and get bias-corrected estimates
            m_hat, v_hat = self._update_moments(i, grad_data)

            # Update parameters
            param.data = param.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# Attach step to Adam
Adam.step = _AdamStepMixin.step


class AdamW(Optimizer):
    """AdamW optimizer with decoupled weight decay."""

    def __init__(
        self,
        params: List[Tensor],
        lr: float = DEFAULT_LEARNING_RATE_ADAM,
        betas: tuple = (DEFAULT_BETA1, DEFAULT_BETA2),
        eps: float = DEFAULT_EPS,
        weight_decay: float = DEFAULT_WEIGHT_DECAY_ADAMW,
    ):
        """Initialize AdamW optimizer.

        Args:
            params: List of parameters to optimize
            lr: Learning rate (default: 0.001)
            betas: Coefficients for computing running averages of gradient and its square (default: (0.9, 0.999))
            eps: Term added to denominator for numerical stability (default: 1e-8)
            weight_decay: Weight decay factor (default: 0.01)
        """
        super().__init__(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # Initialize moment buffers (created lazily)
        self.m_buffers = [None for _ in self.params]  # First moment (mean)
        self.v_buffers = [None for _ in self.params]  # Second moment (variance)


class _AdamWUpdateMomentsMixin:
    """Mixin added to AdamW for moment updates."""

    def _update_moments(self, i: int, grad_data: np.ndarray) -> tuple:
        """Update first and second moment estimates with bias correction for AdamW.

        EXAMPLE:
        >>> m_hat, v_hat = self._update_moments(0, np.array([0.1, 0.2]))
        """
        # Initialize buffers if needed
        if self.m_buffers[i] is None:
            self.m_buffers[i] = np.zeros_like(grad_data)
            self.v_buffers[i] = np.zeros_like(grad_data)

        # Update biased first moment estimate
        self.m_buffers[i] = (
            self.beta1 * self.m_buffers[i] + (1 - self.beta1) * grad_data
        )

        # Update biased second moment estimate
        self.v_buffers[i] = self.beta2 * self.v_buffers[i] + (1 - self.beta2) * (
            grad_data**2
        )

        # Compute bias correction
        bias_correction1 = 1 - self.beta1**self.step_count
        bias_correction2 = 1 - self.beta2**self.step_count

        # Compute bias-corrected moments
        m_hat = self.m_buffers[i] / bias_correction1
        v_hat = self.v_buffers[i] / bias_correction2

        return m_hat, v_hat


# Attach _update_moments to AdamW
AdamW._update_moments = _AdamWUpdateMomentsMixin._update_moments


class _AdamWStepMixin:
    """Mixin added to AdamW for step method."""

    def step(self):
        """Perform AdamW update step with decoupled weight decay"""
        # Increment step counter first
        self.step_count += 1

        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            # Extract gradient using shared helper
            grad_data = self._extract_gradient(param)

            # Update moments using PURE gradients (no weight decay mixed in)
            m_hat, v_hat = self._update_moments(i, grad_data)

            # Apply gradient-based update
            param.data = param.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

            # Apply decoupled weight decay (separate from gradient update)
            if self.weight_decay != 0:
                param.data = param.data * (1 - self.lr * self.weight_decay)


# Attach step to AdamW
AdamW.step = _AdamWStepMixin.step
