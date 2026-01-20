import numpy as np
from .tensor import Tensor

TOLERANCE = 1e-10

__all__ = ["Sigmoid", "ReLU", "Tanh", "GELU", "Softmax"]


class Sigmoid:
    """Sigmoid

    sigmoid(x) = 1 / (1 + exp(-x))
    Maps any real number to (0, 1) range.
    Perfect for probabilities and binary classification.
    """

    def parameters(self):
        """Return empty list (activations have no learnable parameters)."""
        return []

    def forward(self, x):
        """Apply sigmoid activation element-wise.

        EXAMPLE:
        >>> sigmoid = Sigmoid()
        >>> x = Tensor([-2, 0, 2])
        >>> result = sigmoid(x)
        >>> print(result.data)
        [0.119, 0.5, 0.881]  # All values between 0 and 1.
        """
        # Clip extreme values to pervent overflow
        # (sigmoid(-500) ≈ 0, sigmoid(500) ≈ 1)
        # Clipping at ±500 ensures exp() stays within float64 range
        z = np.clip(x.data, -500, 500)

        # Use numerically stable sigmoid
        # For positive values: 1 / (1 + exp(-x))
        # For negative values: exp(x) / (1 + exp(x)) = 1 / (1 + exp(-x)) after clipping
        result_data = np.zeros_like(z)

        pos_mask = z >= 0
        result_data[pos_mask] = 1.0 / (1.0 + np.exp(-z[pos_mask]))

        neg_mask = z <= 0
        exp_z = np.exp(z[neg_mask])
        result_data[neg_mask] = exp_z / (1.0 + exp_z)

        return Tensor(result_data)

    def __call__(self, x):
        """Allows the activation to be called like a function."""
        return self.forward(x)

    def backward(self, grad):
        """Compute gradient."""
        pass


class ReLU:
    """ReLU

    relu = max(0, x)

    Sets negative values to zero, keeps positive values unchanged.
    Most popular activation for hidden layers.
    """

    def parameters(self):
        """Return empth list (activations have no learnable parameters)."""
        return []

    def forward(self, x):
        """Apply ReLU activation element-wise.

        EXAMPLE:
        >>> relu = ReLU()
        >>> x = Tensor([-2, -1, 0, 1, 2])
        >>> result = relu(x)
        >>> print(result.data)
        [0, 0, 0, 1, 2]  # Negative values become 0, positive unchanged
        """
        result = np.maximum(0, x.data)
        return Tensor(result)

    def __call__(self, x):
        """Allows the activation to be called like a function."""
        return self.forward(x)

    def backward(self, grad):
        """Compute gradient."""
        pass


class Tanh:
    """Tanh

    activation: f(x) = (e^x - e^(-x))/(e^x + e^(-x))
    Maps any real number to (-1, 1) range.
    Zero-centered alternative to sigmoid.
    """

    def parameters(self):
        """Return empth list (activations have no learnable parameters)."""
        return []

    def forward(self, x):
        """Apply tanh activation element-wise.

        EXAMPLE:
        >>> tanh = Tanh()
        >>> x = Tensor([-2, 0, 2])
        >>> result = tanh(x)
        >>> print(result.data)
        [-0.964, 0.0, 0.964]  # Range (-1, 1), symmetric around 0
        """
        result = np.tanh(x.data)
        return Tensor(result)

    def __call__(self, x):
        """Allows the activation to be called like a function."""
        return self.forward(x)

    def backward(self, grad):
        """Compute gradient."""
        pass


class GELU:
    """GELU activation

    f(x) = x * Φ(x) ≈ x * Sigmoid(1.702 * x)

    Smooth approximation to ReLU, used in modern transformers.
    Where Φ(x) is the cumulative distribution function of standard normal.
    """

    def parameters(self):
        """Return empth list (activations have no learnable parameters)."""
        return []

    def forward(self, x):
        """Apply gelu activation element-wise.

        EXAMPLE:
        >>> gelu = GELU()
        >>> x = Tensor([-1, 0, 1])
        >>> result = gelu(x)
        >>> print(result.data)
        [-0.159, 0.0, 0.841]  # Smooth, like ReLU but differentiable everywhere
        """
        sigmoid_part = 1.0 / (1.0 + np.exp(-1.702 * x.data))
        result = x.data * sigmoid_part
        return Tensor(result)

    def __call__(self, x):
        """Allows the activation to be called like a function."""
        return self.forward(x)

    def backward(self, grad):
        """Compute gradient."""
        pass


class Softmax:
    """Softmax.

    f(x_i) = e^(x_i) / Σ(e^(x_j))
    Converts any vector to a probability distribution.
    Sum of all outputs equals 1.0.
    """

    def parameters(self):
        """Return empth list (activations have no learnable parameters)."""
        return []

    def forward(self, x, dim: int = -1):
        """Apply softmax activation element-wise.

        EXAMPLE:
        >>> softmax = Softmax()
        >>> x = Tensor([1, 2, 3])
        >>> result = softmax(x)
        >>> print(result.data)
        [0.090, 0.245, 0.665]  # Sums to 1.0, larger inputs get higher probability
        """
        x_max_data = np.max(x.data, axis=dim, keepdims=True)
        x_max = Tensor(x_max_data)
        x_shifted = x - x_max

        exp_values = Tensor(np.exp(x_shifted.data))
        exp_sum_data = np.sum(exp_values.data, axis=dim, keepdims=True)
        exp_sum = Tensor(exp_sum_data)

        result = exp_values / exp_sum
        return result

    def __call__(self, x):
        """Allows the activation to be called like a function."""
        return self.forward(x)
