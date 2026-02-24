import numpy as np
import sys
import os

from tinytorch.core.tensor import Tensor


# Constants for numerical differentiation
EPSILON = 1e-7  # Small perturbation for numerical gradient computation


class Function:
    """Base class for differentiable operations.

    Every operation that needs gradients (add, multiply, matmul, etc.)
    will inherit from this class and implement the apply() method.

    Example:
        class AddBackward(Function):
            def apply(self, grad_output):
                # Addition distributes gradients equally
                return grad_output, grad_output
    """

    def __init__(self, *tensors):
        """Initialize function with input tensors."""
        self.saved_tensors = tensors

    def apply(self, grad_output):
        """Compute gradients for inputs.

        Args:
           grad_output: Gradient flowing backward from output
        """
        raise NotImplementedError("Subclass must implement apply()")


class AddBackward(Function):
    """Backward function for addition operation."""

    def apply(self, grad_output):
        """Gradient of addition is distributed equally to both inputs.

        EXAMPLE:
        >>> a = Tensor([1, 2, 3], requires_grad=True)
        >>> b = Tensor([4, 5, 6], requires_grad=True)
        >>> z = a + b  # z = [5, 7, 9]
        >>> # During backward: grad_output = [1, 1, 1]
        >>> # Result: grad_a = [1, 1, 1], grad_b = [1, 1, 1]
        """
        a, b = self.saved_tensors
        grad_a = grad_b = None

        if isinstance(a, Tensor) and a.requires_grad:
            grad_a = grad_output
        if isinstance(b, Tensor) and b.requires_grad:
            grad_b = grad_output
        return grad_a, grad_b


class MulBackward(Function):
    """Backward function for multiplication operation."""

    def apply(self, grad_output):
        """Gradient of multiplication using the product rule.

        EXAMPLE:
        >>> a = Tensor([1, 2, 3], requires_grad=True)
        >>> b = Tensor([4, 5, 6], requires_grad=True)
        >>> z = a * b  # z = [4, 10, 18]
        >>> # During backward: grad_output = [1, 1, 1]
        >>> # Result: grad_a = b * grad_output = [4, 5, 6]
        >>> #         grad_b = a * grad_output = [1, 2, 3]
        """
        a, b = self.saved_tensors
        grad_a = grad_b = None

        if isinstance(a, Tensor) and a.requires_grad:
            grad_a = b * grad_output
        if isinstance(b, Tensor) and b.requires_grad:
            grad_b = a * grad_output
        return grad_a, grad_b


class SubBackward(Function):
    """Backward function for subtraction operation."""

    def apply(self, grad_output):
        """Gradient of subtraction.

        EXAMPLE:
        >>> a = Tensor([5, 7, 9], requires_grad=True)
        >>> b = Tensor([1, 2, 3], requires_grad=True)
        >>> z = a - b  # z = [4, 5, 6]
        >>> # During backward: grad_output = [1, 1, 1]
        >>> # Result: grad_a = [1, 1, 1], grad_b = [-1, -1, -1]
        """
        a, b = self.saved_tensors
        grad_a = grad_b = None

        if isinstance(a, Tensor) and a.requires_grad:
            grad_a = grad_output
        if isinstance(b, Tensor) and b.requires_grad:
            grad_b = -grad_output
        return grad_a, grad_b


class DivBackward(Function):
    """Backward function for division operation."""

    def apply(self, grad_output):
        """Gradient of division using the quotient rule.

        EXAMPLE:
        >>> a = Tensor([4, 9, 16], requires_grad=True)
        >>> b = Tensor([2, 3, 4], requires_grad=True)
        >>> z = a / b  # z = [2, 3, 4]
        >>> # During backward: grad_output = [1, 1, 1]
        >>> # Result: grad_a = (1 / b) * grad_output = [0.5, 0.333..., 0.25]
        >>> #         grad_b = (-a / b^2) * grad_output = [-1.0, -1.0, -1.0]
        """
        a, b = self.saved_tensors
        grad_a = grad_b = None

        if isinstance(a, Tensor) and a.requires_grad:
            if isinstance(b, Tensor):
                grad_a = (1 / b.data) * grad_output
            else:
                grad_a = (1 / b) * grad_output
        if isinstance(b, Tensor) and b.requires_grad:
            grad_b = -grad_output * a.data / (b.data**2)
        return grad_a, grad_b


class MatmulBackward(Function):
    """Gradient computation for matrix multiplication."""

    def apply(self, grad_output):
        """Compute gradients for matrix multiplication.

        EXAMPLE:
        >>> A = Tensor([[1, 2], [3, 4]], requires_grad=True)
        >>> B = Tensor([[5, 6], [7, 8]], requires_grad=True)
        >>> C = A.matmul(B)  # C = [[19, 22], [43, 50]]
        >>> # During backward: grad_output = [[1, 1], [1, 1]]
        >>> # Result: grad_A = grad_output.matmul(B.T) = [[13, 15], [13, 15]]
        >>> #         grad_B = A.T.matmul(grad_output) = [[4, 4], [6, 6]]
        """
        a, b = self.saved_tensors
        grad_a = grad_b = None

        if isinstance(a, Tensor) and a.requires_grad:
            if b.data.ndim >= 2:
                b_T = np.swapaxes(b.data, -2, -1)
            else:
                b_T = b.data.T
            grad_a = np.matmul(grad_output, b_T)

        if isinstance(b, Tensor) and b.requires_grad:
            if a.data.ndim >= 2:
                a_T = np.swapaxes(a.data, -2, -1)
            else:
                a_T = a.data.T
            grad_b = np.matmul(a_T, grad_output)
        return grad_a, grad_b


class TransposeBackward(Function):
    """Gradient computation for transpose operation."""

    def __init__(self, tensors, dim0, dim1):
        """
        Args:
            tensor: Input tensor
            dim0: First dimension to swap (None for default)
            dim1: Second dimension to swap (None for default)
        """
        super().__init__(tensors)
        self.dim0 = dim0
        self.dim1 = dim1

    def apply(self, grad_output):
        """Transpose gradient back to original shape.

        EXAMPLE:
        >>> X = Tensor([[1, 2], [3, 4]], requires_grad=True)
        >>> Y = X.transpose()  # [[1, 3], [2, 4]]
        >>> # During backward: grad_output = [[a, b], [c, d]]
        >>> # grad_X = grad_output.T = [[a, c], [b, d]]
        """
        (a,) = self.saved_tensors
        grad_a = None

        if isinstance(a, Tensor) and a.requires_grad:
            if self.dim0 is None or self.dim1 is None:
                if grad_output.ndim < 2:
                    grad_a = grad_output.copy()
                else:
                    axes = list(range(grad_output.ndim))
                    axes[-1], axes[-2] = axes[-2], axes[-1]
                    grad_a = np.transpose(grad_output, axes)
            else:
                axes = list(range(grad_output.ndim))
                axes[self.dim0], axes[self.dim1] = axes[self.dim1], axes[self.dim0]
                grad_a = np.transpose(grad_output, axes)
        return grad_a


class PermuteBackward(Function):
    """Gradient computation for permute operation."""

    def __init__(self, *tensors, axes):
        """
        Args:
            tensor: Input tensor
            dims: Tuple specifying the permutation of dimensions
        """
        super().__init__(*tensors)
        self.axes = axes
        self.inverse_axes = tuple(np.argsort(axes))

    def apply(self, grad_output):
        """Permute gradient back to original shape.

        EXAMPLE:
        >>> X = Tensor of shape (2, 3, 4), requires_grad=True
        >>> Y = X.permute(1, 0, 2)  # Y shape: (3, 2, 4)
        >>> # During backward: grad_output shape: (3, 2, 4)
        >>> # grad_X shape: (2, 3, 4) obtained by inverse permutation
        """
        (x,) = self.saved_tensors
        grad_x = None

        if isinstance(x, Tensor) and x.requires_grad:
            grad_x = np.transpose(grad_output, self.inverse_axes)

        return (grad_x,)


class EmbeddingBackward(Function):
    """Gradient computation for embedding lookup operation."""

    def __init__(self, weight, indices):
        """
        Args:
            weight: Embedding weight matrix
            indices: Indices used for lookup
        """
        super().__init__(weight)
        self.indices = indices

    def apply(self, grad_output):
        """Compute gradient for embedding lookup.


        EXAMPLE:
        >>> vocab = Tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], requires_grad=True)  # 3 words, 2D
        >>> indices = Tensor([0, 2, 0])  # Select words 0, 2, 0
        >>> output = vocab[indices]  # [[0.1, 0.2], [0.5, 0.6], [0.1, 0.2]]
        >>> # During backward: grad_output = [[1, 1], [1, 1], [1, 1]]
        >>> # grad_vocab[0] accumulates twice: [1, 1] + [1, 1] = [2, 2]
        >>> # grad_vocab[2] once: [1, 1]
        """
        weight = self.saved_tensors
        grad_weight = None

        if isinstance(weight, Tensor) and weight.requires_grad:
            grad_weight = np.zeros_like(weight.data)

            # Scatter gradients back to embedding weights
            # np.add.at accumulates gradients for repeated indices
            # 展开索引, 不关心这些词属于哪句话, 只关心一共查了哪些词
            # 假如一个词在一个batch中出现多次,那么它的梯度应该累加
            indices_flat = self.indices.data.astype(int).flatten()
            # 同样展开: (Batch_Size * Seq_Len, Embedding_Dim)
            grad_output_reshaped = grad_output.reshape(-1, grad_output.shape[-1])
            np.add.at(grad_weight, indices_flat, grad_output_reshaped)

        # 在自动微分框架中, apply 方法通常需要返回与 forward 输入参数数量一致的梯度.
        # 隐含的意思是第 2 个参数 indices 的梯度是 None
        return (grad_weight,)


class SliceBackward(Function):
    """Gradient computation for slicing operation."""

    def __init__(self, tensor, key):
        """
        Args:
            tensor: Original tensor being sliced
            key: Slicing key (index, slice, tuple of slices, etc.)
        """
        super().__init__(tensor)
        self.key = key
        self.original_shape = tensor.shape

    def apply(self, grad_output):
        """Compute gradient for slicing operation.

        EXAMPLE:
        >>> X = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        >>> Y = X[:, 1:3]  # Y = [[2, 3], [5, 6]]
        >>> # During backward: grad_output = [[a, b], [c, d]]
        >>> # grad_X = [[0, a, b], [0, c, d]]
        """
        (x,) = self.saved_tensors
        grad_x = None

        if isinstance(x, Tensor) and x.requires_grad:
            grad_x = np.zeros(self.original_shape, dtype=np.float32)
            grad_x[self.key] = grad_output

        return (grad_x,)


class ReshapeBackward(Function):
    """Gradient computation for reshape operation."""

    def __init__(self, tensor, original_shape):
        """
        Args:
            tensor: Input tensor
            original_shape: Shape before reshape
        """
        super().__init__(tensor)
        self.original_shape = original_shape

    def apply(self, grad_output):
        """Compute gradient for reshape operation.

        EXAMPLE:
        >>> X = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)  # Shape (2, 3)
        >>> Y = X.reshape(3, 2)  # Shape (3, 2)
        >>> # During backward: grad_output shape (3, 2)
        >>> # grad_X shape (2, 3) obtained by reshaping grad_output
        """
        (x,) = self.saved_tensors
        grad_x = None

        if isinstance(x, Tensor) and x.requires_grad:
            grad_x = grad_output.reshape(self.original_shape)

        return (grad_x,)


class SumBackward(Function):
    """Gradient computation for tensor sum."""

    def apply(self, grad_output):
        """Compute gradient for sum operation.

        EXAMPLE:
        >>> X = Tensor([[1, 2], [3, 4]], requires_grad=True)
        >>> Y = X.sum()  # Y = 10
        >>> # During backward: grad_output = 1
        >>> # grad_X = [[1, 1], [1, 1]]
        """
        (x,) = self.saved_tensors
        grad_x = None

        if isinstance(x, Tensor) and x.requires_grad:
            grad_x = np.ones_like(x.data) * grad_output

        return (grad_x,)


class ReLUBackward(Function):
    """Gradient computation for ReLU activation."""

    def apply(self, grad_output):
        """Compute gradient for ReLU operation.

        EXAMPLE:
        >>> X = Tensor([-1, 0, 2, 3], requires_grad=True)
        >>> Y = X.relu()  # Y = [0, 0, 2, 3]
        >>> # During backward: grad_output = [a, b, c, d]
        >>> # grad_X = [0, 0, c, d]
        """
        (x,) = self.saved_tensors
        grad_x = None

        if isinstance(x, Tensor) and x.requires_grad:
            # ReLU gradient: 1 if x > 0, else 0
            relu_grad = (x.data > 0).astype(np.float32)
            grad_x = grad_output * relu_grad

        return (grad_x,)


class SigmoidBackward(Function):
    """Gradient computation for Sigmoid activation.

    Sigmoid: sigmoid(x) = 1/(1 + exp(-x))
    Derivative: sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
    """

    def __init__(self, input_tensor, output_tensor):
        """
        Initialize with both input and output.

        Args:
            input_tensor: Original input to sigmoid
            output_tensor: Output of sigmoid (saves recomputation)
        """
        super().__init__(input_tensor)
        self.output_data = output_tensor.data

    def apply(self, grad_output):
        """Compute gradient for Sigmoid operation.

        EXAMPLE:
        >>> X = Tensor([0, 2], requires_grad=True)
        >>> Y = X.sigmoid()  # Y = [0.5, 0.880797]
        >>> # During backward: grad_output = [a, b]
        >>> # grad_X = [a * 0.25, b * 0.104994]
        """
        (x,) = self.saved_tensors
        grad_x = None

        if isinstance(x, Tensor) and x.requires_grad:
            sigmoid_grad = self.output_data * (1 - self.output_data)
            grad_x = grad_output * sigmoid_grad

        return (grad_x,)


class SoftmaxBackward(Function):
    """Gradient computation for Softmax operation."""

    def __init__(self, input_tensor, output_tensor, dim=-1):
        """
        Initialize with input and output tensors.

        Args:
            input_tensor: Original input to softmax
            output_tensor: Output of softmax
            dim: Dimension along which softmax was computed
        """
        super().__init__(input_tensor)
        self.output_data = output_tensor.data
        self.dim = dim

    def apply(self, grad_output):
        """Compute gradient for Softmax operation.

        EXAMPLE:
        >>> X = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        >>> Y = softmax(X)  # [[0.09, 0.24, 0.67]] approximately
        >>> # During backward: grad_output = [[1, 0, 0]]
        >>> # sum_term = sum([1*0.09, 0*0.24, 0*0.67]) = 0.09
        >>> # grad_X[i] = softmax[i] * (grad_output[i] - sum_term)
        """
        (tensor,) = self.saved_tensors
        grad_x = None

        if isinstance(tensor, Tensor) and tensor.requires_grad:
            # Compute sum(grad_output * softmax) along the softmax dimension
            sum_term = np.sum(
                grad_output * self.output_data, axis=self.dim, keepdims=True
            )

            # Softmax gradient: softmax * (grad_output - sum_term)
            grad_x = self.output_data * (grad_output - sum_term)

        return (grad_x,)


class GELUBackward(Function):
    """Gradient computation for GELU activation."""

    def __init__(self, input_tensor):
        """Initialize with input tensor."""
        super().__init__(input_tensor)

    def apply(self, grad_output):
        """Compute gradient for GELU operation.

        EXAMPLE:
        >>> X = Tensor([0, 1], requires_grad=True)
        >>> Y = X.gelu()  # Y = [0, 0.8413] approximately
        >>> # During backward: grad_output = [a, b]
        >>> # grad_X computed using GELU derivative formula
        """
        (tensor,) = self.saved_tensors

        if isinstance(tensor, Tensor) and tensor.requires_grad:
            x = tensor.data
            # GELU derivative approximation
            # Using the tanh approximation: gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
            x_cubed = x**3
            tanh_arg = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
            tanh_out = np.tanh(tanh_arg)
            sech_squared = 1 - tanh_out**2

            # Derivative: 0.5 * (1 + tanh(...)) + 0.5 * x * sech²(...) * d(tanh_arg)/dx
            d_tanh_arg = sqrt_2_over_pi * (1 + 0.134145 * x**2)
            gelu_grad = 0.5 * (1 + tanh_out) + 0.5 * x * sech_squared * d_tanh_arg

            return (grad_output * gelu_grad,)
        return (None,)


class MSEBackward(Function):
    """Gradient computation for Mean Squared Error Loss."""

    def __init__(self, predictions, targets):
        """Initialize with predictions and targets."""
        super().__init__(predictions)
        self.targets_data = targets.data
        self.num_samples = np.size(targets.data)

    def apply(self, grad_output):
        """Compute gradient for MSE loss.

        EXAMPLE:
        >>> predictions = Tensor([2.0, 3.0], requires_grad=True)
        >>> targets = Tensor([1.0, 2.0])
        >>> loss = MSE(predictions, targets)  # (1² + 1²)/2 = 1.0
        >>> # During backward: grad_output = 1
        >>> # grad = 2 * ([2, 3] - [1, 2]) / 2 = [1, 1]
        """
        (predictions,) = self.saved_tensors
        grad_input = None

        if isinstance(predictions, Tensor) and predictions.requires_grad:
            # Gradient: 2 * (predictions - targets) / N
            grad = 2.0 * (predictions.data - self.targets_data) / self.num_samples
            grad_input = grad_output * grad

        return (grad_input,)


class BCEBackward(Function):
    """
    Gradient computation for Binary Cross-Entropy Loss.

    BCE: L = -[y*log(p) + (1-y)*log(1-p)]
    Derivative: ∂L/∂p = (p - y) / (p*(1-p)*N)
    """

    def __init__(self, predictions, targets):
        """Initialize with predictions and targets."""
        super().__init__(predictions)
        self.targets_data = targets.data
        self.num_samples = np.size(targets.data)

    def apply(self, grad_output):
        """Compute gradient for BCE loss.

        EXAMPLE:
        >>> predictions = Tensor([0.9, 0.1], requires_grad=True)
        >>> targets = Tensor([1.0, 0.0])
        >>> loss = BCE(predictions, targets)  # ≈ 0.10536
        >>> # During backward: grad_output = 1
        >>> # grad = ([0.9 - 1]/(0.9*0.1*2), [0.1 - 0]/(0.1*0.9*2)) = [-0.555..., 0.555...]
        """
        (predictions,) = self.saved_tensors
        grad_input = None

        if isinstance(predictions, Tensor) and predictions.requires_grad:
            eps = EPSILON
            p = np.clip(predictions.data, eps, 1 - eps)
            y = self.targets_data
            # Gradient: (p - y) / (p * (1-p) * N)
            grad = (p - y) / (p * (1 - p) * self.num_samples)
            grad_input = grad_output * grad

        return (grad_input,)


def _stable_softmax(logits_data):
    """Compute stable softmax for numerical stability.

    EXAMPLE:
    >>> logits = np.array([[2.0, 1.0, 0.1]])
    >>> probs = _stable_softmax(logits)
    >>> # probs ≈ [[0.659, 0.242, 0.099]]
    >>> # Each row sums to 1.0
    """
    max_logits = np.max(logits_data, axis=1, keepdims=True)
    exp_logits = np.exp(logits_data - max_logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def _one_hot_encode(targets, batch_size, num_classes):
    """Convert target class indices to one-hot encoding.

    EXAMPLE:
    >>> targets = np.array([0, 2, 1])
    >>> one_hot = _one_hot_encode(targets, batch_size=3, num_classes=3)
    >>> # one_hot = [[1, 0, 0], [0, 0, 1], [0, 1, 0]]
    """
    one_hot = np.zeros((batch_size, num_classes), dtype=np.float32)
    one_hot[np.arange(batch_size), targets] = 1.0
    return one_hot


class CrossEntropyBackward(Function):
    """Gradient computation for Cross-Entropy Loss."""

    def __init__(self, logits, targets):
        """Initialize with logits and target class indices."""
        super().__init__(logits)
        self.targets_data = targets.data.astype(int)
        self.batch_size = logits.data.shape[0]
        self.num_classes = logits.data.shape[1]

    def apply(self, grad_output):
        """Compute gradient for Cross-Entropy loss.

        EXAMPLE:
        >>> logits = Tensor([[2.0, 1.0, 0.1]], requires_grad=True)
        >>> targets = Tensor([0])  # Correct class is 0
        >>> loss = CrossEntropy(logits, targets)
        >>> # softmax ≈ [0.66, 0.24, 0.10]
        >>> # one_hot = [1, 0, 0]
        >>> # grad = ([0.66, 0.24, 0.10] - [1, 0, 0]) / 1 = [-0.34, 0.24, 0.10]
        """
        (logits,) = self.saved_tensors
        grad_input = None

        if isinstance(logits, Tensor) and logits.requires_grad:
            # Compute softmax probabilities
            # Using stable softmax: subtract max for numerical stability
            logits_data = logits.data
            max_logits = np.max(logits_data, axis=1, keepdims=True)
            exp_logits = np.exp(logits_data - max_logits)
            softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

            # Create one-hot encoding of targets
            one_hot = np.zeros((self.batch_size, self.num_classes), dtype=np.float32)
            one_hot[np.arange(self.batch_size), self.targets_data] = 1.0

            # Gradient: (softmax - one_hot) / batch_size
            grad = (softmax - one_hot) / self.batch_size
            grad_input = grad_output * grad

        return (grad_input,)


def enable_autograd(quiet=False):
    """Enable automatic differentiation globally.

    Example:
        enable_autograd()  # Call once
        x = Tensor([2.0], requires_grad=True)
        y = x * 3
        y.backward()
        print(x.grad)  # [3.0
    """
    # 检测是否已经启用自动微分
    if hasattr(Tensor, "_autograd_enabled"):
        # Silently return if already enabled - no need to warn
        return

    # 添加自动微分相关的方法到 Tensor 类
    _original_init = Tensor.__init__

    def gradient_aware_init(self, data, requires_grad=False):
        """Extended Tensor init that supports gradient tracking."""
        _original_init(self, data)
        self.requires_grad = requires_grad
        self.grad = None

    Tensor.__init__ = gradient_aware_init

    # 备份原始操作
    _original_add = Tensor.__add__
    _original_sub = Tensor.__sub__
    _original_mul = Tensor.__mul__
    _original_div = Tensor.__truediv__
    _original_getitem = Tensor.__getitem__

    _original_matmul = Tensor.matmul
    _original_transpose = Tensor.transpose
    _original_reshape = Tensor.reshape

    # Helper to safely check requires_grad
    def _get_requires_grad(tensor):
        """Safely get requires_grad, defaulting to False for pre-autograd tensors."""
        return (
            getattr(tensor, "requires_grad", False)
            if isinstance(tensor, Tensor)
            else False
        )

    def _ensure_grad_attrs(tensor):
        """Ensure tensor has gradient attributes (for tensors created before enable_autograd)."""
        if isinstance(tensor, Tensor):
            if not hasattr(tensor, "requires_grad"):
                tensor.requires_grad = False
            if not hasattr(tensor, "grad"):
                tensor.grad = None

    # Enhanced operations that track gradients
    def tracked_add(self, other):
        """Addition with gradient tracking."""
        _ensure_grad_attrs(self)

        if not isinstance(other, Tensor):
            other = Tensor(other)
        _ensure_grad_attrs(other)

        result = _original_add(self, other)
        _ensure_grad_attrs(result)

        # Track gradient if needed
        if _get_requires_grad(self) or _get_requires_grad(other):
            result.requires_grad = True
            result._grad_fn = AddBackward(self, other)

        return result

    def tracked_mul(self, other):
        """Multiplication with gradient tracking."""
        _ensure_grad_attrs(self)

        if not isinstance(other, Tensor):
            other_tensor = Tensor(other)
        else:
            other_tensor = other
        _ensure_grad_attrs(other_tensor)

        result = _original_mul(self, other)
        _ensure_grad_attrs(result)

        # Track gradient if needed
        if _get_requires_grad(self) or _get_requires_grad(other):
            result.requires_grad = True
            result._grad_fn = MulBackward(self, other)

        return result

    def tracked_matmul(self, other):
        """Matrix multiplication with gradient tracking."""
        _ensure_grad_attrs(self)
        _ensure_grad_attrs(other)

        # Call original matmul from Module 01
        result = _original_matmul(self, other)
        _ensure_grad_attrs(result)

        # Track gradient if needed
        if _get_requires_grad(self) or _get_requires_grad(other):
            result.requires_grad = True
            result._grad_fn = MatmulBackward(self, other)

        return result

    def tracked_transpose(self, dim0=None, dim1=None):
        """Transpose with gradient tracking."""
        _ensure_grad_attrs(self)

        # Call original transpose from Module 01
        result = _original_transpose(self, dim0, dim1)
        _ensure_grad_attrs(result)

        # Track gradient if needed
        if _get_requires_grad(self):
            result.requires_grad = True
            result._grad_fn = TransposeBackward(self, dim0, dim1)

        return result

    def tracked_reshape(self, *shape):
        """Reshape with gradient tracking."""
        _ensure_grad_attrs(self)
        original_shape = self.shape

        # Call original reshape from Module 01
        result = _original_reshape(self, *shape)
        _ensure_grad_attrs(result)

        # Track gradient if needed
        if _get_requires_grad(self):
            result.requires_grad = True
            result._grad_fn = ReshapeBackward(self, original_shape)

        return result

    def tracked_sub(self, other):
        """Subtraction with gradient tracking."""
        _ensure_grad_attrs(self)

        # Convert scalar to Tensor if needed
        if not isinstance(other, Tensor):
            other = Tensor(other)
        _ensure_grad_attrs(other)

        # Call original operation
        result = _original_sub(self, other)
        _ensure_grad_attrs(result)

        # Track gradient if needed
        if _get_requires_grad(self) or _get_requires_grad(other):
            result.requires_grad = True
            result._grad_fn = SubBackward(self, other)

        return result

    def tracked_div(self, other):
        """Division with gradient tracking."""
        _ensure_grad_attrs(self)

        # Convert scalar to Tensor if needed
        if not isinstance(other, Tensor):
            other = Tensor(other)
        _ensure_grad_attrs(other)

        # Call original operation
        result = _original_div(self, other)
        _ensure_grad_attrs(result)

        # Track gradient if needed
        if _get_requires_grad(self) or _get_requires_grad(other):
            result.requires_grad = True
            result._grad_fn = DivBackward(self, other)

        return result

    def tracked_getitem(self, key):
        """Indexing/slicing with gradient tracking."""
        _ensure_grad_attrs(self)

        # Call original __getitem__ from Module 01
        result = _original_getitem(self, key)
        _ensure_grad_attrs(result)

        # Track gradient if needed
        if _get_requires_grad(self):
            result.requires_grad = True
            result._grad_fn = SliceBackward(self, key)

        return result

    def sum_op(self, axis=None, keepdims=False):
        """Sum operation with gradient tracking."""
        _ensure_grad_attrs(self)

        result_data = np.sum(self.data, axis=axis, keepdims=keepdims)
        result = Tensor(result_data)

        if _get_requires_grad(self):
            result.requires_grad = True
            result._grad_fn = SumBackward(self)

        return result

    def backward(self, gradient=None):
        """Compute gradients via backpropagation.

        Example:
        x = Tensor([2.0], requires_grad=True)
        y = x * 3
        y.backward()  # Computes gradients for x
        print(x.grad)  # [3.0]
        """
        # Ensure gradient attribute exist
        _ensure_grad_attrs(self)

        # Only compute gradients if requires_grad is True
        if not _get_requires_grad(self):
            return

        # Initialize gradient if not already set
        if gradient is None:
            if self.data.size == 1:
                gradient = np.ones_like(self.data)
            else:
                raise ValueError(
                    f"backward() called on non-scalar tensor without gradient argument.\n"
                    f"  Tensor shape: {self.shape}\n"
                    f"  Issue: For non-scalar outputs, you must provide the gradient from the next layer.\n"
                    f"  Fix: Call backward(gradient) with the gradient tensor from the loss function."
                )

        # Initialize or accumulate gradient
        if self.grad is None:
            self.grad = np.zeros_like(self.data)

        # Handle broadcasting: sum gradients to match self.data shape
        # This happens when operations broadcasted tensors.
        if gradient.shape != self.grad.shape:
            # Step1: Remove extra leading dimensions added during forward pass
            # Example: gradient (batch_size, features) → self.grad (features,)
            while gradient.ndim > self.grad.ndim:
                gradient = gradient.sum(axis=0)

            # Step2: Sum along broadcasted dimensions
            # Example: bias with shape (1,) broadcast to (batch_size,) during forward
            for i in range(gradient.ndim):
                if self.grad.shape[i] == 1 and gradient.shape[i] != 1:
                    gradient = gradient.sum(axis=i, keepdims=True)

        # upgrad current tensor's gradient
        self.grad += gradient

        # Propagate gradients through computation graph
        grad_fn = getattr(self, "_grad_fn", None)
        if grad_fn is not None:
            grads = grad_fn.apply(gradient)

            # Recursively call backward on parent tensors
            for tensor, grad in zip(grad_fn.saved_tensors, grads):
                if (
                    isinstance(tensor, Tensor)
                    and tensor.requires_grad
                    and grad is not None
                ):
                    tensor.backward(grad)

    def zero_grad(self):
        """Reset gradients to zero."""
        self.grad = None

    # Install enhanced operations
    Tensor.__add__ = tracked_add
    Tensor.__sub__ = tracked_sub
    Tensor.__mul__ = tracked_mul
    Tensor.__truediv__ = tracked_div
    Tensor.__getitem__ = tracked_getitem
    Tensor.matmul = tracked_matmul
    Tensor.transpose = tracked_transpose
    Tensor.reshape = tracked_reshape
    Tensor.sum = sum_op
    Tensor.backward = backward
    Tensor.zero_grad = zero_grad

    # Patch activations and losses to track gradients
    try:
        from tinytorch.core.activations import Sigmoid, ReLU, Softmax, GELU
        from tinytorch.core.losses import (
            BinaryCrossEntropyLoss,
            MSELoss,
            CrossEntropyLoss,
        )

        # Store original methods
        _original_sigmoid_forward = Sigmoid.forward
        _original_relu_forward = ReLU.forward
        _original_softmax_forward = Softmax.forward
        _original_gelu_forward = GELU.forward
        _original_bce_forward = BinaryCrossEntropyLoss.forward
        _original_mse_forward = MSELoss.forward
        _original_ce_forward = CrossEntropyLoss.forward

        def tracked_sigmoid_forward(self, x):
            """Sigmoid with gradient tracking."""
            result_data = 1.0 / (1.0 + np.exp(-x.data))
            result = Tensor(result_data)

            if x.requires_grad:
                result.requires_grad = True
                result._grad_fn = SigmoidBackward(x, result)

            return result

        def tracked_relu_forward(self, x):
            """ReLU with gradient tracking."""
            result_data = np.maximum(0, x.data)
            result = Tensor(result_data)

            if x.requires_grad:
                result.requires_grad = True
                result._grad_fn = ReLUBackward(x)

            return result

        def tracked_softmax_forward(self, x, dim=-1):
            """Softmax with gradient tracking."""
            # Call original forward to get result using Tensor operations
            result = _original_softmax_forward(self, x, dim=dim)

            # Attach the correct gradient function
            if x.requires_grad:
                result.requires_grad = True
                result._grad_fn = SoftmaxBackward(x, result, dim)

            return result

        def tracked_gelu_forward(self, x):
            """GELU with gradient tracking."""
            # Call original forward to get result
            result = _original_gelu_forward(self, x)

            # Attach the correct gradient function
            if x.requires_grad:
                result.requires_grad = True
                result._grad_fn = GELUBackward(x)

            return result

        def tracked_bce_forward(self, predictions, targets):
            """Binary cross-entropy with gradient tracking."""
            # Compute BCE loss
            eps = EPSILON
            clamped_preds = np.clip(predictions.data, eps, 1 - eps)
            log_preds = np.log(clamped_preds)
            log_one_minus_preds = np.log(1 - clamped_preds)
            bce_per_sample = -(
                targets.data * log_preds + (1 - targets.data) * log_one_minus_preds
            )
            bce_loss = np.mean(bce_per_sample)

            result = Tensor(bce_loss)

            if predictions.requires_grad:
                result.requires_grad = True
                result._grad_fn = BCEBackward(predictions, targets)

            return result

        def tracked_mse_forward(self, predictions, targets):
            """MSE loss with gradient tracking."""
            # Compute MSE loss
            diff = predictions.data - targets.data
            squared_diff = diff**2
            mse = np.mean(squared_diff)

            result = Tensor(mse)

            if predictions.requires_grad:
                result.requires_grad = True
                result._grad_fn = MSEBackward(predictions, targets)

            return result

        def tracked_ce_forward(self, logits, targets):
            """Cross-entropy loss with gradient tracking."""
            from tinytorch.core.losses import log_softmax

            # Compute log-softmax for numerical stability
            log_probs = log_softmax(logits, dim=-1)

            # Select log-probabilities for correct classes
            batch_size = logits.shape[0]
            target_indices = targets.data.astype(int)
            selected_log_probs = log_probs.data[np.arange(batch_size), target_indices]

            # Return negative mean
            ce_loss = -np.mean(selected_log_probs)

            result = Tensor(ce_loss)

            if logits.requires_grad:
                result.requires_grad = True
                result._grad_fn = CrossEntropyBackward(logits, targets)

            return result

        # Install patched methods
        Sigmoid.forward = tracked_sigmoid_forward
        ReLU.forward = tracked_relu_forward
        Softmax.forward = tracked_softmax_forward
        GELU.forward = tracked_gelu_forward
        BinaryCrossEntropyLoss.forward = tracked_bce_forward
        MSELoss.forward = tracked_mse_forward
        CrossEntropyLoss.forward = tracked_ce_forward

    except ImportError:
        # Activations/losses not yet available (happens during module development)
        pass

    # Mark as enabled
    Tensor._autograd_enabled = True

    if not quiet:
        print("✅ Autograd enabled! Tensors now track gradients.")
        print("   - Operations build computation graphs")
        print("   - backward() computes gradients")
        print("   - requires_grad=True enables tracking")


# Auto-enable when module is imported
# Always quiet to avoid cluttering user imports
import os

enable_autograd(quiet=True)
