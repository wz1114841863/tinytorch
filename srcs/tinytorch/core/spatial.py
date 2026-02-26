import numpy as np
import time

from tinytorch.core.tensor import Tensor
from tinytorch.core.autograd import enable_autograd, Function, ReLUBackward

enable_autograd()

# Constants for convolution defaults
DEFAULT_KERNEL_SIZE = 3  # Default kernel size for convolutions
DEFAULT_STRIDE = 1  # Default stride for convolutions
DEFAULT_PADDING = 0  # Default padding for convolutions

# Constants for memory calculations
BYTES_PER_FLOAT32 = 4  # Standard float32 size in bytes
KB_TO_BYTES = 1024  # Kilobytes to bytes conversion
MB_TO_BYTES = 1024 * 1024  # Megabytes to bytes conversion


def validate_4d_input(x, layer_name):
    """Validate that input tensor is 4D (batch, channels, height, width)."""

    if len(x.shape) == 4:
        return  # Valid input

    if len(x.shape) == 3:
        raise ValueError(
            f"{layer_name} expected 4D input (batch, channels, height, width), got 3D: {x.shape}\n"
            f"  Missing batch dimension\n"
            f"  {layer_name} processes batches of images, not single images\n"
            f"  Add batch dim: x.reshape(1, {x.shape[0]}, {x.shape[1]}, {x.shape[2]})"
        )
    elif len(x.shape) == 2:
        raise ValueError(
            f"{layer_name} expected 4D input (batch, channels, height, width), got 2D: {x.shape}\n"
            f"  Got a matrix, expected an image tensor\n"
            f"  {layer_name} needs spatial dimensions (height, width) plus batch and channels\n"
            f"  If this is a flattened image, reshape it: x.reshape(1, channels, height, width)"
        )
    else:
        raise ValueError(
            f"{layer_name} expected 4D input (batch, channels, height, width), got {len(x.shape)}D: {x.shape}\n"
            f"  Wrong number of dimensions\n"
            f"  {layer_name} expects: (batch_size, channels, height, width)\n"
            f"  Reshape your input to 4D with the correct dimensions"
        )


class Conv2dBackward(Function):
    """Gradient computation for 2D convolution."""

    def __init__(self, x, weight, bias, stride, padding, kernel_size, padded_shape):
        if bias is not None:
            super().__init__(x, weight, bias)
        else:
            super().__init__(x, weight)

        self.x = x
        self.weight = weight
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.padded_shape = padded_shape

    def apply(self, grad_output):
        batch_size, out_channels, out_height, out_width = grad_output.shape
        _, in_channels, in_height, in_width = self.x.shape
        kernel_h, kernel_w = self.kernel_size

        # Apply padding to input if needed (for gradient computation)
        if self.padding > 0:
            padded_input = np.pad(
                self.x.data,
                (
                    (0, 0),
                    (0, 0),
                    (self.padding, self.padding),
                    (self.padding, self.padding),
                ),
                mode="constant",
                constant_values=0,
            )
        else:
            padded_input = self.x.data

        # Initialize gradients
        grad_input_padded = np.zeros_like(padded_input)
        grad_weight = np.zeros_like(self.weight.data)
        grad_bias = np.zeros_like(self.bias.data) if self.bias is not None else None

        # Compute gradients using nested loops (inefficient but clear)
        for b in range(batch_size):
            for out_ch in range(out_channels):
                for oh in range(out_height):
                    for ow in range(out_width):
                        grad_out_val = grad_output.data[b, out_ch, oh, ow]

                        # Calculate input region corresponding to this output
                        in_h_start = oh * self.stride
                        in_w_start = ow * self.stride

                        for k_h in range(kernel_h):
                            for k_w in range(kernel_w):
                                for in_ch in range(in_channels):
                                    in_h = in_h_start + k_h
                                    in_w = in_w_start + k_w

                                    # gradient w.r.t weights
                                    grad_weight[out_ch, in_ch, k_h, k_w] += (
                                        padded_input[b, in_ch, in_h, in_w]
                                        * grad_out_val
                                    )

                                    # gradient w.r.t input
                                    grad_input_padded[b, in_ch, in_h, in_w] += (
                                        self.weight.data[out_ch, in_ch, k_h, k_w]
                                        * grad_out_val
                                    )

        # Compute gradient w.r.t. bias (sum over batch and spatial dimensions)
        if grad_bias is not None:
            for out_ch in range(out_channels):
                grad_bias[out_ch] = grad_output[:, out_ch, :, :].sum()

        # Remove padding from input gradient
        if self.padding > 0:
            grad_input = grad_input_padded[
                :, :, self.padding : -self.padding, self.padding : -self.padding
            ]
        else:
            grad_input = grad_input_padded

        # Return gradients as numpy arrays (autograd system handles storage)
        # Following TinyTorch protocol: return (grad_input, grad_weight, grad_bias)
        return grad_input, grad_weight, grad_bias


class Conv2d(Function):
    """Convolutional layer implementation for 2D inputs (images)."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        self.stride = stride
        self.padding = padding

        kernel_h, kernel_w = self.kernel_size
        fan_in = in_channels * kernel_h * kernel_w
        std = np.sqrt(2.0 / fan_in)

        self.weight = Tensor(
            np.random.normal(0, std, (out_channels, in_channels, kernel_h, kernel_w)),
            requires_grad=True,
        )

        if bias:
            self.bias = Tensor(np.zeros(out_channels), requires_grad=True)
        else:
            self.bias = None

    def _compute_output_shape(self, in_h, in_w):
        """Calculate output spatial dimensions for convolution.

        EXAMPLE:
        >>> conv = Conv2d(3, 16, kernel_size=3, padding=1, stride=1)
        >>> oh, ow = conv._compute_output_shape(32, 32)
        >>> print(oh, ow)  # 32, 32 (same padding preserves size)

        >>> conv2 = Conv2d(3, 16, kernel_size=3, padding=0, stride=1)
        >>> oh, ow = conv2._compute_output_shape(32, 32)
        >>> print(oh, ow)  # 30, 30 (shrinks by kernel_size - 1)
        """
        kernel_h, kernel_w = self.kernel_size
        out_h = (in_h + 2 * self.padding - kernel_h) // self.stride + 1
        out_w = (in_w + 2 * self.padding - kernel_w) // self.stride + 1
        return out_h, out_w

    def _apply_padding(self, x):
        """Apply zero-padding to the input tensor.

        EXAMPLE:
        >>> conv = Conv2d(1, 1, kernel_size=3, padding=1)
        >>> x = np.ones((1, 1, 3, 3))
        >>> padded = conv._apply_padding(x)
        >>> print(padded.shape)  # (1, 1, 5, 5) -- 3+2*1=5
        """
        if self.padding > 0:
            return np.pad(
                x,
                (
                    (0, 0),
                    (0, 0),
                    (self.padding, self.padding),
                    (self.padding, self.padding),
                ),
                mode="constant",
                constant_values=0,
            )
        return x

    def _convolve_loops(self, padded, batch_size, out_h, out_w):
        """The core convolution: sliding window dot products over the input.

        LOOP STRUCTURE:
        for b in range(batch_size):
            for out_ch in range(out_channels):
                for oh in range(out_h):
                    for ow in range(out_w):
                        conv_sum = 0.0
                        for k_h in range(kernel_h):
                            for k_w in range(kernel_w):
                                for in_ch in range(in_channels):
                                    conv_sum += padded[b, in_ch, ...] * weight[out_ch, in_ch, ...]
                        output[b, out_ch, oh, ow] = conv_sum
        """
        out_channels = self.out_channels
        in_channels = self.in_channels
        kernel_h, kernel_w = self.kernel_size

        output = np.zeros((batch_size, out_channels, out_h, out_w))

        for b in range(batch_size):
            for out_ch in range(out_channels):
                for oh in range(out_h):
                    for ow in range(out_w):
                        in_h_start = oh * self.stride
                        in_w_start = ow * self.stride
                        conv_sum = 0.0

                        for k_h in range(kernel_h):
                            for k_w in range(kernel_w):
                                for in_ch in range(in_channels):
                                    input_val = padded[
                                        b, in_ch, in_h_start + k_h, in_w_start + k_w
                                    ]
                                    weight_val = self.weight.data[
                                        out_ch, in_ch, k_h, k_w
                                    ]
                                    conv_sum += input_val * weight_val

                        output[b, out_ch, oh, ow] = conv_sum

        return output

    def forward(self, x):
        """Forward pass through Conv2d layer.

        EXAMPLE:
        >>> conv = Conv2d(3, 16, kernel_size=3, padding=1)
        >>> x = Tensor(np.random.randn(2, 3, 32, 32))  # batch=2, RGB, 32x32
        >>> out = conv(x)
        >>> print(out.shape)  # Should be (2, 16, 32, 32)
        """
        validate_4d_input(x, "Conv2D")
        batch_size, in_channels, in_h, in_w = x.shape

        out_height, out_width = self._compute_output_shape(in_h, in_w)
        padded_input = self._apply_padding(x.data)
        output = self._convolve_loops(padded_input, batch_size, out_height, out_width)

        if self.bias is not None:
            for out_ch in range(self.out_channels):
                output[:, out_ch, :, :] += self.bias.data[out_ch]

        # Return Tensor with gradient tracking enabled
        result = Tensor(
            output, requires_grad=(x.requires_grad or self.weight.requires_grad)
        )

        if result.requires_grad:
            result._grad_fn = Conv2dBackward(
                x,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.kernel_size,
                padded_input.shape,
            )

        return result

    def parameters(self):
        """Return trainable parameters."""
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params

    def __call__(self, x):
        """Enable model(x) syntax."""
        return self.forward(x)


class MaxPool2dBackward(Function):
    """Gradient computation for 2D max pooling."""

    def __init__(self, x, output_shape, kernel_size, stride, padding):
        super().__init__(x)
        self.x = x
        self.output_shape = output_shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # Store max positions for gradient routing
        self.max_positions = {}

    def apply(self, grad_output):
        batch_size, channels, in_height, in_width = self.x.shape
        _, _, out_height, out_width = self.output_shape
        kernel_h, kernel_w = self.kernel_size

        # Apply padding if needed
        if self.padding > 0:
            padded_input = np.pad(
                self.x.data,
                (
                    (0, 0),
                    (0, 0),
                    (self.padding, self.padding),
                    (self.padding, self.padding),
                ),
                mode="constant",
                constant_values=-np.inf,
            )
            grad_input_padded = np.zeros_like(padded_input)
        else:
            padded_input = self.x.data
            grad_input_padded = np.zeros_like(self.x.data)

        # Route gradients to max positions
        for b in range(batch_size):
            for c in range(channels):
                for out_h in range(out_height):
                    for out_w in range(out_width):
                        in_h_start = out_h * self.stride
                        in_w_start = out_w * self.stride

                        # Find max position in this window
                        max_val = -np.inf
                        max_h, max_w = 0, 0
                        for k_h in range(kernel_h):
                            for k_w in range(kernel_w):
                                in_h = in_h_start + k_h
                                in_w = in_w_start + k_w
                                val = padded_input[b, c, in_h, in_w]
                                if val > max_val:
                                    max_val = val
                                    max_h, max_w = in_h, in_w

                        # Route gradient to max position
                        grad_input_padded[b, c, max_h, max_w] += grad_output[
                            b, c, out_h, out_w
                        ]

        # Remove padding
        if self.padding > 0:
            grad_input = grad_input_padded[
                :, :, self.padding : -self.padding, self.padding : -self.padding
            ]
        else:
            grad_input = grad_input_padded

        # Return as tuple (following Function protocol)
        return (grad_input,)


class MaxPool2d:
    """2D Max Pooling layer for spatial dimension reduction."""

    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        if stride is None:
            self.stride = self.kernel_size[0]
        else:
            self.stride = stride

        self.padding = padding

    def _compute_pool_output_shape(self, in_h, in_w):
        """Calculate output spatial dimensions for max pooling."""
        kernel_h, kernel_w = self.kernel_size
        out_h = (in_h + 2 * self.padding - kernel_h) // self.stride + 1
        out_w = (in_w + 2 * self.padding - kernel_w) // self.stride + 1
        return out_h, out_w

    def _maxpool_loops(self, padded, batch_size, channels, out_h, out_w):
        """The core max pooling: sliding window max over the input."""
        kernel_h, kernel_w = self.kernel_size
        output = np.zeros((batch_size, channels, out_h, out_w))

        for b in range(batch_size):
            for c in range(channels):
                for oh in range(out_h):
                    for ow in range(out_w):
                        in_h_start = oh * self.stride
                        in_w_start = ow * self.stride

                        max_val = -np.inf
                        for k_h in range(kernel_h):
                            for k_w in range(kernel_w):
                                input_val = padded[
                                    b, c, in_h_start + k_h, in_w_start + k_w
                                ]
                                max_val = max(max_val, input_val)

                        output[b, c, oh, ow] = max_val

        return output

    def forward(self, x):
        """Forward pass through MaxPool2d layer."""
        validate_4d_input(x, "MaxPool2D")
        batch_size, channels, in_h, in_w = x.shape

        out_height, out_width = self._compute_pool_output_shape(in_h, in_w)
        if self.padding > 0:
            padded_input = np.pad(
                x.data,
                (
                    (0, 0),
                    (0, 0),
                    (self.padding, self.padding),
                    (self.padding, self.padding),
                ),
                mode="constant",
                constant_values=-np.inf,
            )
        else:
            padded_input = x.data
        output = self._maxpool_loops(
            padded_input, batch_size, channels, out_height, out_width
        )

        result = Tensor(output, requires_grad=x.requires_grad)

        if result.requires_grad:
            result._grad_fn = MaxPool2dBackward(
                x, result.shape, self.kernel_size, self.stride, self.padding
            )

        return result

    def parameters(self):
        """Return empty list (pooling has no parameters)."""
        return []

    def __call__(self, x):
        """Enable model(x) syntax."""
        return self.forward(x)


class AvgPool2d:
    """2D Average Pooling layer for spatial dimension reduction."""

    def __init__(self, kernel_size, stride=None, padding=0):
        """Initialize AvgPool2d layer."""
        super().__init__()

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        if stride is None:
            self.stride = self.kernel_size[0]
        else:
            self.stride = stride

        self.padding = padding

    def _compute_pool_output_shape(self, in_h, in_w):
        """Calculate output spatial dimensions for pooling."""

        kernel_h, kernel_w = self.kernel_size
        out_height = (in_h + 2 * self.padding - kernel_h) // self.stride + 1
        out_width = (in_w + 2 * self.padding - kernel_w) // self.stride + 1
        return out_height, out_width

    def _avgpool_loops(self, padded, batch_size, channels, out_h, out_w):
        """
        The core average pooling: compute mean of each window.

        TODO: Implement the nested loop average pooling

        APPROACH:
        1. Initialize output array of shape (batch_size, channels, out_h, out_w)
        2. Loop over: batch, channel, output row, output column
        3. For each output position, sum all values in the kernel window
        4. Divide by window area (kernel_h * kernel_w) to get the average
        5. Store the average at the output position

        LOOP STRUCTURE:
        for b in range(batch_size):
            for c in range(channels):
                for oh in range(out_h):
                    for ow in range(out_w):
                        window_sum = 0.0
                        for k_h in range(kernel_h):
                            for k_w in range(kernel_w):
                                window_sum += padded[b, c, ...]
                        output[b, c, oh, ow] = window_sum / (kernel_h * kernel_w)

        HINT: Unlike max pooling, you accumulate a sum and then divide.
        The input position is (oh * stride + k_h, ow * stride + k_w).
        """
        ### BEGIN SOLUTION
        kernel_h, kernel_w = self.kernel_size
        output = np.zeros((batch_size, channels, out_h, out_w))

        for b in range(batch_size):
            for c in range(channels):
                for oh in range(out_h):
                    for ow in range(out_w):
                        in_h_start = oh * self.stride
                        in_w_start = ow * self.stride

                        window_sum = 0.0
                        for k_h in range(kernel_h):
                            for k_w in range(kernel_w):
                                input_val = padded[
                                    b, c, in_h_start + k_h, in_w_start + k_w
                                ]
                                window_sum += input_val

                        output[b, c, oh, ow] = window_sum / (kernel_h * kernel_w)

        return output

    def forward(self, x):
        """Forward pass through AvgPool2d layer.

        EXAMPLE:
        >>> pool = AvgPool2d(kernel_size=2, stride=2)
        >>> x = Tensor(np.random.randn(1, 3, 8, 8))
        >>> out = pool(x)
        >>> print(out.shape)  # Should be (1, 3, 4, 4)
        """
        validate_4d_input(x, "AvgPool2d")
        batch_size, channels, in_height, in_width = x.shape
        out_height, out_width = self._compute_pool_output_shape(in_height, in_width)

        if self.padding > 0:
            padded_input = np.pad(
                x.data,
                (
                    (0, 0),
                    (0, 0),
                    (self.padding, self.padding),
                    (self.padding, self.padding),
                ),
                mode="constant",
                constant_values=0,
            )
        else:
            padded_input = x.data

        output = self._avgpool_loops(
            padded_input, batch_size, channels, out_height, out_width
        )

        result = Tensor(output, requires_grad=x.requires_grad)
        return result


    def parameters(self):
        """Return empty list (pooling has no parameters)."""
        return []

    def __call__(self, x):
        """Enable model(x) syntax."""
        return self.forward(x)


class BatchNorm2d:
    """Batch Normalization for 2D spatial inputs (images)."""

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Learnable parameters
        self.gamma = Tensor(np.ones(num_features), requires_grad=True)
        self.beta = Tensor(np.zeros(num_features), requires_grad=True)

        # Running estimates (not learnable)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        self.training = True

    def train(self):
        """Set layer to training mode."""
        self.training = True
        return self

    def eval(self):
        """Set layer to evaluation mode."""
        self.training = False
        return self

    def _validate_input(self, x):
        if len(x.shape) != 4:
            if len(x.shape) == 3:
                raise ValueError(
                    f"BatchNorm2d expected 4D input (batch, channels, height, width), got 3D: {x.shape}\n"
                    f"  ❌ Missing batch dimension\n"
                    f"  💡 BatchNorm2d computes statistics over the batch dimension\n"
                    f"  🔧 Add batch dim: x.reshape(1, {x.shape[0]}, {x.shape[1]}, {x.shape[2]})"
                )
            elif len(x.shape) == 2:
                raise ValueError(
                    f"BatchNorm2d expected 4D input (batch, channels, height, width), got 2D: {x.shape}\n"
                    f"  ❌ Got a matrix, expected an image tensor\n"
                    f"  💡 BatchNorm2d normalizes over spatial dimensions per channel\n"
                    f"  🔧 If this is a flattened image, reshape it: x.reshape(1, channels, height, width)"
                )
            else:
                raise ValueError(
                    f"BatchNorm2d expected 4D input (batch, channels, height, width), got {len(x.shape)}D: {x.shape}\n"
                    f"  ❌ Wrong number of dimensions\n"
                    f"  💡 BatchNorm2d expects: (batch_size, channels, height, width)\n"
                    f"  🔧 Reshape your input to 4D with the correct dimensions"
                )

        batch_size, channels, height, width = x.shape

        if channels != self.num_features:
            raise ValueError(
                f"BatchNorm2d channel mismatch: expected {self.num_features} channels, got {channels}\n"
                f"  ❌ Input has {channels} channels but BatchNorm2d was created for {self.num_features}\n"
                f"  💡 BatchNorm2d(num_features) must match the channel dimension of your input\n"
                f"  🔧 Either fix your input shape or create BatchNorm2d({channels})"
            )

    def _get_stats(self, x):
        """Compute mean and variance for batch normalization."""

        if self.training:
            # Compute batch statistics per channel
            # Mean over batch and spatial dimensions: axes (0, 2, 3)
            batch_mean = np.mean(x.data, axis=(0, 2, 3))  # Shape: (C,)
            batch_var = np.var(x.data, axis=(0, 2, 3))  # Shape: (C,)

            # Update running statistics (exponential moving average)
            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * batch_mean
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * batch_var

            return batch_mean, batch_var
        else:
            # Use running statistics (frozen during eval)
            return self.running_mean, self.running_var

    def forward(self, x):
        """Forward pass through BatchNorm2d.

        EXAMPLE:
        >>> bn = BatchNorm2d(16)
        >>> x = Tensor(np.random.randn(2, 16, 8, 8))
        >>> y = bn(x)
        >>> print(y.shape)  # (2, 16, 8, 8)

        """
        ### BEGIN SOLUTION
        self._validate_input(x)

        batch_size, channels, height, width = x.shape
        mean, var = self._get_stats(x)

        # Normalize: (x - mean) / sqrt(var + eps)
        # Reshape mean and var for broadcasting: (C,) -> (1, C, 1, 1)
        mean_reshaped = mean.reshape(1, channels, 1, 1)
        var_reshaped = var.reshape(1, channels, 1, 1)

        x_normalized = (x.data - mean_reshaped) / np.sqrt(var_reshaped + self.eps)

        # Apply scale (gamma) and shift (beta)
        gamma_reshaped = self.gamma.data.reshape(1, channels, 1, 1)
        beta_reshaped = self.beta.data.reshape(1, channels, 1, 1)

        output = gamma_reshaped * x_normalized + beta_reshaped

        # Return Tensor with gradient tracking
        result = Tensor(
            output, requires_grad=x.requires_grad or self.gamma.requires_grad
        )

        return result

    def parameters(self):
        """Return learnable parameters (gamma and beta)."""
        return [self.gamma, self.beta]

    def __call__(self, x):
        """Enable model(x) syntax."""
        return self.forward(x)


class SimpleCNN:
    """Simple CNN demonstrating spatial operations integration.

    Architecture:
    - Conv2d(3→16, 3×3) + ReLU + MaxPool(2×2)
    - Conv2d(16→32, 3×3) + ReLU + MaxPool(2×2)
    - Flatten + Linear(features→num_classes)
    """

    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = MaxPool2d(kernel_size=2, stride=2)

        # Calculate flattened size
        self.flattened_size = 32 * 8 * 8

        self.num_classes = num_classes
        self.flattened_size = 32 * 8 * 8

    def forward(self, x):
        """Forward pass through SimpleCNN."""

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)

        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)

        return x

    def relu(self, x):
        """ReLU activation with gradient tracking for CNN."""
        result_data = np.maximum(0, x.data)
        result = Tensor(result_data)
        if x.requires_grad:
            result.requires_grad = True
            result._grad_fn = ReLUBackward(x)
        return result

    def parameters(self):
        """Return all trainable parameters."""
        params = []
        params.extend(self.conv1.parameters())
        params.extend(self.conv2.parameters())
        return params

    def __call__(self, x):
        """Enable model(x) syntax."""
        return self.forward(x)
