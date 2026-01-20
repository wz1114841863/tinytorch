import numpy as np

from .tensor import Tensor
from .activations import ReLU, Sigmoid

__all__ = [
    "XAVIER_SCALE_FACTOR",
    "HE_SCALE_FACTOR",
    "DROPOUT_MIN_PROB",
    "DROPOUT_MAX_PROB",
    "Layer",
    "Linear",
    "Dropout",
    "Sequential",
]


# Constants for weight initialization
XAVIER_SCALE_FACTOR = 1.0  # Xavier/Glorot initialization uses sqrt(1/fan_in)
HE_SCALE_FACTOR = 2.0  # He initialization uses sqrt(2/fan_in) for ReLU

# Constants for dropout
DROPOUT_MIN_PROB = 0.0  # Minimum dropout probability (no dropout)
DROPOUT_MAX_PROB = 1.0  # Maximum dropout probability (drop everything)


class Layer:
    """Base class for all neural network layers.

    All layers should inherit from this class and implement:
        - forward(x): Compute layer output.
        - parameters(x): Return list of trainable parameters.
    """

    def forward(self, x):
        """Forward pass through the layer.

        ArgsL
            x: Input tensor

        Returns:
            Output tensor after transformation.
        """
        raise NotImplementedError("Subclass must implement forward()")

    def __call__(self, x, *args, **kwargs):
        """Allow layer to be called like a funcion."""
        return self.forward(x, *args, **kwargs)

    def parameters(self):
        """
        Return list of trainable parameters.

        Returns:
            List of Tensor objects (weights and biases)
        """
        return []  # Base class has no parameters

    def __repr__(self):
        """String representation of the layer."""
        return f"{self.__class__.__name__}()"


class Linear(Layer):
    """Linear layer.

    layer: y = xW + b.
    """

    def __init__(self, in_features, out_features, bias=True):
        """Initialize linear layer with proper weight initialization.

        __init__ 的 Docstring

        EXAMPLE:
        >>> layer = Linear(784, 10)  # MNIST classifier final layer
        >>> print(layer.weight.shape)
        (784, 10)
        >>> print(layer.bias.shape)
        (10,)
        """
        self.in_features = in_features
        self.out_features = out_features

        # Xavier/Glorot initialization for stable gradients
        scale = np.sqrt(XAVIER_SCALE_FACTOR / in_features)
        weight_data = np.random.randn(in_features, out_features) * scale
        self.weight = Tensor(weight_data)

        if bias:
            bias_data = np.zeros(out_features)
            self.bias = Tensor(bias_data)
        else:
            self.bias = None

    def forward(self, x):
        """Forward pass through linear layer."""
        output = x.matmul(self.weight)
        if self.bias is not None:
            output = output + self.bias

        return output

    def parameters(self):
        """Return list of trainable parameters."""
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params

    def __repr__(self):
        """String representation for debugging."""
        bias_str = f", bias={self.bias is not None}"
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}{bias_str})"


class Dropout(Layer):
    """Dropout layer for regularization.

    During training: randomly zeros elements with probability p, scales survivors by 1 / (1-p).
    During inference: passed input through unchanged
    """

    def __init__(self, p=0.5):
        """Initialize dropout layer."""
        if not DROPOUT_MIN_PROB <= p <= DROPOUT_MAX_PROB:
            raise ValueError(
                f"Dropout probability must be between {DROPOUT_MIN_PROB} and {DROPOUT_MAX_PROB}, got {p}"
            )
        self.p = p

    def forward(self, x, training=True):
        """Forward pass through dropout layer."""
        if not training or self.p == DROPOUT_MIN_PROB:
            return x

        if self.p == DROPOUT_MAX_PROB:
            return Tensor(np.zeros_like(x.data))

        keep_prob = 1.0 - self.p
        mask = np.random.random(x.data.shape) < keep_prob
        mask_tensor = Tensor(mask.astype(np.float32))
        scale = Tensor(np.array(1.0 / keep_prob))

        output = x * mask_tensor * scale
        return output

    def __call__(self, x, training=True):
        """Allows the layer to be called like a function."""
        return self.forward(x, training)

    def parameters(self):
        """Dropout has no parameters."""
        return []

    def __repr__(self):
        return f"Dropout(p={self.p})"


class Sequential:
    """Containers that chains layers by sequential."""

    def __init__(self, *layers):
        """Initialize with layers to chain together."""
        if len(layers) == 1 and isinstance(layers[0], (list, tuple)):
            self.layers = list(layers[0])
        else:
            self.layers = list(layers)

    def forward(self, x):
        """Forward pass through all layers sequentially."""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def __call__(self, x):
        """Allow model to be called like a function"""
        return self.forward(x)

    def paramters(self):
        """Collect all parameters from all layers."""
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def __repr__(self):
        layer_reprs = ", ".join(repr(layer) for layer in self.layers)
        return f"Sequential({layer_reprs})"
