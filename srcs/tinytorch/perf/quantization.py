import numpy as np
import time
import warnings
from typing import Tuple, Dict, List, Optional

from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear, Sequential
from tinytorch.core.activations import ReLU

INT8_MIN_VALUE = -128
INT8_MAX_VALUE = 127
INT8_RANGE = 256  # Number of possible INT8 values (from -128 to 127 inclusive)
EPSILON = 1e-8  # Small value for numerical stability (constant tensor detection)

BYTES_PER_FLOAT32 = 4  # Standard float32 size in bytes
BYTES_PER_INT8 = 1  # INT8 size in bytes
MB_TO_BYTES = 1024 * 1024  # Megabytes to bytes conversion


def quantize_int8(tensor):
    """Quantize FP32 tensor to INT8 using symmetric quantization."""
    data = tensor.data
    min_val = float(np.min(data))
    max_val = float(np.max(data))

    if abs(max_val - min_val) < EPSILON:
        # If the tensor is effectively constant, we can set scale to 1 and zero_point to 0
        scale = 1.0
        zero_point = 0
        quantized_data = np.zeros_like(data, dtype=np.int8)
        return Tensor(quantized_data), scale, zero_point

    scale = (max_val - min_val) / (INT8_RANGE - 1)
    zero_point = int(np.round(INT8_MIN_VALUE - min_val / scale))
    zero_point = int(np.clip(zero_point, INT8_MIN_VALUE, INT8_MAX_VALUE))

    quantized_data = np.round(data / scale + zero_point)
    quantized_data = np.clip(quantized_data, INT8_MIN_VALUE, INT8_MAX_VALUE).astype(
        np.int8
    )
    return Tensor(quantized_data), scale, zero_point


def dequantize_int8(q_tensor, scale, zero_point):
    """Dequantize INT8 tensor back to FP32."""
    q_data = q_tensor.data.astype(np.float32)
    dequantized_data = (q_data - zero_point) * scale
    return Tensor(dequantized_data)


class QuantizedLinear:

    def __init__(self, linear_layer):
        """Initialize the quantized linear layer by quantizing weights and biases.

        EXAMPLE:
        >>> original_layer = Linear(128, 64)
        >>> original_layer.weight = Tensor(np.random.randn(128, 64) * 0.1)
        >>> original_layer.bias = Tensor(np.random.randn(64) * 0.01)
        >>> quantized_layer = QuantizedLinear(original_layer)
        >>> print(quantized_layer.q_weight.data.dtype)
        int8
        """
        self.original_layer = linear_layer
        self.q_weight, self.weight_scale, self.weight_zero_point = quantize_int8(
            linear_layer.weight
        )
        if linear_layer.bias is not None:
            self.q_bias, self.bias_scale, self.bias_zero_point = quantize_int8(
                linear_layer.bias
            )
        else:
            self.q_bias = None
            self.bias_scale = None
            self.bias_zero_point = None

        # Store input quantization parameters.
        self.input_scale = None
        self.input_zero_point = None

    def calibrate(self, sample_inputs):
        """Calibrate input quantization parameters using sample data.

        EXAMPLE:
        >>> layer = QuantizedLinear(Linear(64, 32))
        >>> sample_data = [Tensor(np.random.randn(1, 64)) for _ in range(10)]
        >>> layer.calibrate(sample_data)
        >>> print(layer.input_scale is not None)
        True
        """
        all_values = []
        for inp in sample_inputs:
            all_values.append(inp.data.flatten())
        all_valuse = np.array(all_values)
        min_val = float(np.min(all_values))
        max_val = float(np.max(all_values))

        if abs(max_val - min_val) < EPSILON:
            self.input_scale = 1.0
            self.input_zero_point = 0
        else:
            self.input_scale = (max_val - min_val) / (INT8_RANGE - 1)
            self.input_zero_point = int(
                np.round(INT8_MIN_VALUE - min_val / self.input_scale)
            )
            self.input_zero_point = int(
                np.clip(self.input_zero_point, INT8_MIN_VALUE, INT8_MAX_VALUE)
            )

    def forward(self, x):
        """Forward pass with quantized computation."""
        # just for education
        weight_fp32 = dequantize_int8(
            self.q_weight, self.weight_scale, self.weight_zero_point
        )
        result = x.matmul(weight_fp32)
        if self.q_bias is not None:
            bias_fp32 = dequantize_int8(
                self.q_bias, self.bias_scale, self.bias_zero_point
            )
            result = Tensor(result.data + bias_fp32.data)

        return result

    def __call__(self, x):
        """Allows the quantized linear layer to be called like a function."""
        return self.forward(x)

    def parameters(self):
        """Return quantized parameters."""
        params = [self.q_weight]
        if self.q_bias is not None:
            params.append(self.q_bias)
        return params

    def memory_usage(self):
        """Calculate memory usage in bytes."""
        # Original FP32 usage
        original_weight_bytes = self.original_layer.weight.data.size * BYTES_PER_FLOAT32
        original_bias_bytes = 0
        if self.original_layer.bias is not None:
            original_bias_bytes = self.original_layer.bias.data.size * BYTES_PER_FLOAT32

        # Quantized INT8 usage
        quantized_weight_bytes = self.q_weight.data.size * BYTES_PER_INT8
        quantized_bias_bytes = 0
        if self.q_bias is not None:
            quantized_bias_bytes = self.q_bias.data.size * BYTES_PER_INT8

        # Add overhead for scales and zero points (small)
        # 2 floats: one scale for weights, one scale for bias (if present)
        overhead_bytes = BYTES_PER_FLOAT32 * 2

        quantized_total = quantized_weight_bytes + quantized_bias_bytes + overhead_bytes
        original_total = original_weight_bytes + original_bias_bytes

        return {
            "original_bytes": original_total,
            "quantized_bytes": quantized_total,
            "compression_ratio": (
                original_total / quantized_total if quantized_total > 0 else 1.0
            ),
        }


def _collect_layer_inputs(model, layer_index, calibration_data, max_samples=100):
    """Collect input samples for a specific layer from the model using calibration data."""
    sample_inputs = []
    for data in calibration_data[:max_samples]:
        x = data
        for j in range(layer_index):
            x = model.layers[j].forward(x)
        sample_inputs.append(x)
    return sample_inputs


def _quantize_single_layer(layer, calibration_inputs):
    """Quantize a single layer and return the quantized version."""
    quantized_layer = QuantizedLinear(layer)

    if calibration_inputs is not None:
        quantized_layer.calibrate(calibration_inputs)

    return quantized_layer


def quantize_model(model, calibration_data=None):
    """Quantize an entire Sequential model layer by layer.

    EXAMPLE:
    >>> model = Sequential(Linear(128, 64), ReLU(), Linear(64, 32))
    >>> calibration_data = [Tensor(np.random.randn(1, 128)) for _ in range(10)]
    >>> quantized_model = quantize_model(model, calibration_data)
    >>> print(isinstance(quantized_model.layers[0], QuantizedLinear))
    True
    """
    if hasattr(model, "layers"):
        for i, layer in enumerate(model.layers):
            if isinstance(layer, Linear):
                cal_inputs = None
                if calibration_data is not None:
                    cal_inputs = _collect_layer_inputs(model, i, calibration_data)

                model.layers[i] = _quantize_single_layer(layer, cal_inputs)
    elif isinstance(model, Linear):
        raise ValueError(
            f"Cannot quantize single Linear layer in-place\n"
            f"  ❌ quantize_model() modifies models in-place, but a single layer has no container to modify\n"
            f"  💡 In-place modification requires a container (like Sequential) that holds layer references\n"
            f"  🔧 Use QuantizedLinear directly: quantized_layer = QuantizedLinear(your_linear_layer)"
        )

    else:
        raise ValueError(
            f"Unsupported model type for quantization: {type(model).__name__}\n"
            f"  ❌ quantize_model() expects a model with .layers attribute (like Sequential)\n"
            f"  💡 The function iterates through model.layers to find and replace Linear layers\n"
            f"  🔧 Wrap your layers in Sequential: model = Sequential(layer1, activation, layer2)"
        )


def _measure_layer_bytes(layer, is_quantized):
    """Measure the memory usage of a layer's parameters in bytes."""
    if is_quantized and isinstance(layer, QuantizedLinear):
        memory_info = layer.memory_usage()
        param_count = sum(p.data.size for p in layer.parameters())
        return param_count, memory_info["quantized_bytes"]

    if hasattr(layer, "parameters"):
        params = layer.parameters()
        param_count = sum(p.data.size for p in params)
        byte_count = param_count * BYTES_PER_FLOAT32
        return param_count, byte_count

    return 0, 0


def analyze_model_sizes(original_model, quantized_model):
    """Analyze and compare the sizes of the original and quantized models."""
    # Measure original model
    original_params = 0
    original_bytes = 0
    for layer in original_model.layers:
        p, b = _measure_layer_bytes(layer, is_quantized=False)
        original_params += p
        original_bytes += b

    # Measure quantized model
    quantized_params = 0
    quantized_bytes = 0
    for layer in quantized_model.layers:
        is_q = isinstance(layer, QuantizedLinear)
        p, b = _measure_layer_bytes(layer, is_quantized=is_q)
        quantized_params += p
        quantized_bytes += b

    compression_ratio = original_bytes / quantized_bytes if quantized_bytes > 0 else 1.0
    memory_saved = original_bytes - quantized_bytes

    return {
        "original_params": original_params,
        "quantized_params": quantized_params,
        "original_bytes": original_bytes,
        "quantized_bytes": quantized_bytes,
        "compression_ratio": compression_ratio,
        "memory_saved_mb": memory_saved / MB_TO_BYTES,
        "memory_saved_percent": (
            (memory_saved / original_bytes) * 100 if original_bytes > 0 else 0
        ),
    }


class Quantizer:
    """Utility class for quantizing models and analyzing size reductions."""

    @staticmethod
    def quantize_tensor(tensor):
        return quantize_int8(tensor)

    @staticmethod
    def dequantize_tensor(q_tensor, scale, zero_point):
        return dequantize_int8(q_tensor, scale, zero_point)

    @staticmethod
    def quantize_model(model, calibration_data):
        """Quantize all Linear layers in a model and return stats.

        Unlike the standalone quantize_model() which modifies in-place,
        this returns a dictionary with quantization info for benchmarking.
        """
        quantized_layers = {}
        original_size = 0
        total_elements = 0
        param_idx = 0

        # Iterate through model parameters
        for layer in model.layers:
            for param in layer.parameters():
                param_size = param.data.nbytes
                original_size += param_size
                total_elements += param.data.size

                # Quantize parameter using the standalone function
                q_param, scale, zp = quantize_int8(param)

                quantized_layers[f"param_{param_idx}"] = {
                    "quantized": q_param,
                    "scale": scale,
                    "zero_point": zp,
                    "original_shape": param.data.shape,
                }
                param_idx += 1

        # INT8 uses 1 byte per element
        quantized_size = total_elements

        return {
            "quantized_layers": quantized_layers,
            "original_size_mb": original_size / MB_TO_BYTES,
            "quantized_size_mb": quantized_size / MB_TO_BYTES,
            "compression_ratio": (
                original_size / quantized_size if quantized_size > 0 else 1.0
            ),
        }

    @staticmethod
    def compare_models(original_model, quantized_info):
        """Compare memory usage between original and quantized models."""
        return {
            "original_mb": quantized_info["original_size_mb"],
            "quantized_mb": quantized_info["quantized_size_mb"],
            "compression_ratio": quantized_info["compression_ratio"],
            "memory_saved_mb": quantized_info["original_size_mb"]
            - quantized_info["quantized_size_mb"],
        }


def verify_quantization_works(original_model, quantized_model):
    """
    Verify quantization actually reduces memory using real .nbytes measurements.

    Example:
        >>> original = Sequential(Linear(100, 50))
        >>> quantized = Sequential(Linear(100, 50))
        >>> quantize_model(quantized)
        >>> results = verify_quantization_works(original, quantized)
        >>> assert results['actual_reduction'] >= 3.5  # Real 4× reduction
    """
    print("Verifying actual memory reduction with .nbytes...")

    # Collect actual bytes from original FP32 model
    original_bytes = sum(
        param.data.nbytes
        for param in original_model.parameters()
        if hasattr(param, "data") and hasattr(param.data, "nbytes")
    )

    # Collect actual bytes from quantized INT8 model
    quantized_bytes = sum(
        layer.q_weight.data.nbytes
        + (layer.q_bias.data.nbytes if layer.q_bias is not None else 0)
        for layer in quantized_model.layers
        if isinstance(layer, QuantizedLinear)
    )

    # Calculate actual reduction
    actual_reduction = original_bytes / max(quantized_bytes, 1)

    # Display results
    print(f"   Original model: {original_bytes / MB_TO_BYTES:.2f} MB (FP32)")
    print(f"   Quantized model: {quantized_bytes / MB_TO_BYTES:.2f} MB (INT8)")
    print(f"   Actual reduction: {actual_reduction:.1f}x")
    print(
        f"   {'PASS' if actual_reduction >= 3.5 else 'FAIL'} Meets 4x reduction target"
    )

    # Verify target met
    assert (
        actual_reduction >= 3.5
    ), f"Expected ~4x reduction, got {actual_reduction:.1f}x"

    print(
        f"\nVERIFIED: Quantization achieves real {actual_reduction:.1f}x memory reduction!"
    )

    return {
        "actual_reduction": actual_reduction,
        "original_mb": original_bytes / MB_TO_BYTES,
        "quantized_mb": quantized_bytes / MB_TO_BYTES,
        "verified": actual_reduction >= 3.5,
    }
