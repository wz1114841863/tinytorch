import sys
import os
import time
import numpy as np
import tracemalloc
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import gc

from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear
from tinytorch.core.spatial import Conv2d

# Constants for memory and performance measurement
BYTES_PER_FLOAT32 = 4  # Standard float32 size in bytes
KB_TO_BYTES = 1024  # Kilobytes to bytes conversion
MB_TO_BYTES = 1024 * 1024  # Megabytes to bytes conversion


class Profiler:
    """Professional-grade ML model profiler for performance analysis."""

    def __init__(self):
        """Initialize profiler with measurement state."""
        self.measurements = {}
        self.operation_counts = defaultdict(int)
        self.memory_tracker = None

    def _count_layer_parameters(self, layer):
        """Count parameters in a given layer."""
        params = 0
        if hasattr(layer, "weight"):
            params += layer.weight.data.size
            if hasattr(layer, "bias") and layer.bias is not None:
                params += layer.bias.data.size
        return params

    def count_parameters(self, model):
        """Count total parameters in the model."""
        if hasattr(model, "layers"):
            return sum(
                p.data.size for layer in model.layers for p in layer.parameters()
            )
        elif hasattr(model, "parameters"):
            return sum(p.data.size for p in model.parameters())
        elif hasattr(model, "weight"):
            return self._count_layer_parameters(model)
        return 0

    def _count_linear_flops(self, model, input_shape):
        """Count FLOPs for a Linear layer.

        Linear FLOP Formula:
        FLOPs = in_features × out_features × 2
                     ↑              ↑          ↑
              Input dimension  Output dimension  Multiply + Add
        """
        in_features = input_shape[-1]
        out_features = model.weight.shape[1] if hasattr(model, "weight") else 1
        return 2 * in_features * out_features  # Multiply and add per output feature

    def _count_conv_flops(self, model, input_shape):
        """Count FLOPs for a Conv2d layer.

        Conv2d FLOP Formula:
        FLOPs = 2 × C_in × K_h × K_w × H_out × W_out
                     ↑      ↑     ↑     ↑       ↑       ↑
              Input channels  Kernel height  Kernel width  Output height  Output width
        """
        if not (hasattr(model, "kernel_size") and hasattr(model, "in_channels")):
            return 0

        in_channels = model.in_channels
        out_channels = model.out_channels
        kernel_h = kernel_w = model.kernel_size

        input_h, input_w = input_shape[-2], input_shape[-1]
        stride = model.stride if hasattr(model, "stride") else 1
        output_h = input_h // stride
        output_w = input_w // stride

        return (
            output_h * output_w * kernel_h * kernel_w * in_channels * out_channels * 2
        )

    def _count_sequential_flops(self, model, input_shape):
        """Count FLOPs for a Sequential model by summing per-layer FLOPs."""
        total_flops = 0
        current_shape = input_shape
        for layer in model.layers:
            total_flops += self.count_flops(layer, current_shape)
            if hasattr(layer, "weight"):
                current_shape = current_shape[:-1] + (layer.weight.shape[1],)
        return total_flops

    def count_flops(self, model, input_shape):
        """count FLOPS for one forward pass."""
        model_name = model.__class__.__name__

        if model_name == "Linear":
            return self._count_linear_flops(model, input_shape)
        elif model_name == "Conv2d":
            return self._count_conv_flops(model, input_shape)
        elif model_name == "Sequential" or hasattr(model, "layers"):
            return self._count_sequential_flops(model, input_shape)
        else:
            return int(np.prod(input_shape))

    def _calculate_parameter_memory(self, model):
        """Calculate memory used by model parameters in megabytes."""
        param_count = self.count_parameters(model)
        return (param_count * BYTES_PER_FLOAT32) / MB_TO_BYTES

    def calculate_memory_efficiency(self, useful_memory_mb, peak_memory_mb):
        """Calculate memory efficiency as a percentage."""
        ratio = useful_memory_mb / max(peak_memory_mb, 0.001)
        return min(ratio, 1.0)

    def measure_memory(self, model, input_shape):
        """Measure memory usage during a forward pass."""
        tracemalloc.start()
        _baseline_memory = tracemalloc.get_traced_memory()[0]

        parameter_memory_mb = self._calculate_parameter_memory(model)

        dummy_input = Tensor(np.random.randn(*input_shape))
        activation_memory_mb = (dummy_input.data.nbytes * 2) / MB_TO_BYTES

        _ = model.forward(dummy_input)

        _current_memory, peak_memory = tracemalloc.get_traced_memory()
        peak_memory_mb = (peak_memory - _baseline_memory) / MB_TO_BYTES
        tracemalloc.stop()

        useful_memory = parameter_memory_mb + activation_memory_mb
        return {
            "parameter_memory_mb": parameter_memory_mb,
            "activation_memory_mb": activation_memory_mb,
            "peak_memory_mb": max(peak_memory_mb, useful_memory),
            "memory_efficiency": self._calculate_memory_efficiency(
                useful_memory, peak_memory_mb
            ),
        }

    def measure_latency(
        self, model, input_tensor, warmup: int = 10, iterations: int = 100
    ):
        """M Measure model inference latency with statistical rigor."""
        for _ in range(warmup):
            _ = model.forward(input_tensor)

        times = []
        for _ in range(iterations):
            start_time = time.perf_counter()
            _ = model.forward(input_tensor)
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to milliseconds

        # Calculate statistics - use median for robustness
        times = np.array(times)
        median_latency = np.median(times)

        return float(median_latency)

    def profile_layer(self, layer, input_shape):
        dummy_input = Tensor(np.random.randn(*input_shape))

        params = self.count_parameters(layer)
        flops = self.count_flops(layer, input_shape)
        memory = self.measure_memory(layer, input_shape)
        latency = self.measure_latency(layer, dummy_input, warmup=3, iterations=10)
        gflops_per_second = (flops / 1e9) / max(latency / 1000, 1e-6)

        return {
            "layer_type": layer.__class__.__name__,
            "parameters": params,
            "flops": flops,
            "latency_ms": latency,
            "gflops_per_second": gflops_per_second,
            **memory,
        }

    def _compute_derived_metrics(self, flops, latency_ms, peak_memory_mb):
        """Compute throughput and efficiency metrics from raw measurements."""
        latency_seconds = latency_ms / 1000.0
        gflops_per_second = (flops / 1e9) / max(latency_seconds, 1e-6)
        memory_bandwidth = peak_memory_mb / max(latency_seconds, 1e-6)
        theoretical_peak_gflops = 100.0
        computational_efficiency = min(gflops_per_second / theoretical_peak_gflops, 1.0)

        return {
            "gflops_per_second": gflops_per_second,
            "memory_bandwidth_mbs": memory_bandwidth,
            "computational_efficiency": computational_efficiency,
        }

    def _analyze_bottleneck(self, gflops_per_second, memory_bandwidth_mbs):
        """Analyze whether the bottleneck is compute or memory bound."""
        is_memory_bound = memory_bandwidth_mbs > gflops_per_second * 100
        return {
            "is_memory_bound": is_memory_bound,
            "is_compute_bound": not is_memory_bound,
            "bottleneck": "memory" if is_memory_bound else "compute",
        }

    def profile_forward_pass(self, model, input_tensor):
        """Comprehensive profiling of a model's forward pass.

        EXAMPLE:
        >>> model = Linear(256, 128)
        >>> input_data = Tensor(np.random.randn(32, 256))
        >>> profiler = Profiler()
        >>> profile = profiler.profile_forward_pass(model, input_data)
        >>> print(f"Throughput: {profile['gflops_per_second']:.2f} GFLOP/s")
        Throughput: 2.45 GFLOP/s
        """
        param_count = self.count_parameters(model)
        flops = self.count_flops(model, input_tensor.shape)
        memory_stats = self.measure_memory(model, input_tensor.shape)
        latency_ms = self.measure_latency(model, input_tensor, warmup=5, iterations=20)

        derived = self._compute_derived_metrics(
            flops, latency_ms, memory_stats["peak_memory_mb"]
        )
        bottleneck = self._analyze_bottleneck(
            derived["gflops_per_second"], derived["memory_bandwidth_mbs"]
        )

        return {
            "parameters": param_count,
            "flops": flops,
            "latency_ms": latency_ms,
            **memory_stats,
            **derived,
            **bottleneck,
        }

    def _estimate_backward_costs(self, forward_flops, forward_latency_ms):
        """Estimate backward pass costs based on forward pass metrics."""
        return {
            "backward_flops": forward_flops * 2,
            "backward_latency_ms": forward_latency_ms * 2,
        }

    def _estimate_optimizer_memory(self, gradient_memory_mb):
        """Estimate additional memory used by optimizer states."""
        return {
            "sgd": 0,
            "adam": gradient_memory_mb * 2,
            "adamw": gradient_memory_mb * 2,
        }

    def profile_backward_pass(self, model, input_tensor, _loss_fn=None):
        """Profile both forward and backward passes for training analysis.

        EXAMPLE:
        >>> model = Linear(128, 64)
        >>> input_data = Tensor(np.random.randn(16, 128))
        >>> profiler = Profiler()
        >>> profile = profiler.profile_backward_pass(model, input_data)
        >>> print(f"Training iteration: {profile['total_latency_ms']:.2f} ms")
        Training iteration: 0.45 ms
        """
        fwd = self.profile_forward_pass(model, input_tensor)
        bwd = self._estimate_backward_costs(fwd["flops"], fwd["latency_ms"])

        gradient_memory_mb = fwd["parameter_memory_mb"]
        total_flops = fwd["flops"] + bwd["backward_flops"]
        total_latency_ms = fwd["latency_ms"] + bwd["backward_latency_ms"]
        total_memory_mb = (
            fwd["parameter_memory_mb"]
            + fwd["activation_memory_mb"]
            + gradient_memory_mb
        )

        return {
            "forward_flops": fwd["flops"],
            "forward_latency_ms": fwd["latency_ms"],
            "forward_memory_mb": fwd["peak_memory_mb"],
            **bwd,
            "gradient_memory_mb": gradient_memory_mb,
            "total_flops": total_flops,
            "total_latency_ms": total_latency_ms,
            "total_memory_mb": total_memory_mb,
            "total_gflops_per_second": (total_flops / 1e9)
            / (total_latency_ms / 1000.0),
            "optimizer_memory_estimates": self._estimate_optimizer_memory(
                gradient_memory_mb
            ),
            "memory_efficiency": fwd["memory_efficiency"],
            "bottleneck": fwd["bottleneck"],
        }


def quick_profile(model, input_tensor, profiler=None):
    """Quick profiling function for immediate insights.

    Example:
        >>> model = Linear(128, 64)
        >>> input_data = Tensor(np.random.randn(16, 128))
        >>> results = quick_profile(model, input_data)
        >>> # Displays formatted output automatically
    """
    if profiler is None:
        profiler = Profiler()

    profile = profiler.profile_forward_pass(model, input_tensor)

    # Display formatted results
    print("🧪 Quick Profile Results:")
    print(f"   Parameters: {profile['parameters']:,}")
    print(f"   FLOPs: {profile['flops']:,}")
    print(f"   Latency: {profile['latency_ms']:.2f} ms")
    print(f"   Memory: {profile['peak_memory_mb']:.2f} MB")
    print(f"   Bottleneck: {profile['bottleneck']}")
    print(f"   Efficiency: {profile['computational_efficiency']*100:.1f}%")

    return profile


def analyze_weight_distribution(model, percentiles=[10, 25, 50, 75, 90]):
    """Analyze weight distribution across layers.

    Example:
        >>> model = Linear(512, 512)
        >>> stats = analyze_weight_distribution(model)
        >>> print(f"Weights < 0.01: {stats['below_threshold_001']:.1f}%")
    """
    weights = []
    if hasattr(model, "parameters"):
        for param in model.parameters():
            weights.extend(param.data.flatten().tolist())
    elif hasattr(model, "weight"):
        weights.extend(model.weight.data.flatten().tolist())
    else:
        return {"error": "No weights found"}

    weights = np.array(weights)
    abs_weights = np.abs(weights)
    stats = {
        "total_weights": len(weights),
        "mean": float(np.mean(abs_weights)),
        "std": float(np.std(abs_weights)),
        "min": float(np.min(abs_weights)),
        "max": float(np.max(abs_weights)),
    }

    for p in percentiles:
        stats[f"percentile_{p}"] = float(np.percentile(abs_weights, p))

    for threshold in [0.001, 0.01, 0.1]:
        below = np.sum(abs_weights < threshold) / len(weights) * 100
        stats[f'below_threshold_{str(threshold).replace(".", "")}'] = below

    return stats
