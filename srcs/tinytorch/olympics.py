import numpy as np
import time
import json
import platform
import sys

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear
from tinytorch.core.activations import GELU, ReLU


class SimpleMLP:
    """Simple 2-layer MLP for benchmarking demonstration."""

    def __init__(self, input_size=10, hidden_size=20, output_size=3):
        """Initialize simple MLP with random weights."""
        self.fc1 = Linear(input_size, hidden_size)
        self.relu = ReLU()
        self.fc2 = Linear(hidden_size, output_size)

        self.fc1.weight.data = np.random.randn(input_size, hidden_size) * 0.01
        self.fc1.bias.data = np.zeros(hidden_size)
        self.fc2.weight.data = np.random.randn(hidden_size, output_size) * 0.01
        self.fc2.bias.data = np.zeros(output_size)

    def forward(self, x):
        """Forward pass through the network"""
        x = self.fc1.forward(x)
        x = self.relu.forward(x)
        x = self.fc2.forward(x)
        return x

    def parameters(self):
        """Return model parameters for perf."""
        return [self.fc1.weight, self.fc1.bias, self.fc2.weight, self.fc2.bias]

    def count_parameters(self):
        """Count total number of parameters."""
        total = 0
        for param in self.parameters():
            total += param.data.size
        return total


class BenchmarkReport:
    """Benchmark report for model performace."""

    def __init__(self, model_name="model"):
        self.model_name = model_name
        self.metrics = {}
        self.system_info = self._get_system_info()
        self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    def _get_system_info(self):
        """Collect system information for reproducibility."""
        return {
            "platform": platform.platform(),
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
        }

    def benchmark_model(self, model, X_test, y_test, num_runs=100):
        """Benchmark model performance comprehensively."""
        param_count = model.count_parameters()
        model_size_mb = (param_count * 4) / (1024 * 1024)  # Assuming FP32

        # Measure accuracy
        predictions = model.forward(X_test)
        pred_labels = np.argmax(predictions.data, axis=1)
        accuracy = np.mean(pred_labels == y_test)

        # Measure latency (average over multiple runs)
        # Why multiple runs? See "Variance" section in Foundations
        latencies = []
        for _ in range(num_runs):
            start = time.time()
            _ = model.forward(X_test[:1])  # Single sample inference
            latencies.append((time.time() - start) * 1000)  # Convert to ms

        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)

        # Store metrics (all as Python native types for JSON serialization)
        self.metrics = {
            "parameter_count": int(param_count),
            "model_size_mb": float(model_size_mb),
            "accuracy": float(accuracy),
            "latency_ms_mean": float(avg_latency),
            "latency_ms_std": float(std_latency),
            "throughput_samples_per_sec": float(1000 / avg_latency),
        }

        print(f"\n📊 Benchmark Results for {self.model_name}:")
        print(f"  Parameters: {param_count:,}")
        print(f"  Size: {model_size_mb:.2f} MB")
        print(f"  Accuracy: {accuracy*100:.1f}%")
        print(f"  Latency: {avg_latency:.2f}ms ± {std_latency:.2f}ms")

        return self.metrics

    def measure_latency(self, model, X_sample, num_runs=100):
        """Measure inference latency over multiple runs."""
        latencies = []
        for _ in range(num_runs):
            start = time.time()
            _ = model.forward(X_sample[:1])
            latencies.append((time.time() - start) * 1000)
        return latencies

    def measure_memory(self, model):
        """Measure model memory footprint."""
        param_count = model.count_parameters()
        return (param_count * 4) / (1024 * 1024)


def generate_submission(
    baseline_report: BenchmarkReport,
    optimized_report: Optional[BenchmarkReport] = None,
    student_name: Optional[str] = None,
    techniques_applied: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Generate a standardized benchmark submission."""
    submission = {
        "tinytorch_version": "0.1.0",
        "submission_type": "capstone_benchmark",
        "timestamp": baseline_report.timestamp,
        "system_info": baseline_report.system_info,
        "baseline": {
            "model_name": baseline_report.model_name,
            "metrics": baseline_report.metrics,
        },
    }

    # Add student name if provided
    if student_name:
        submission["student_name"] = student_name

    # Add optimization results if provided
    if optimized_report:
        submission["optimized"] = {
            "model_name": optimized_report.model_name,
            "metrics": optimized_report.metrics,
            "techniques_applied": techniques_applied or [],
        }

        # Calculate improvement metrics
        baseline_latency = baseline_report.metrics["latency_ms_mean"]
        optimized_latency = optimized_report.metrics["latency_ms_mean"]
        baseline_size = baseline_report.metrics["model_size_mb"]
        optimized_size = optimized_report.metrics["model_size_mb"]

        submission["improvements"] = {
            "speedup": float(baseline_latency / optimized_latency),
            "compression_ratio": float(baseline_size / optimized_size),
            "accuracy_delta": float(
                optimized_report.metrics["accuracy"]
                - baseline_report.metrics["accuracy"]
            ),
        }

    return submission


def save_submission(submission: Dict[str, Any], filepath: str = "submission.json"):
    """Save submission to JSON file."""
    Path(filepath).write_text(json.dumps(submission, indent=2))
    print(f"\n✅ Submission saved to: {filepath}")
    return filepath
