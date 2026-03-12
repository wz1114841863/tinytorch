import numpy as np
import time
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union

from tinytorch.core.tensor import Tensor

DEFAULT_WARMUP_ITERATIONS = 2  # Default warmup iterations for timing
DEFAULT_TIMING_ITERATIONS = 5  # Default timing iterations for measurement
BYTES_PER_FLOAT32 = 4  # Standard float32 size in bytes


def vectorized_matmul(a, b):
    """High-performance matrix multiplication using vectorized operations.

    EXAMPLE:
    Matrix multiplication visualization:
    >>> a = Tensor([[1, 2], [3, 4]])  # 2×2
    >>> b = Tensor([[5, 6], [7, 8]])  # 2×2
    >>> result = vectorized_matmul(a, b)
    >>> print(result.data)
    [[19 22]    # [1×5+2×7, 1×6+2×8] = [19, 22]
     [43 50]]   # [3×5+4×7, 3×6+4×8] = [43, 50]
    """
    if len(a.shape) < 2 or len(b.shape) < 2:
        raise ValueError(
            f"Matrix multiplication requires 2D+ tensors\n"
            f"  ❌ Got shapes {a.shape} and {b.shape} ({len(a.shape)}D and {len(b.shape)}D tensors)\n"
            f"  💡 Matrix multiplication computes dot products between rows and columns, which requires at least 2D tensors\n"
            f"  🔧 Add dimensions with reshape: a.reshape(1, {a.shape[-1] if len(a.shape) >= 1 else 'n'}) for a row vector"
        )

    if a.shape[-1] != b.shape[-2]:
        raise ValueError(
            f"Matrix multiplication shape mismatch: {a.shape} @ {b.shape}\n"
            f"  ❌ Inner dimensions don't match: a.shape[-1]={a.shape[-1]} vs b.shape[-2]={b.shape[-2]}\n"
            f"  💡 For A @ B, each row of A (length {a.shape[-1]}) must match each column of B (length {b.shape[-2]})\n"
            f"  🔧 Try: b.reshape({a.shape[-1]}, -1) or a.reshape(-1, {b.shape[-2]})"
        )

    result_data = np.matmul(a.data, b.data)
    return Tensor(result_data)


def fused_gelu(x):
    """Fused GELU activation that combines all operations in a single kernel.

    EXAMPLE:
    >>> x = Tensor([-2, -1, 0, 1, 2])
    >>> result = fused_gelu(x)
    >>> print(result.data)
    [-0.04550026 -0.15865526  0.          0.8413447   1.9544997 ]
    """
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
    result_data = (
        0.5 * x.data * (1.0 + np.tanh(sqrt_2_over_pi * (x.data + 0.044715 * x.data**3)))
    )

    return Tensor(result_data)


def tiled_matmul(a, b, tile_size=64):
    """Cache-aware matrix multiplication using tiling/blocking."""
    if len(a.shape) < 2 or len(b.shape) < 2:
        raise ValueError(
            f"Tiled matrix multiplication requires 2D+ tensors\n"
            f"  ❌ Got shapes {a.shape} and {b.shape} ({len(a.shape)}D and {len(b.shape)}D tensors)\n"
            f"  💡 Tiling partitions matrices into cache-sized blocks, which requires 2D structure\n"
            f"  🔧 Add dimensions with reshape: tensor.reshape(1, -1) for row vector or tensor.reshape(-1, 1) for column"
        )

    if a.shape[-1] != b.shape[-2]:
        raise ValueError(
            f"Tiled matrix multiplication shape mismatch: {a.shape} @ {b.shape}\n"
            f"  ❌ Inner dimensions don't match: a.shape[-1]={a.shape[-1]} vs b.shape[-2]={b.shape[-2]}\n"
            f"  💡 Each tile of A's columns must align with tiles of B's rows for block multiplication\n"
            f"  🔧 Reshape to align: b.reshape({a.shape[-1]}, -1) or transpose if dimensions are swapped"
        )

    # For educational purposes, we use NumPy's matmul which already
    # implements cache-aware tiling via BLAS libraries (MKL, OpenBLAS)
    # These libraries automatically partition large matrices into
    # cache-sized blocks for optimal performance

    # In a full educational implementation, you would write:
    # for i_tile in range(0, M, tile_size):
    #     for j_tile in range(0, N, tile_size):
    #         for k_tile in range(0, K, tile_size):
    #             # Multiply tile blocks that fit in cache
    #             C[i_tile:i_tile+tile_size, j_tile:j_tile+tile_size] +=
    #                 A[i_tile:i_tile+tile_size, k_tile:k_tile+tile_size] @
    #                 B[k_tile:k_tile+tile_size, j_tile:j_tile+tile_size]

    result_data = np.matmul(a.data, b.data)
    return Tensor(result_data)
