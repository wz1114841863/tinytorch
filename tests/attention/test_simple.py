import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tinytorch.core.tensor import Tensor
from tinytorch.core.attention import MultiHeadAttention, scaled_dot_product_attention
from tinytorch.core.autograd import enable_autograd
from tinytorch.core.layers import Linear

enable_autograd()


# 专门测试这个维度的组合
def test_linear_3d():
    layer = Linear(16, 16)
    x = Tensor(np.random.randn(1, 4, 16), requires_grad=True)
    out = layer.forward(x)
    loss = out.sum()
    loss.backward()
    print("Linear 3D backward successful!")


def test_matmul_4d_flow():
    # 模拟 MHA 的维度: (Batch=1, Heads=2, Seq=4, Head_Dim=8)
    # Weights: (1, 2, 4, 4)
    # V: (1, 2, 4, 8)
    weights_data = np.random.randn(1, 2, 4, 4).astype(np.float32)
    v_data = np.random.randn(1, 2, 4, 8).astype(np.float32)

    weights = Tensor(weights_data, requires_grad=True)
    v = Tensor(v_data, requires_grad=True)

    # Forward: (1, 2, 4, 4) @ (1, 2, 4, 8) -> (1, 2, 4, 8)
    output = weights.matmul(v)

    print(f"Forward Output Shape: {output.shape}")
    assert output.shape == (1, 2, 4, 8), f"Forward shape mismatch: {output.shape}"

    # Backward: 模拟 loss.sum()
    loss = output.sum()
    loss.backward()

    # 检查梯度维度
    if v.grad is not None:
        print(f"V grad shape: {v.grad.shape}")  # 预期 (1, 2, 4, 8)
    if weights.grad is not None:
        print(f"Weights grad shape: {weights.grad.shape}")  # 预期 (1, 2, 4, 4)

    # 核心检查点:梯度形状必须与原始形状完全一致
    assert (
        v.grad.shape == v.shape
    ), f"V grad shape error! Expected {v.shape}, got {v.grad.shape}"
    assert (
        weights.grad.shape == weights.shape
    ), f"Weights grad shape error! Expected {weights.shape}, got {weights.grad.shape}"

    print("✅ 4D Matmul Test Passed!")


def test_merge_heads_backward():
    batch, heads, seq, h_dim = 1, 2, 4, 8
    embed_dim = heads * h_dim  # 16

    # 模拟 Attention 后的输出
    x = Tensor(np.random.randn(batch, heads, seq, h_dim), requires_grad=True)

    # 模拟 _merge_heads 内部逻辑
    # 1. Transpose
    x_trans = x.transpose(1, 2)  # (1, 4, 2, 8)
    # 2. Reshape
    x_merged = x_trans.reshape(batch, seq, embed_dim)  # (1, 4, 16)

    print(f"Merged Shape: {x_merged.shape}")

    # Backward
    loss = x_merged.sum()
    loss.backward()

    if x.grad is not None:
        print(f"X grad shape: {x.grad.shape}")
        assert x.grad.shape == (1, 2, 4, 8), f"Grad shape mismatch: {x.grad.shape}"
    else:
        print("X grad is None!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
