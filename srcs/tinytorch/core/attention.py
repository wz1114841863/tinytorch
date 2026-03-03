import numpy as np
import math
import time
from typing import Optional, Tuple, List
from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear
from tinytorch.core.activations import Softmax

MASK_VALUE = -1e9


def _compute_attention_scores(Q, K):
    """Compute raw attention scores via Q @ K^T.

    EXAMPLE:
    >>> Q = Tensor(np.random.randn(1, 3, 4))  # 3 tokens, dim=4
    >>> K = Tensor(np.random.randn(1, 3, 4))
    >>> scores = _compute_attention_scores(Q, K)
    >>> print(scores.shape)  # (1, 3, 3) -- every token scored against every other
    """
    K_t = K.transpose(-2, -1)
    return Q.matmul(K_t)


def _scale_scores(scores, d_model):
    """Scale attention scores by sqrt(d_model).

    EXAMPLE:
    >>> scores = Tensor(np.array([[[4.0, 8.0]]]))
    >>> scaled = _scale_scores(scores, d_model=4)
    >>> print(scaled.data)  # [[[ 2.0, 4.0]]] -- divided by sqrt(4)=2
    """
    scale_factor = 1.0 / math.sqrt(d_model)
    return scores * scale_factor


def _apply_mask(scores, mask):
    """Apply mask to attention scores, setting masked positions to MASK_VALUE.

    EXAMPLE:
    >>> scores = Tensor(np.array([[[1.0, 2.0], [3.0, 4.0]]]))
    >>> mask = Tensor(np.array([[1, 0]]))  # Mask out second token
    >>> masked_scores = _apply_mask(scores, mask)
    >>> print(masked_scores.data)  # [[[1.0, -1e9], [3.0, -1e9]]]
    """
    adder = (1.0 - mask.data) * MASK_VALUE
    return scores + Tensor(adder)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Compute scaled dot-product attention.

    EXAMPLE:
    >>> Q = Tensor(np.random.randn(2, 4, 64))
    >>> K = Tensor(np.random.randn(2, 4, 64))
    >>> V = Tensor(np.random.randn(2, 4, 64))
    >>> output, weights = scaled_dot_product_attention(Q, K, V)
    >>> print(output.shape)   # (2, 4, 64)
    >>> print(weights.shape)  # (2, 4, 4)
    """
    scores = _compute_attention_scores(Q, K)
    scores = _scale_scores(scores, Q.shape[-1])
    if mask is not None:
        scores = _apply_mask(scores, mask)
    softmax = Softmax()
    attention_weights = softmax(scores, dim=-1)
    output = attention_weights.matmul(V)
    return output, attention_weights


class MultiHeadAttention:
    """Multi-head attention mechanism."""

    def __init__(self, embed_dim, num_heads):
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"Multi-head attention dimension mismatch\n"
                f"  ❌ embed_dim={embed_dim} is not divisible by num_heads={num_heads} (remainder={embed_dim % num_heads})\n"
                f"  💡 Multi-head attention splits embed_dim equally among heads, so embed_dim must be a multiple of num_heads\n"
                f"  🔧 Try: embed_dim={num_heads * (embed_dim // num_heads + 1)} (next valid size) or num_heads={embed_dim // (embed_dim // num_heads)} (fewer heads)"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)

        self.out_proj = Linear(embed_dim, embed_dim)

    def _split_heads(self, x, batch_size, seq_len):
        """Reshape to separate attention heads for parallel processing.

        EXAMPLE:
        >>> mha = MultiHeadAttention(embed_dim=64, num_heads=8)
        >>> x = Tensor(np.random.randn(2, 10, 64))  # batch=2, seq=10
        >>> split = mha._split_heads(x, 2, 10)
        >>> print(split.shape)  # (2, 8, 10, 8) -- 8 heads of dim 8
        """
        x = x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x, batch_size, seq_len):
        """Reshape back to original dimensions after processing heads.

        EXAMPLE:
        >>> mha = MultiHeadAttention(embed_dim=64, num_heads=8)
        >>> x = Tensor(np.random.randn(2, 8, 10, 8))  # batch=2, heads=8, seq=10
        >>> merged = mha._merge_heads(x, 2, 10)
        >>> print(merged.shape)  # (2, 10, 64) -- merged back to original
        """
        x = x.transpose(1, 2)
        return x.reshape(batch_size, seq_len, self.embed_dim)

    def forward(self, x, mask=None):
        """Forward pass through multi-head attention.

        EXAMPLE:
        >>> mha = MultiHeadAttention(embed_dim=64, num_heads=8)
        >>> x = Tensor(np.random.randn(2, 10, 64))  # batch=2, seq=10, dim=64
        >>> output = mha.forward(x)
        >>> print(output.shape)  # (2, 10, 64) - same as input
        """
        batch_size, seq_len, embed_dim = x.shape
        if embed_dim != self.embed_dim:
            raise ValueError(
                f"MultiHeadAttention input dimension mismatch\n"
                f"  ❌ Expected embed_dim={self.embed_dim}, got {embed_dim} from input shape {x.shape}\n"
                f"  💡 The last dimension of input must match embed_dim from initialization (MultiHeadAttention({self.embed_dim}, {self.num_heads}))\n"
                f"  🔧 Try: x.reshape({x.shape[0]}, {x.shape[1]}, {self.embed_dim}) or create new MultiHeadAttention({embed_dim}, num_heads)"
            )

        Q = self.q_proj.forward(x)
        K = self.k_proj.forward(x)
        V = self.v_proj.forward(x)

        Q = self._split_heads(Q, batch_size, seq_len)
        K = self._split_heads(K, batch_size, seq_len)
        V = self._split_heads(V, batch_size, seq_len)

        mask_reshaped = mask
        if mask is not None and len(mask.shape) == 3:
            batch_size_mask, seq_len_mask, _ = mask.shape
            mask_data = mask.data.reshape(
                batch_size_mask, 1, seq_len_mask, seq_len_mask
            )
            mask_reshaped = Tensor(mask_data)

        attended, _ = scaled_dot_product_attention(Q, K, V, mask=mask_reshaped)
        concat_output = self._merge_heads(attended, batch_size, seq_len)
        output = self.out_proj.forward(concat_output)
        return output

    def __call__(self, x, mask=None):
        return self.forward(x, mask=mask)

    def parameters(self):
        params = []
        params.extend(self.q_proj.parameters())
        params.extend(self.k_proj.parameters())
        params.extend(self.v_proj.parameters())
        params.extend(self.out_proj.parameters())
        return params
