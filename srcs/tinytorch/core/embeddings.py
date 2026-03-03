import numpy as np
import math

from typing import List, Optional, Tuple
from tinytorch.core.tensor import Tensor
from tinytorch.core.autograd import Function, enable_autograd

enable_autograd()

# Constants for memory calculations
BYTES_PER_FLOAT32 = 4  # Standard float32 size in bytes
KB_TO_BYTES = 1024  # Kilobytes to bytes conversion
MB_TO_BYTES = 1024 * 1024  # Megabytes to bytes conversion


class EmbeddingBackward(Function):
    """Gradient computation for embedding lookup operation."""

    def __init__(self, weight, indices):
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
        (weight,) = self.saved_tensors
        grad_weight = None

        if isinstance(weight, Tensor) and weight.requires_grad:
            grad_weight = np.zeros_like(weight.data)

            indices_flat = self.indices.data.astype(int).flatten()
            grad_output_reshaped = grad_output.reshape(-1, grad_output.shape[-1])
            np.add.at(grad_weight, indices_flat, grad_output_reshaped)

        return (grad_weight,)


class Embedding:
    """Embedding layer for mapping discrete indices to continuous vectors."""

    def __init__(self, vocab_size, embed_dim):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # Xavier initialization for better gradient flow
        limit = math.sqrt(6.0 / (vocab_size + embed_dim))
        self.weight = Tensor(np.random.uniform(-limit, limit, (vocab_size, embed_dim)))

    def forward(self, indices):
        """Perform embedding lookup.

        EXAMPLE:
        >>> vocab = Embedding(3, 2)  # 3 words, 2D embeddings
        >>> indices = Tensor([0, 2, 0])  # Select words 0, 2, 0
        >>> output = vocab.forward(indices)  # [[w0], [w2], [w0]]
        """
        if np.any(indices.data >= self.vocab_size) or np.any(indices.data < 0):
            min_idx = int(np.min(indices.data))
            max_idx = int(np.max(indices.data))
            raise ValueError(
                f"Embedding index out of range for vocabulary size {self.vocab_size}\n"
                f"  ❌ Found indices: min={min_idx}, max={max_idx} (valid range: 0 to {self.vocab_size - 1})\n"
                f"  💡 Token IDs must be within the vocabulary. IDs >= vocab_size reference non-existent tokens\n"
                f"  🔧 Check your tokenizer output, or increase vocab_size to at least {max_idx + 1}"
            )

        embedd = self.weight.data[indices.data.astype(int)]
        result = Tensor(embedd)

        if self.weight.requires_grad:
            result.requires_grad = True
            result._grad_fn = EmbeddingBackward(self.weight, indices)

        return result

    def __call__(self, indices: Tensor) -> Tensor:
        """Allows the embedding to be called like a function."""
        return self.forward(indices)

    def parameters(self) -> List[Tensor]:
        """Return trainable parameters."""
        return [self.weight]

    def __repr__(self):
        return f"Embedding(vocab_size={self.vocab_size}, embed_dim={self.embed_dim})"


class PositionalEncoding:
    """Learnable positional encoding layer."""

    def __init__(self, max_seq_len, embed_dim):
        """Initialize learnable positional encoding."""
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim

        # Initialize position embedding matrix
        # Smaller initialization than token embeddings since these are additive
        limit = math.sqrt(2.0 / embed_dim)
        self.position_embeddings = Tensor(
            np.random.uniform(-limit, limit, (max_seq_len, embed_dim))
        )

    def forward(self, x):
        """Add positional encodings to input embeddings."""
        if len(x.shape) == 2:
            raise ValueError(
                f"Expected 3D input (batch, seq, embed), got 2D: {x.shape}\n"
                f"  ❌ Missing batch dimension\n"
                f"  💡 PositionalEncoding expects batched embeddings, not single sequences\n"
                f"  🔧 Add batch dim: x.reshape(1, {x.shape[0]}, {x.shape[1]})"
            )
        elif len(x.shape) != 3:
            raise ValueError(
                f"Expected 3D input (batch, seq, embed), got {len(x.shape)}D: {x.shape}\n"
                f"  ❌ Input must have exactly 3 dimensions\n"
                f"  💡 PositionalEncoding expects shape (batch_size, sequence_length, embedding_dim)"
            )

        batch_size, seq_len, embed_dim = x.shape

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length exceeds maximum: {seq_len} > {self.max_seq_len}\n"
                f"  ❌ Input sequence has {seq_len} positions, but max_seq_len is {self.max_seq_len}\n"
                f"  💡 Learned positional encodings have a fixed maximum length set at initialization\n"
                f"  🔧 Either truncate input to {self.max_seq_len} tokens, or create a new PositionalEncoding(max_seq_len={seq_len}, ...)"
            )

        if embed_dim != self.embed_dim:
            raise ValueError(
                f"Embedding dimension mismatch: input has {embed_dim}, expected {self.embed_dim}\n"
                f"  ❌ PositionalEncoding was created with embed_dim={self.embed_dim}, but input has embed_dim={embed_dim}\n"
                f"  💡 Token embeddings and positional encodings must have the same dimension to be added together\n"
                f"  🔧 Ensure your Embedding layer uses embed_dim={self.embed_dim}, or create PositionalEncoding(embed_dim={embed_dim}, ...)"
            )

        # Slice position embeddings for this sequence length using Tensor slicing
        pos_embeddings = self.position_embeddings[:seq_len]  # (seq_len, embed_dim)

        # Reshape to add batch dimension: (1, seq_len, embed_dim)
        pos_data = pos_embeddings.data[np.newaxis, :, :]
        pos_embeddings_batched = Tensor(pos_data)

        # Add positional information
        result = x + pos_embeddings_batched

        return result

    def __call__(self, x: Tensor) -> Tensor:
        """Allows the positional encoding to be called like a function."""
        return self.forward(x)

    def parameters(self) -> List[Tensor]:
        """Return trainable parameters."""
        return [self.position_embeddings]

    def __repr__(self):
        return f"PositionalEncoding(max_seq_len={self.max_seq_len}, embed_dim={self.embed_dim})"


def _compute_sinusoidal_table(max_len: int, embed_dim: int) -> np.ndarray:
    """Compute the raw sinusoidal positional encoding table as a numpy array.

    EXAMPLE:
    >>> table = _compute_sinusoidal_table(4, 8)
    >>> table.shape
    (4, 8)
    >>> table[0, 0]  # sin(0) = 0.0
    0.0
    >>> table[0, 1]  # cos(0) = 1.0
    1.0
    """
    # Create position indices [0, 1, 2, ..., max_len-1]
    position = np.arange(max_len, dtype=np.float32)[:, np.newaxis]  # (max_len, 1)

    # Create dimension indices for calculating frequencies
    div_term = np.exp(
        np.arange(0, embed_dim, 2, dtype=np.float32) * -(math.log(10000.0) / embed_dim)
    )  # (embed_dim//2,)

    # Initialize the positional encoding matrix
    pe = np.zeros((max_len, embed_dim), dtype=np.float32)

    # Apply sine to even indices (0, 2, 4, ...)
    pe[:, 0::2] = np.sin(position * div_term)

    # Apply cosine to odd indices (1, 3, 5, ...)
    if embed_dim % 2 == 1:
        # Handle odd embed_dim by only filling available positions
        pe[:, 1::2] = np.cos(position * div_term[:-1])
    else:
        pe[:, 1::2] = np.cos(position * div_term)

    return pe


def create_sinusoidal_embeddings(max_seq_len: int, embed_dim: int) -> Tensor:
    """Create sinusoidal positional encodings as used in "Attention Is All You Need".

    EXAMPLE:
    >>> pe = create_sinusoidal_embeddings(512, 64)
    >>> print(pe.shape)
    (512, 64)
    >>> # Position 0: [0, 1, 0, 1, 0, 1, ...] (sin(0)=0, cos(0)=1)
    >>> # Each position gets unique trigonometric signature
    """
    pe = _compute_sinusoidal_table(max_seq_len, embed_dim)
    return Tensor(pe)


class EmbeddingLayer:
    """Complete embedding system combining token and positional embeddings."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        max_seq_len: int = 512,
        pos_encoding: str = "learned",
        scale_embeddings: bool = False,
    ):
        """Initialize complete embedding system.

        EXAMPLE:
        >>> layer = EmbeddingLayer(vocab_size=100, embed_dim=64, pos_encoding='learned')
        >>> layer.token_embedding  # Embedding(vocab_size=100, embed_dim=64)
        >>> layer.pos_encoding     # PositionalEncoding(max_seq_len=512, embed_dim=64)
        """
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.pos_encoding_type = pos_encoding
        self.scale_embeddings = scale_embeddings

        # Token embedding layer
        self.token_embedding = Embedding(vocab_size, embed_dim)

        # Positional encoding
        if pos_encoding == "learned":
            self.pos_encoding = PositionalEncoding(max_seq_len, embed_dim)
        elif pos_encoding == "sinusoidal":
            # Create fixed sinusoidal encodings (no parameters)
            self.pos_encoding = create_sinusoidal_embeddings(max_seq_len, embed_dim)
        elif pos_encoding is None:
            self.pos_encoding = None
        else:
            raise ValueError(
                f"Unknown positional encoding type: '{pos_encoding}'\n"
                f"  ❌ pos_encoding must be 'learned', 'sinusoidal', or None\n"
                f"  💡 'learned' = trainable position embeddings (task-specific but fixed max length)\n"
                f"     'sinusoidal' = mathematical sin/cos patterns (no parameters, can extrapolate)\n"
                f"     None = no positional encoding (order-agnostic model)\n"
                f"  🔧 Use: EmbeddingLayer(..., pos_encoding='learned') or pos_encoding='sinusoidal'"
            )

    def __call__(self, tokens):
        """Allows the embedding layer to be called like a function."""
        return self.forward(tokens)

    def parameters(self):
        """Return all trainable parameters."""
        params = self.token_embedding.parameters()
        if self.pos_encoding_type == "learned":
            params.extend(self.pos_encoding.parameters())
        return params

    def __repr__(self):
        return (
            f"EmbeddingLayer(vocab_size={self.vocab_size}, "
            f"embed_dim={self.embed_dim}, "
            f"pos_encoding='{self.pos_encoding_type}')"
        )

    def forward(self, tokens):
        """Forward pass through complete embedding system.

        EXAMPLE:
        >>> layer = EmbeddingLayer(vocab_size=100, embed_dim=64)
        >>> tokens = Tensor([[1, 2, 3], [4, 5, 6]])
        >>> output = layer.forward(tokens)
        >>> output.shape
        (2, 3, 64)

        """
        # Handle 1D input by adding batch dimension
        if len(tokens.shape) == 1:
            # NOTE: Tensor reshape preserves gradients
            tokens = tokens.reshape(1, -1)
            squeeze_batch = True
        else:
            squeeze_batch = False

        # Get token embeddings
        token_embeds = self.token_embedding.forward(tokens)  # (batch, seq, embed)

        # Scale embeddings if requested (transformer convention)
        if self.scale_embeddings:
            scale_factor = math.sqrt(self.embed_dim)
            token_embeds = (
                token_embeds * scale_factor
            )  # Use Tensor multiplication to preserve gradients

        # Add positional encoding
        if self.pos_encoding_type == "learned":
            # Use learnable positional encoding
            output = self.pos_encoding.forward(token_embeds)
        elif self.pos_encoding_type == "sinusoidal":
            # Use fixed sinusoidal encoding (not learnable)
            batch_size, seq_len, embed_dim = token_embeds.shape
            pos_embeddings = self.pos_encoding[:seq_len]  # Slice using Tensor slicing

            # Reshape to add batch dimension
            pos_data = pos_embeddings.data[np.newaxis, :, :]
            pos_embeddings_batched = Tensor(pos_data)  # Sinusoidal are fixed

            output = token_embeds + pos_embeddings_batched
        else:
            # No positional encoding
            output = token_embeds

        # Remove batch dimension if it was added
        if squeeze_batch:
            # Use Tensor slicing (now supported in Module 01)
            output = output[0]

        return output
