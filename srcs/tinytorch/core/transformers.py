import numpy as np

from tinytorch.core.activations import GELU
from tinytorch.core.attention import MultiHeadAttention
from tinytorch.core.autograd import Function
from tinytorch.core.embeddings import EmbeddingLayer
from tinytorch.core.layers import Linear
from tinytorch.core.tensor import Tensor

BYTES_PER_FLOAT32 = 4
MB_TO_BYTES = 1024 * 1024


def create_causal_mask(seq_len):
    """Create a causal mask for self-attention.

    EXAMPLE:
    >>> mask = create_causal_mask(4)
    >>> print(mask.data)
    [[1. 0. 0. 0.]
     [1. 1. 0. 0.]
     [1. 1. 1. 0.]
     [1. 1. 1. 1.]]
    """
    mask = np.tril(np.ones((seq_len, seq_len), dtype=np.float32))
    return Tensor(mask[np.newaxis, :, :])  # Add batch dimension


class _LayerNormBackward(Function):
    """Gradient computation for the full layer normalization operation.

    output = gamma * ((x - mean) / std) + beta
    """

    def __init__(self, x, gamma, beta, normalized_data, std_data):
        super().__init__(x, gamma, beta)
        self.normalized_data = normalized_data
        self.std_data = std_data

    def apply(self, grad_output):
        """Compute gradients for layer normalization."""
        x, gamma, beta = self.saved_tensors

        grad_x = grad_gamma = grad_beta = None
        normalized = self.normalized_data
        std_data = self.std_data

        # Gradient for beta: sum over all dims except last
        if isinstance(beta, Tensor) and beta.requires_grad:
            grad_beta = grad_output.copy()
            while grad_beta.ndim > 1:
                grad_beta = grad_beta.sum(axis=0)

        # Gradient for gamma: sum of (grad_output * normalized) over all dims except last
        if isinstance(gamma, Tensor) and gamma.requires_grad:
            grad_gamma = (grad_output * normalized).copy()
            while grad_gamma.ndim > 1:
                grad_gamma = grad_gamma.sum(axis=0)

        # Gradient for x
        if isinstance(x, Tensor) and x.requires_grad:
            gamma_data = gamma.data if isinstance(gamma, Tensor) else gamma
            grad_norm = grad_output * gamma_data

            mean_grad = np.mean(grad_norm, axis=-1, keepdims=True)
            mean_grad_norm = np.mean(grad_norm * normalized, axis=-1, keepdims=True)
            grad_x = (1.0 / std_data) * (
                grad_norm - mean_grad - normalized * mean_grad_norm
            )

        return (grad_x, grad_gamma, grad_beta)


class LayerNorm:
    """Layer normalization module."""

    def __init__(self, normalized_shape, eps=1e-5):
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.gamma = Tensor(np.ones(normalized_shape))  # Scale parameter
        self.beta = Tensor(np.zeros(normalized_shape))  # Shift parameter

    def forward(self, x):
        mean_data = np.mean(x.data, axis=-1, keepdims=True)
        diff = x.data - mean_data
        variance = np.mean(diff**2, axis=-1, keepdims=True)
        std_data = np.sqrt(variance + self.eps)
        normalized_data = diff / std_data
        output_data = self.gamma.data * normalized_data + self.beta.data
        output = Tensor(output_data, requires_grad=x.requires_grad)

        if x.requires_grad or self.gamma.requires_grad or self.beta.requires_grad:
            output.requires_grad = True
            output._grad_fn = _LayerNormBackward(
                x, self.gamma, self.beta, normalized_data, std_data
            )

        return output

    def __call__(self, x):
        """Allows the layer norm to be called like a function."""
        return self.forward(x)

    def parameters(self):
        """Return learnable parameters."""
        return [self.gamma, self.beta]


class MLP:
    """Multi-layer perceptron used in transformer feed-forward networks."""

    def __init__(self, embed_dim, hidden_dim=None, droupout_porb=0.1):
        if hidden_dim is None:
            hidden_dim = embed_dim * 4

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.linear1 = Linear(embed_dim, hidden_dim)
        self.gelu = GELU()
        self.linear2 = Linear(hidden_dim, embed_dim)

    def forward(self, x):
        hidden = self.linear1.forward(x)
        hidden = self.gelu.forward(hidden)
        output = self.linear2.forward(hidden)
        return output

    def __call__(self, x):
        """Allows the MLP to be called like a function."""
        return self.forward(x)

    def parameters(self):
        """Return all learnable parameters."""
        params = []
        params.extend(self.linear1.parameters())
        params.extend(self.linear2.parameters())
        return params


# | export
class TransformerBlock:
    """Complete Transformer Block with self-attention, MLP, and residual connections."""

    def __init__(
        self, embed_dim, num_heads, mlp_ratio=4, ff_dim=None, dropout_prob=0.1
    ):
        """Initialize a complete transformer block.

        EXAMPLE:
        >>> block = TransformerBlock(embed_dim=512, num_heads=8)
        >>> x = Tensor(np.random.randn(2, 10, 512))  # (batch, seq, embed)
        >>> output = block.forward(x)
        >>> assert output.shape == (2, 10, 512)
        """

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.ln1 = LayerNorm(embed_dim)  # Before attention
        self.ln2 = LayerNorm(embed_dim)  # Before MLP

        if ff_dim is not None:
            hidden_dim = ff_dim
        else:
            hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(embed_dim, hidden_dim)

    def forward(self, x, mask=None):
        """Forward pass through transformer block.

        COMPUTATION FLOW:
        x → ln1 → attention → + x → ln2 → mlp → + → output
        """
        # First sub-layer: Multi-head self-attention with residual connection
        # Pre-norm: LayerNorm before attention
        normed1 = self.ln1.forward(x)
        attention_out = self.attention.forward(normed1, mask)
        x = x + attention_out

        normed2 = self.ln2.forward(x)
        mlp_out = self.mlp.forward(normed2)
        output = x + mlp_out

        return output

    def __call__(self, x, mask=None):
        """Allows the transformer block to be called like a function."""
        return self.forward(x, mask)

    def parameters(self):
        """Return all learnable parameters."""
        params = []
        params.extend(self.attention.parameters())
        params.extend(self.ln1.parameters())
        params.extend(self.ln2.parameters())
        params.extend(self.mlp.parameters())
        return params


class GPT:

    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, max_seq_len=1024):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        self.embedding_layer = EmbeddingLayer(vocab_size, embed_dim, max_seq_len)
        self.blocks = []
        for _ in range(num_layers):
            block = TransformerBlock(embed_dim, num_heads)
            self.blocks.append(block)

        self.ln_f = LayerNorm(embed_dim)  # Final layer norm
        self.lm_head = Linear(
            embed_dim, vocab_size, bias=False
        )  # Language modeling head

    def forward(self, tokens):
        batch_size, seq_len = tokens.shape
        x = self.embedding_layer.forward(tokens)
        mask = self._create_causal_mask(seq_len)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block.forward(x, mask)

        # Final layer normalization
        x = self.ln_f.forward(x)

        # Language modeling head
        logits = self.lm_head.forward(x)

        return logits

    def __call__(self, tokens):
        """Allows the GPT model to be called like a function."""
        return self.forward(tokens)

    def _create_causal_mask(self, seq_len):
        """Create causal mask to prevent attending to future positions."""
        mask = np.triu(np.ones((seq_len, seq_len)) * -np.inf, k=1)
        return Tensor(mask)

    def parameters(self):
        """Return all learnable parameters."""
        params = []
        params.extend(self.embedding_layer.parameters())

        for block in self.blocks:
            params.extend(block.parameters())

        params.extend(self.ln_f.parameters())
        params.extend(self.lm_head.parameters())

        return params

    def _sample_next_token(self, logits, temperature=1.0):
        scaled_logits = logits / temperature
        exp_logits = np.exp(
            scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True)
        )
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        next_token = np.random.choice(self.vocab_size, p=probs[0])
        return next_token

    def generate(self, prompt_tokens, max_new_tokens=50, temperature=1.0):
        """Generate text given a prompt."""
        current_tokens = Tensor(prompt_tokens.data.copy())

        for _ in range(max_new_tokens):
            logits = self.forward(current_tokens)
            last_logits = logits.data[:, -1, :]
            next_token_id = self._sample_next_token(last_logits, temperature)
            next_token = np.array([[next_token_id]])
            current_tokens = Tensor(
                np.concatenate([current_tokens.data, next_token], axis=1)
            )
        return current_tokens


def demonstrate_transformer_integration():
    """Demonstrate complete transformer pipeline.

    This simulates training a small language model on a simple vocabulary.
    """
    print("🔗 Integration Demo: Complete Language Model Pipeline")
    print("Building a mini-GPT for character-level text generation")

    # Create a small vocabulary (character-level)
    vocab = list("abcdefghijklmnopqrstuvwxyz .")
    vocab_size = len(vocab)
    char_to_idx = {char: i for i, char in enumerate(vocab)}
    idx_to_char = {i: char for i, char in enumerate(vocab)}

    print(f"Vocabulary size: {vocab_size}")
    print(f"Characters: {''.join(vocab)}")

    # Create model
    model = GPT(
        vocab_size=vocab_size, embed_dim=64, num_layers=2, num_heads=4, max_seq_len=32
    )

    # Sample text encoding
    text = "hello world."
    tokens = [char_to_idx[char] for char in text]
    input_tokens = Tensor(np.array([tokens]))

    print(f"\nOriginal text: '{text}'")
    print(f"Tokenized: {tokens}")
    print(f"Input shape: {input_tokens.shape}")

    # Forward pass
    logits = model.forward(input_tokens)
    print(f"Output logits shape: {logits.shape}")
    print(f"Each position predicts next token from {vocab_size} possibilities")

    # Generation demo
    prompt_text = "hello"
    prompt_tokens = [char_to_idx[char] for char in prompt_text]
    prompt = Tensor(np.array([prompt_tokens]))

    print("\nGeneration demo:")
    print(f"Prompt: '{prompt_text}'")

    generated = model.generate(prompt, max_new_tokens=8, temperature=1.0)
    generated_text = "".join([idx_to_char[idx] for idx in generated.data[0]])

    print(f"Generated: '{generated_text}'")
    print("(Note: Untrained model produces random text)")

    return model


TinyGPT = GPT
