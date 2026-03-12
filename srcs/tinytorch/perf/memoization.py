import numpy as np
import time
from typing import Tuple, Optional, Dict, List

from tinytorch.core.tensor import Tensor
from tinytorch.perf.profiling import Profiler

_BYTES_PER_FLOAT32 = 4  # Standard float32 size in bytes
_MB_TO_BYTES = 1024 * 1024  # Megabytes to bytes conversion


def profile_naive_generation():
    """Profile transformer generation to discover the O(n²) bottleneck."""
    profiler = Profiler()

    def naive_attention_step(seq_len, hidden_dim=64):
        q = Tensor(np.random.rand(1, seq_len, hidden_dim))
        k = Tensor(np.random.rand(1, seq_len, hidden_dim))
        v = Tensor(np.random.rand(1, seq_len, hidden_dim))

        scores = q @ k.T
        output = scores @ v

        return output

    # Profile at increasing sequence lengths
    print("🔬 Profiling Transformer Generation (Without Caching):\n")
    print("   Seq Len  |  Latency (ms)  |  Growth")
    print("   ---------|----------------|----------")

    sequence_lengths = [10, 20, 40, 80, 160]
    latencies = []

    for seq_len in sequence_lengths:
        # Measure latency for this sequence length
        latency = profiler.measure_latency(
            lambda: naive_attention_step(seq_len), None, warmup=5, iterations=20
        )
        latencies.append(latency)

        # Calculate growth rate
        if len(latencies) > 1:
            growth = latencies[-1] / latencies[-2]
            print(f"   {seq_len:3d}      |  {latency:6.2f}        |  {growth:.2f}×")
        else:
            print(f"   {seq_len:3d}      |  {latency:6.2f}        |  baseline")

    print("\n💡 Key Observations:")
    print("   • Latency grows QUADRATICALLY with sequence length")
    print("   • Each new token forces recomputation of ALL previous K,V pairs")
    print("   • For 160 tokens: ~4× time vs 80 tokens (2² growth)")

    print("\n🎯 The Problem:")
    print("   K and V values for previous tokens NEVER change,")
    print("   yet we recompute them every single step!")

    print("\n✨ The Solution:")
    print("   CACHE the K,V values! (That's memoization)")
    print("   • First compute: Calculate and store K,V")
    print("   • Later steps: Reuse stored K,V")
    print("   • Complexity: O(n²) → O(n)")
    print("   • Speedup: 10-15× for typical generation\n")


class KVCache:
    """Cache for storing K and V values during transformer generation."""

    def __init__(self, batch_size, max_seq_len, num_layers, num_heads, head_dim):
        """initialize KV cache for efficient generation.

        EXAMPLE:
        >>> cache = KVCache(batch_size=2, max_seq_len=128, num_layers=4,
        ...                 num_heads=8, head_dim=64)
        >>> cache.seq_pos  # 0 (no tokens cached yet)
        >>> len(cache.caches)  # 4 (one per layer)
        >>> cache.caches[0][0].shape  # (2, 8, 128, 64) - key cache for layer 0
        """
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Current sequence position (how many tokens are cached)
        self.seq_pos = 0

        # Initialize the cache for each layer
        self.caches = []

        for layer_idx in range(num_layers):
            # Pre-allocate cache tensors with maximum size
            # Shape: (batch_size, num_heads, max_seq_len, head_dim)
            key_cache = Tensor(np.zeros((batch_size, num_heads, max_seq_len, head_dim)))
            value_cache = Tensor(
                np.zeros((batch_size, num_heads, max_seq_len, head_dim))
            )

            self.caches.append([key_cache, value_cache])

    def update(self, layer_idx, key, value):
        """Update cache with new key-value pairs for given layer.

        >>> cache = KVCache(batch_size=1, max_seq_len=10, num_layers=2,
        ...                 num_heads=4, head_dim=64)
        >>> new_k = Tensor(np.random.randn(1, 4, 1, 64))
        >>> new_v = Tensor(np.random.randn(1, 4, 1, 64))
        >>> cache.update(layer_idx=0, key=new_k, value=new_v)
        >>> cache.seq_pos  # Still 0 (update doesn't advance position)
        >>> cache.advance()
        >>> cache.seq_pos  # Now 1
        """
        if layer_idx >= self.num_layers:
            raise ValueError(
                f"Invalid layer index for cache update\n"
                f"  ❌ layer_idx={layer_idx} is out of range [0, {self.num_layers - 1}]\n"
                f"  💡 KVCache was initialized with num_layers={self.num_layers}, so valid indices are 0 to {self.num_layers - 1}\n"
                f"  🔧 Check your transformer block loop: for layer_idx in range({self.num_layers})"
            )

        if self.seq_pos >= self.max_seq_len:
            raise ValueError(
                f"KV cache is full - cannot add more tokens\n"
                f"  ❌ Current position {self.seq_pos} has reached max_seq_len={self.max_seq_len}\n"
                f"  💡 The cache was pre-allocated for {self.max_seq_len} tokens maximum. Autoregressive generation cannot exceed this limit.\n"
                f"  🔧 Either: (1) call cache.reset() to start a new sequence, or (2) create a larger cache with max_seq_len > {self.max_seq_len}"
            )

        key_cache, value_cache = self.caches[layer_idx]
        key_cache.data[:, :, self.seq_pos : self.seq_pos + 1, :] = key.data
        value_cache.data[:, :, self.seq_pos : self.seq_pos + 1, :] = value.data

    def get(self, layer_idx):
        """Retrieve cached key-value pairs for attention computation.


        EXAMPLE:
        >>> cache = KVCache(batch_size=1, max_seq_len=100, num_layers=2,
        ...                 num_heads=4, head_dim=64)
        >>> # After processing 3 tokens
        >>> cache.seq_pos = 3
        >>> cached_k, cached_v = cache.get(layer_idx=0)
        >>> cached_k.shape  # (1, 4, 3, 64) - only first 3 positions
        >>> cached_v.shape  # (1, 4, 3, 64)
        """
        if layer_idx >= self.num_layers:
            raise ValueError(
                f"Invalid layer index for cache retrieval\n"
                f"  ❌ layer_idx={layer_idx} is out of range [0, {self.num_layers - 1}]\n"
                f"  💡 KVCache was initialized with num_layers={self.num_layers}, so valid indices are 0 to {self.num_layers - 1}\n"
                f"  🔧 Check your transformer block loop: for layer_idx in range({self.num_layers})"
            )
        key_cache, value_cache = self.caches[layer_idx]
        valid_len = self.seq_pos
        cached_keys = Tensor(key_cache.data[:, :, :valid_len, :])
        cached_values = Tensor(value_cache.data[:, :, :valid_len, :])

        return cached_keys, cached_values

    def advance(self) -> None:
        """Advance sequence position after processing current token."""
        self.seq_pos += 1

    def reset(self) -> None:
        """Reset cache for new generation sequence."""
        self.seq_pos = 0

        for layer_idx in range(self.num_layers):
            key_cache, value_cache = self.caches[layer_idx]
            key_cache.data.fill(0.0)
            value_cache.data.fill(0.0)

    def get_memory_usage(self) -> Dict[str, float]:
        """Calculate memory usage of the cache system."""
        cache_size = self.batch_size * self.num_heads * self.max_seq_len * self.head_dim

        # Each layer has key_cache + value_cache
        total_cache_tensors = self.num_layers * 2
        total_elements = cache_size * total_cache_tensors
        total_bytes = total_elements * _BYTES_PER_FLOAT32
        total_mb = total_bytes / _MB_TO_BYTES

        return {
            "total_mb": total_mb,
            "per_layer_mb": total_mb / self.num_layers,
            "cache_tensors": total_cache_tensors,
            "total_elements": total_elements,
        }


def _cached_generation_step(x, attention, cache_obj, layer_idx):
    """Execute a single cached generation step for one new token."""
    batch_size = x.shape[0]
    num_heads = attention.num_heads
    head_dim = attention.head_dim

    Q_new = attention.q_proj.forward(x)  # (batch, 1, embed_dim)
    K_new = attention.k_proj.forward(x)
    V_new = attention.v_proj.forward(x)

    Q_heads = Tensor(
        np.transpose(
            Q_new.data.reshape(batch_size, -1, num_heads, head_dim), (0, 2, 1, 3)
        )
    )
    K_heads = Tensor(
        np.transpose(
            K_new.data.reshape(batch_size, -1, num_heads, head_dim), (0, 2, 1, 3)
        )
    )
    V_heads = Tensor(
        np.transpose(
            V_new.data.reshape(batch_size, -1, num_heads, head_dim), (0, 2, 1, 3)
        )
    )

    cache_obj.update(layer_idx, K_heads, V_heads)
    K_all, V_all = cache_obj.get(layer_idx)
    K_transposed = np.transpose(K_all.data, (0, 1, 3, 2))
    scores = np.matmul(Q_heads.data, K_transposed) / np.sqrt(head_dim)
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    attention_output = np.matmul(attention_weights, V_all.data)
    attention_output_transposed = np.transpose(attention_output, (0, 2, 1, 3))
    concat_output = Tensor(
        attention_output_transposed.reshape(batch_size, 1, num_heads * head_dim)
    )

    return attention.out_proj.forward(concat_output)


def _create_cache_storage(model):
    """Validate model architecture and create a KVCache sized for it.

    EXAMPLE:
    >>> model = MockGPT()  # embed_dim=128, num_heads=4, etc.
    >>> cache, head_dim = _create_cache_storage(model)
    >>> cache.num_layers  # 4
    >>> head_dim  # 32
    >>> model._cache_enabled  # True
    """
    required_attrs = ["embed_dim", "num_layers", "num_heads", "max_seq_len", "blocks"]
    for attr in required_attrs:
        if not hasattr(model, attr):
            raise AttributeError(
                f"Model missing required attribute for KV caching\n"
                f"  ❌ Model does not have '{attr}' attribute\n"
                f"  💡 enable_kv_cache() requires a GPT-style transformer with architecture attributes: {', '.join(required_attrs)}\n"
                f"  🔧 Ensure your model class defines: self.{attr} = <value> in __init__()"
            )

    # Calculate head dimension
    head_dim = model.embed_dim // model.num_heads
    if model.embed_dim % model.num_heads != 0:
        raise ValueError(
            f"Invalid model architecture for multi-head attention\n"
            f"  ❌ embed_dim={model.embed_dim} is not divisible by num_heads={model.num_heads} (remainder: {model.embed_dim % model.num_heads})\n"
            f"  💡 Each attention head needs equal dimensions. embed_dim must be evenly divisible by num_heads.\n"
            f"  🔧 Use embed_dim={model.num_heads * (model.embed_dim // model.num_heads + 1)} (next valid size) or num_heads={[h for h in [1,2,4,8,12,16] if model.embed_dim % h == 0]}"
        )

    cache = KVCache(
        batch_size=1,
        max_seq_len=model.max_seq_len,
        num_layers=model.num_layers,
        num_heads=model.num_heads,
        head_dim=head_dim,
    )
    model._kv_cache = cache
    model._cache_enabled = True
    return cache, head_dim


def _cached_attention_forward(block, x, cache_obj, layer_idx, original_forward):
    """Dispatch attention through the correct path based on context.

    EXAMPLE:
    >>> # Training path (seq_len=10 > 1):
    >>> output = _cached_attention_forward(block, x_train, cache, 0, orig_fwd)
    >>> # -> calls original_forward(x_train, None)
    >>>
    >>> # Cached path (seq_len=1, cache has history):
    >>> output = _cached_attention_forward(block, x_gen, cache, 0, orig_fwd)
    >>> # -> calls _cached_generation_step(x_gen, block.attention, cache, 0)
    """
    seq_len = x.shape[1]

    # PATH 1: TRAINING (seq_len > 1)
    # Full sequence - use original attention for gradient flow
    if seq_len > 1:
        return original_forward(x, None)

    # PATH 2: FIRST TOKEN (cache empty)
    # Nothing to retrieve yet - use original attention
    if cache_obj.seq_pos == 0:
        return original_forward(x, None)

    # PATH 3: CACHED GENERATION
    # Use helper function for the O(n) cached computation
    return _cached_generation_step(x, block.attention, cache_obj, layer_idx)


def _cached_generate(model, prompt_tokens, max_new_tokens, temperature, cache):
    """Run autoregressive generation using the KV cache.

    EXAMPLE:
    >>> generated = _cached_generate(model, prompt=[0, 1, 2],
    ...                               max_new_tokens=5, temperature=1.0,
    ...                               cache=cache)
    >>> len(generated)  # 5 new tokens
    """
    generated = []

    # Prefill: process entire prompt to populate cache
    prompt_array = np.array([prompt_tokens])
    prompt_tensor = Tensor(prompt_array)
    logits = model.forward(prompt_tensor)

    for _ in range(len(prompt_tokens)):
        cache.advance()

    # Get logits for last prompt position (predicts next token)
    last_logits = logits.data[0, -1, :]  # (vocab_size,)

    # Generate: one token at a time using cache
    for _ in range(max_new_tokens):
        scaled_logits = last_logits / max(temperature, 1e-8)
        max_logit = np.max(scaled_logits)
        exp_logits = np.exp(scaled_logits - max_logit)
        probs = exp_logits / np.sum(exp_logits)

        # Sample next token
        next_token = int(np.random.choice(len(probs), p=probs))
        generated.append(next_token)

        # Feed single token through model (cache handles history)
        token_tensor = Tensor(np.array([[next_token]]))  # (1, 1)
        logits = model.forward(token_tensor)  # (1, 1, vocab_size)
        cache.advance()

        last_logits = logits.data[0, -1, :]

    return generated


def enable_kv_cache(model):
    """Enable KV caching for a transformer model WITHOUT modifying previous Module code.

    EXAMPLE:
    >>> from tinytorch.core.transformers import GPT
    >>> model = GPT(vocab_size=100, embed_dim=128, num_layers=4, num_heads=4)
    >>> cache = enable_kv_cache(model)
    >>> hasattr(model, '_kv_cache')  # True
    >>> model._cache_enabled  # True
    >>> cache.num_layers  # 4 (matches model)
    """
    # Step 1: Validate model and create cache
    cache, head_dim = _create_cache_storage(model)

    # Step 2: Patch each transformer block's attention
    for layer_idx, block in enumerate(model.blocks):
        if not hasattr(block, "_original_attention_forward"):
            block._original_attention_forward = block.attention.forward

        def make_cached_forward(layer_idx, original_forward, cache_obj):
            """Factory to create cached forward with correct layer_idx closure."""

            def cached_forward(x, mask=None):
                return _cached_attention_forward(
                    block, x, cache_obj, layer_idx, original_forward
                )

            return cached_forward

        block.attention.forward = make_cached_forward(
            layer_idx, block._original_attention_forward, cache
        )

    # Step 3: Print confirmation
    print(f"⚡ KV Cache enabled for model!")
    print(
        f"   Architecture: {model.num_layers} layers × {model.num_heads} heads × {head_dim}D"
    )
    print(f"   Memory: {cache.get_memory_usage()['total_mb']:.2f} MB")
    print(f"   Cache stored in: model._kv_cache")
    print()
    print(f"💡 To disable: call disable_kv_cache(model)")
    print()

    return cache


def disable_kv_cache(model):
    """Disable KV caching and restore original attention behavior.

    EXAMPLE:
    cache = enable_kv_cache(model)
    # ... do cached generation ...
    disable_kv_cache(model)  # Back to normal
    """
    if not hasattr(model, "_cache_enabled") or not model._cache_enabled:
        print("⚠️  KV cache not enabled on this model")
        return

    for block in model.blocks:
        if hasattr(block, "_original_attention_forward"):
            block.attention.forward = block._original_attention_forward

    model._cache_enabled = False
    if hasattr(model, "_kv_cache"):
        delattr(model, "_kv_cache")

    print("✓ KV cache disabled, original attention restored")
