# src/tinytorch/__init__.py
__version__ = "0.1.0"

# 函数和类的导入示例
from .core.tensor import Tensor
from .core.activations import Sigmoid, ReLU, Tanh, GELU, Softmax
from .core.layers import Linear, Dropout, Sequential
from .core.losses import BinaryCrossEntropyLoss, CrossEntropyLoss, MSELoss
from .core.dataloader import (
    Dataset,
    DataLoader,
    TensorDataset,
    RandomHorizontalFlip,
    RandomCrop,
    Compose,
)
from .core.autograd import enable_autograd
from .core.optimizers import SGD, Adam, AdamW
from .core.tokenization import Tokenizer, CharTokenizer, BPETokenizer
from .core.embeddings import Embedding, PositionalEncoding, EmbeddingLayer
from .core.transformers import (
    LayerNorm,
    MLP,
    TransformerBlock,
    GPT,
    TinyGPT,
    create_causal_mask,
)
from .perf.profiling import Profiler, quick_profile

enable_autograd()
