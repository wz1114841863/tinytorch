# src/tinytorch/__init__.py
__version__ = "0.1.0"

# 函数和类的导入示例
from .core.tensor import Tensor
from .core.activations import Sigmoid, ReLU, Tanh, GELU, Softmax
from .core.layers import Linear, Dropout, Sequential
