import numpy as np

# Constants for memory calculations
BYTES_PER_FLOAT32 = 4
KB_TO_BYTES = 1024
MB_TO_BYTES = 1024 * 1024


class Tensor:
    """Education tensor: the foundation of machine learning computaion.

    This class provide the core data structure for all ML operations:
        - data: The actual numerical values (Numpy Array)
        - shape: Dimensions of the tensor
        - size: Total number of elements
        - dtype: Data type (float32)

    All arithmetic, matrix, and shape operations are built on this foundation.
    """

    def __init__(self, data):
        """Create a new tensor from data.

        EXAMPLE:
        >>> t = Tensor([1, 2, 3])
        >>> print(t.shape)
        (3,)
        >>> print(t.size)
        3
        """
        self.data = np.array(data, dtype=np.float32)
        self.shape = self.data.shape
        self.size = self.data.size
        self.dtype = self.data.dtype

    def __repr__(self):
        """String representation of tensor for debugging."""
        return f"Tensor(data={self.data}, shape={self.shape})"

    def __str__(self):
        """Human-readable string representation."""
        return f"Tensor({self.data})"

    def numpy(self):
        """Return the underlying Numpy array."""
        return self.data

    def memory_footprint(self):
        """Calculate exact memory usage in bytes.

        Returns:
            int: Memory usage in bytes (e.g., 1000x1000 float32 = 4MB)
        """
        return self.data.nbytes

    def __add__(self, other):
        """Add two tensors element-wise with broadcasting support.

        EXAMPLE:
        >>> a = Tensor([1, 2, 3])
        >>> b = Tensor([4, 5, 6])
        >>> c = a + b
        >>> print(c.data)
        [5. 7. 9.]
        """
        if isinstance(other, Tensor):
            return Tensor(self.data + other.data)
        else:
            return Tensor(self.data + other)

    def __sub__(self, other):
        """Subtract two tensors element-wise.

        EXAMPLE:
        >>> a = Tensor([5, 7, 9])
        >>> b = Tensor([1, 2, 3])
        >>> c = a - b
        >>> print(c.data)
        [4. 5. 6.]
        """
        if isinstance(other, Tensor):
            return Tensor(self.data - other.data)
        else:
            return Tensor(self.data - other)

    def __mul__(self, other):
        """Multiply two tensors element-wise (NOT matrix multiplication).

        EXAMPLE:
        >>> a = Tensor([1, 2, 3])
        >>> b = Tensor([4, 5, 6])
        >>> c = a * b
        >>> print(c.data)
        [ 4. 10. 18.]
        """
        if isinstance(other, Tensor):
            return Tensor(self.data * other.data)
        else:
            return Tensor(self.data * other)

    def __truediv__(self, other):
        """Divide two tensors element-wise.

        EXAMPLE:
        >>> a = Tensor([4, 6, 8])
        >>> b = Tensor([2, 2, 2])
        >>> c = a / b
        >>> print(c.data)
        [2. 3. 4.]
        """
        if isinstance(self, Tensor):
            return Tensor(self.data / other.data)
        else:
            return Tensor(self.data / other)

    def matmul(self, other):
        """Matrix multiplication of two tensors.

        EXAMPLE:
        >>> a = Tensor([[1, 2], [3, 4]])  # 2x2
        >>> b = Tensor([[5, 6], [7, 8]])  # 2x2
        >>> c = a.matmul(b)
        >>> print(c.data)
        [[19. 22.]
        [43. 50.]]
        """
        if not isinstance(other, Tensor):
            raise TypeError(
                f"Expected Tensor for matrix multiplication, got {type(other)}"
            )
        if self.shape == () or other.shape == ():
            return Tensor(self.data * other.data)
        if len(self.shape) == 0 or len(self.shape) == 0:
            return Tensor(self.data * other.data)
        if len(self.shape) >= 2 and len(other.shape) >= 2:
            if self.shape[-1] != other.shape[-2]:
                raise ValueError(
                    f"Cannot perform matrix multipication: {self.shape} @ {other.shape}"
                    f"Inner dimensions must match: {self.shape[-1]} ≠ {other.shape[-2]}"
                )
        a = self.data
        b = other.data

        if len(a.shape) == 2 and len(b.shape) == 2:
            M, K = a.shape
            K2, N = b.shape
            result_data = np.zeros((M, N), dtype=a.dtype)

            # Explicit nested loops
            # Each output element is a dot product of a row from A and a column from B\
            for i in range(M):
                for j in range(N):
                    # Dot product of row i from A with column j from B
                    result_data[i, j] = np.dot(a[i, :], b[:, j])
        else:
            # For batched operations (3D+), use np.matmul for correctness
            result_data = np.matmul(a, b)

        return Tensor(result_data)

    def __matmul__(self, other):
        """Enable @ operator for matrix multipication."""
        return self.matmul(other)

    def __getitem__(self, key):
        """Enable indexing and slicing operations on Tensors.

        EXAMPLE:
        >>> t = Tensor([[1, 2, 3], [4, 5, 6]])
        >>> row = t[0]  # First row
        >>> print(row.data)
        [1. 2. 3.]
        >>> element = t[0, 1]  # Single element
        >>> print(element.data)
        2.0
        """
        result_data = self.data[key]
        if not isinstance(result_data, np.ndarray):
            result_data = np.array(result_data)
        return Tensor(result_data)

    def reshape(self, *shape):
        """Reshape tensor to new dimensions.

        EXAMPLE:
        >>> t = Tensor([1, 2, 3, 4, 5, 6])
        >>> reshaped = t.reshape(2, 3)
        >>> print(reshaped.data)
        [[1. 2. 3.]
         [4. 5. 6.]]
        >>> auto = t.reshape(2, -1)  # Infers -1 as 3
        >>> print(auto.shape)
        (2, 3)
        """
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            new_shape = tuple(shape[0])
        else:
            new_shape = shape

        # process -1
        if -1 in new_shape:
            if new_shape.count(-1) > 1:
                raise ValueError("Can only specify unkown dimension with -1.")
            known_size = 1
            unknown_idx = new_shape.index(-1)
            for i, dim in enumerate(new_shape):
                if i != unknown_idx:
                    known_size *= dim
            unknown_dim = self.size // known_size
            new_shape = list(new_shape)
            new_shape[unknown_idx] = unknown_dim
            new_shape = tuple(new_shape)

        if np.prod(new_shape) != self.size:
            target_size = int(np.prod(new_shape))
            raise ValueError(f"Total elements must matchL {self.size} ≠ {target_size}")
        reshape_data = np.reshape(self.data, new_shape)
        return Tensor(reshape_data)

    def transpose(self, dim0=None, dim1=None):
        """Transpose tensor dimensions.

         EXAMPLE:
        >>> t = Tensor([[1, 2, 3], [4, 5, 6]])  # 2x3
        >>> transposed = t.transpose()
        >>> print(transposed.data)
            [[1. 4.]
            [2. 5.]
            [3. 6.]]  # 3x2
        """
        if dim0 is None and dim1 is None:
            if len(self.shape) < 2:
                return Tensor(self.data.copy())
            else:
                axes = list(range(len(self.shape)))
                axes[-2], axes[-1] = axes[-1], axes[-2]
                transposed_data = np.transpose(self.data, axes)
        else:
            if dim0 is None or dim1 is None:
                raise ValueError("Both dim0 and dim1 must be specified")
            axes = list(range(len(self.shape)))
            axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
            transposed_data = np.transpose(self.data, axes)
        return Tensor(transposed_data)

    def sum(self, axis=None, keepdims=False):
        """Sum tensor along specified axis.

        Params:
            axis=N sums along dimension N
            keepdims=True preserves original number of dimensions

        EXAMPLE:
        >>> t = Tensor([[1, 2, 3], [4, 5, 6]])
        >>> total = t.sum()
        >>> print(total.data)
        21.0
        >>> col_sum = t.sum(axis=0)
        >>> print(col_sum.data)
        [5. 7. 9.]
        """
        result = np.sum(self.data, axis=axis, keepdims=keepdims)
        return Tensor(result)

    def mean(self, axis=None, keepdims=False):
        """Mean tensor along specified axis.

        Params:
            axis=N means along dimension N
            keepdims=True preserves original number of dimensions

        EXAMPLE:
        >>> t = Tensor([[1, 2, 3], [4, 5, 6]])
        >>> avg = t.mean()
        >>> print(avg.data)
        3.5
        >>> col_mean = t.mean(axis=0)
        >>> print(col_mean.data)
        [2.5 3.5 4.5]
        """
        result = np.mean(self.data, axis=axis, keepdims=keepdims)
        return Tensor(result)

    def max(self, axis=None, keepdims=False):
        """Max tensor along specified axis.

        Params:
            axis=N maxs along dimension N
            keepdims=True preserves original number of dimensions

        EXAMPLE:
        >>> t = Tensor([[1, 2, 3], [4, 5, 6]])
        >>> maximum = t.max()
        >>> print(maximum.data)
        6.0
        >>> row_max = t.max(axis=1)
        >>> print(row_max.data)
        [3. 6.]
        """
        result = np.max(self.data, axis=axis, keepdims=keepdims)
        return Tensor(result)
