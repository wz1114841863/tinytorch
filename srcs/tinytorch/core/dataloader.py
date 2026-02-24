import numpy as np
import random
import time
import sys

from typing import Iterator, Tuple, List, Optional, Union
from abc import ABC, abstractmethod
from .tensor import Tensor

__all__ = [
    "Dataset",
    "TensorDataset",
    "DataLoader",
    "RandomHorizontalFlip",
    "RandomCrop",
    "Compose",
]


class Dataset(ABC):
    """Abstract base class for all datasets.

    EXAMPLE:
    >>> class MyDataset(Dataset):
    ...     def __len__(self): return 100
    ...     def __getitem__(self, idx): return idx
    >>> dataset = MyDataset()
    >>> print(len(dataset))  # 100
    >>> print(dataset[42])   # 42
    """

    @abstractmethod
    def __len__(self):
        """Return the total number of samples in the datasets."""
        pass

    @abstractmethod
    def __getitem__(self, idx):
        """Return the sample at the given index."""
        pass


class TensorDataset(Dataset):
    """Dataset wrapping tensors for supervised learning.

    EXAMPLE:
    >>> features = Tensor([[1, 2], [3, 4], [5, 6]])  # 3 samples, 2 features each
    >>> labels = Tensor([0, 1, 0])                    # 3 labels
    >>> dataset = TensorDataset(features, labels)
    >>> print(len(dataset))  # 3
    >>> print(dataset[1])    # (Tensor([3, 4]), Tensor(1)
    """

    def __init__(self, *tensors):
        """Create dataset from multiple tensors."""
        assert len(tensors) > 0, "Must provide at least ont tensor."

        self.tensors = tensors
        first_size = len(tensors[0].data)
        for i, tensor in enumerate(tensors):
            if len(tensor.data) != first_size:
                raise ValueError(
                    f"All tensors must have same size in first dimension. "
                    f"Tensor 0: {first_size}, Tensor {i}: {len(tensor.data)}"
                )

    def __len__(self):
        """Return number of samples (size of first dimension).

        EXAMPLE:
        >>> features = Tensor([[1, 2], [3, 4], [5, 6]])  # 3 samples
        >>> labels = Tensor([0, 1, 0])
        >>> dataset = TensorDataset(features, labels)
        >>> print(len(dataset))  # 3
        """
        return len(self.tensors[0].data)

    def __getitem__(self, idx):
        """Return tuple of tensor slices at given index"""
        if idx >= len(self) or idx < 0:
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self)}"
            )
        return tuple(Tensor(tensor.data[idx]) for tensor in self.tensors)


class DataLoader:
    """Data loader with batching and shuffling support.

    EXAMPLE:
    >>> dataset = TensorDataset(Tensor([[1,2], [3,4], [5,6]]), Tensor([0,1,0]))
    >>> loader = DataLoader(dataset, batch_size=2, shuffle=True)
    >>> for batch in loader:
    ...     features_batch, labels_batch = batch
    ...     print(f"Features: {features_batch.shape}, Labels: {labels_batch.shape}")
    """

    def __init__(self, dataset, batch_size, shuffle=False):
        """Create Dataloader for batched iteration."""
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        """Return number of batches per epoch

        EXAMPLE:
        >>> dataset = TensorDataset(Tensor([[1], [2], [3], [4], [5]]))
        >>> loader = DataLoader(dataset, batch_size=2)
        >>> print(len(loader))  # 3 (batches: [2, 2, 1])
        """
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        """Return iterator over batches.

        EXAMPLE:
        >>> dataset = TensorDataset(Tensor([[1], [2], [3], [4]]))
        >>> loader = DataLoader(dataset, batch_size=2)
        >>> for batch in loader:
        ...     print(batch[0].shape)  # (2, 1)
        """
        indices = list(range(len(self.dataset)))

        if self.shuffle:
            random.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            batch = [self.dataset[idx] for idx in batch_indices]
            yield self._collate_batch(batch)

    def _collate_batch(self, batch):
        """Collate individual samples into batch tensors.

        EXAMPLE:
        >>> # batch = [(Tensor([1,2]), Tensor(0)),
        ...            (Tensor([3,4]), Tensor(1))]
        >>> # Returns: (Tensor([[1,2], [3,4]]), Tensor([0, 1]))
        """
        if len(batch) == 0:
            return ()

        num_tensors = len(batch[0])
        batched_tensors = []
        for tensor_idx in range(num_tensors):
            tensor_list = [sample[tensor_idx].data for sample in batch]

            batched_data = np.stack(tensor_list, axis=0)
            batched_tensors.append(Tensor(batched_data))

        return tuple(batched_tensors)


class RandomHorizontalFlip:
    """Randomly flip images horizontally with given probability."""

    def __init__(self, p=0.5):
        """Initialize RandomHorizontalFlip."""
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"Probability must be between 0 and 1, got {p}")
        self.p = p

    def __call__(self, x):
        """Apply random horizontal flip to input.

        EXAMPLE:
        >>> flip = RandomHorizontalFlip(0.5)
        >>> img = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3 image
        >>> # 50% chance output is [[3, 2, 1], [6, 5, 4]]
        """
        if np.random.random() < self.p:
            if isinstance(x, Tensor):
                return Tensor(np.flip(x.data, axis=-1).copy())
            else:
                return np.flip(x, axis=-1).copy()

        return x


class RandomCrop:
    """Randomly crop image after padding."""

    def __init__(self, size, padding=4):
        """Initialize RandomCrop.

        EXAMPLE:
        >>> crop = RandomCrop(32, padding=4)  # CIFAR-10 standard
        >>> # Pads to 40x40, then crops back to 32x32
        """
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.padding = padding

    def __call__(self, x):
        """Apply random crop after padding.

        EXAMPLE:
        >>> crop = RandomCrop(32, padding=4)
        >>> img = np.random.randn(3, 32, 32)  # CIFAR-10 format (C, H, W)
        >>> out = crop(img)
        >>> print(out.shape)  # (3, 32, 32)
        """
        is_tensor = isinstance(x, Tensor)
        data = x.data if is_tensor else x

        target_h, target_w = self.size

        if len(data.shape) == 2:
            # (H, W) format
            h, w = data.shape
            padded = np.pad(data, self.padding, mode="constant", constant_values=0)

            top = np.random.randint(0, 2 * self.padding + h - target_h + 1)
            left = np.random.randint(0, 2 * self.padding + w - target_w + 1)

            cropped = padded[top : top + target_h, left : left + target_w]

        elif len(data.shape) == 3:
            if data.shape[0] <= 4:
                c, h, w = data.shape
                padded = np.pad(
                    data,
                    (
                        (0, 0),
                        (self.padding, self.padding),
                        (self.padding, self.padding),
                    ),
                    mode="constant",
                    constant_values=0,
                )

                top = np.random.randint(0, 2 * self.padding + 1)
                left = np.random.randint(0, 2 * self.padding + 1)

                cropped = padded[:, top : top + target_h, left : left + target_w]
            else:
                h, w, c = data.shape
                padded = np.pad(
                    data,
                    (
                        (self.padding, self.padding),
                        (self.padding, self.padding),
                        (0, 0),
                    ),
                    mode="constant",
                    constant_values=0,
                )

                top = np.random.randint(0, 2 * self.padding + 1)
                left = np.random.randint(0, 2 * self.padding + 1)

                cropped = padded[top : top + target_h, left : left + target_w, :]
        else:
            raise ValueError(f"Expected 2D or 3D input, got shape {data.shape}")

        return Tensor(cropped) if is_tensor else cropped


class Compose:
    """Compose multiple transforms into a pipeline."""

    def __init__(self, transforms):
        """Initialize Compose with list of transforms.

        EXAMPLE:
        >>> transforms = Compose([
        ...     RandomHorizontalFlip(0.5),
        ...     RandomCrop(32, padding=4)
        ... ])
        """
        self.transforms = transforms

    def __call__(self, x):
        """Apply all transforms in sequence."""
        for transform in self.transforms:
            x = transform(x)
        return x
