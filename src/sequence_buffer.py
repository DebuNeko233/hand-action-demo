from __future__ import annotations
from collections import deque
import numpy as np

class SequenceBuffer:
    def __init__(self, length: int, feature_dim: int) -> None:
        self.length = length
        self.feature_dim = feature_dim
        self.buffer: deque[np.ndarray] = deque(maxlen=length)

    def append(self, feature: np.ndarray) -> None:
        arr = np.asarray(feature, dtype=np.float32)
        if arr.shape != (self.feature_dim,):
            raise ValueError(f"Expected ({self.feature_dim},), got {arr.shape}")
        self.buffer.append(arr)

    def is_ready(self) -> bool:
        return len(self.buffer) == self.length

    def get_sequence(self) -> np.ndarray:
        if not self.is_ready():
            raise RuntimeError("Sequence buffer is not full")
        return np.stack(self.buffer, axis=0).astype(np.float32)

    def clear(self) -> None:
        self.buffer.clear()

    def __len__(self) -> int:
        return len(self.buffer)
