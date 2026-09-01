from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

class HandSequenceDataset(Dataset):
    def __init__(self, root: Path, class_names: list[str], augment: bool = False) -> None:
        self.samples: list[tuple[Path, int]] = []
        self.augment = augment
        for label, name in enumerate(class_names):
            for p in sorted((root / name).glob("*.npy")):
                self.samples.append((p, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        x = np.load(path).astype(np.float32)
        if x.ndim != 2:
            raise ValueError(f"Invalid sequence shape in {path}: {x.shape}")
        if self.augment:
            x = x.copy()
            x[:, :63] *= np.random.uniform(0.9, 1.1)
            x += np.random.normal(0.0, 0.005, size=x.shape).astype(np.float32)
        return torch.from_numpy(x), torch.tensor(label, dtype=torch.long)
