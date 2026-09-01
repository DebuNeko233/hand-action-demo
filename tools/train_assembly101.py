from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset

from config import HIDDEN_SIZE, INPUT_SIZE, NUM_LAYERS, SEQUENCE_LENGTH
from src.model import HandActionLSTM


class Assembly101Dataset(Dataset):
    def __init__(self, root: Path, split: str, class_names: list[str], augment: bool = False) -> None:
        self.root = root
        self.augment = augment
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        self.samples: list[tuple[Path, int]] = []
        split_root = root / split
        for name in class_names:
            for p in sorted((split_root / name).glob("*.npy")):
                self.samples.append((p, self.class_to_idx[name]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        x = np.load(path).astype(np.float32)
        if x.shape != (SEQUENCE_LENGTH, INPUT_SIZE):
            raise ValueError(f"Invalid sequence shape in {path}: {x.shape}")
        if self.augment:
            x = x.copy()
            x[:, :63] *= np.random.uniform(0.95, 1.05)
            x += np.random.normal(0.0, 0.003, size=x.shape).astype(np.float32)
        return torch.from_numpy(x), torch.tensor(label, dtype=torch.long)


def evaluate(model, loader, device, loss_fn):
    model.eval()
    total_loss = correct = total = 0
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            total_loss += loss.item() * len(y)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += len(y)
            ys.extend(y.cpu().numpy().tolist())
            ps.extend(pred.cpu().numpy().tolist())
    return total_loss / max(1, total), correct / max(1, total), ys, ps


def main() -> None:
    ap = argparse.ArgumentParser(description="Train HandActionLSTM on MediaPipe features extracted from Assembly101.")
    ap.add_argument("--dataset", type=Path, default=ROOT / "dataset_assembly101")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--model-out", type=Path, default=ROOT / "models" / "hand_action_assembly101_lstm.pth")
    args = ap.parse_args()

    manifest_path = args.dataset / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing {manifest_path}. Run tools/prepare_assembly101.py first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    class_names = manifest.get("class_names", [])
    if not class_names:
        raise RuntimeError("Assembly101 manifest contains no classes.")

    seed = 42
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    train_ds = Assembly101Dataset(args.dataset, "train", class_names, augment=True)
    val_ds = Assembly101Dataset(args.dataset, "validation", class_names, augment=False)
    test_ds = Assembly101Dataset(args.dataset, "test", class_names, augment=False)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError("Train/validation dataset is empty.")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0) if len(test_ds) else None

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    model = HandActionLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, len(class_names)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    best_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = correct = total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)

        tr_loss = train_loss / max(1, total)
        tr_acc = correct / max(1, total)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, device, loss_fn)
        scheduler.step(val_loss)
        print(f"Epoch {epoch:03d} train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "input_size": INPUT_SIZE,
                "sequence_length": SEQUENCE_LENGTH,
                "hidden_size": HIDDEN_SIZE,
                "num_layers": NUM_LAYERS,
                "dataset": "Assembly101-MediaPipe",
                "label_level": manifest.get("label_level", "verb"),
            }, args.model_out)

    ckpt = torch.load(args.model_out, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    target_loader = test_loader if test_loader is not None else val_loader
    split_name = "test" if test_loader is not None else "validation"
    loss, acc, ys, ps = evaluate(model, target_loader, device, loss_fn)
    print(f"best_val_acc={best_acc:.4f} {split_name}_loss={loss:.4f} {split_name}_acc={acc:.4f}")
    print(classification_report(ys, ps, labels=list(range(len(class_names))), target_names=class_names, zero_division=0))
    print(confusion_matrix(ys, ps, labels=list(range(len(class_names)))))
    print(f"model={args.model_out}")


if __name__ == "__main__":
    main()
