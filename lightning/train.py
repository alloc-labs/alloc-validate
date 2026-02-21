#!/usr/bin/env python3
"""Lightning Trainer with alloc callback.

Trains a configurable model on randomly generated 32x32 images for a few steps
using PyTorch Lightning. Fully deterministic (seeded), no downloads,
completes in < 60s on CPU.

Signature scenario: callback integration — validates that
`alloc.LightningCallback()` correctly hooks into the Lightning Trainer loop
and that callback timing data appears in the artifact sidecar.

Usage:
    alloc run -- python lightning/train.py
    alloc run -- python lightning/train.py --model medium-cnn --max-steps 50
"""

from __future__ import annotations

import argparse

import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import alloc

try:
    _lightning_callback_cls = alloc.LightningCallback  # type: ignore[attr-defined]
except AttributeError:
    _lightning_callback_cls = None


# ---------------------------------------------------------------------------
# Model definitions (same architectures as pytorch/ workload)
# ---------------------------------------------------------------------------


class SmallCNN(nn.Module):
    """Minimal CNN for 10-class image classification (~26K params)."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class MediumCNN(nn.Module):
    """4 conv layers, wider channels (~200K params)."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class MLPModel(nn.Module):
    """3-layer MLP (~50K params)."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "small-cnn": SmallCNN,
    "medium-cnn": MediumCNN,
    "mlp": MLPModel,
}


# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------


class ImageClassifier(L.LightningModule):
    """Wraps a torch.nn.Module for Lightning training."""

    def __init__(self, model: nn.Module, lr: float = 1e-3) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(self.parameters(), lr=self.lr)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def make_synthetic_data(n_samples: int = 2048) -> TensorDataset:
    """Generate synthetic 32x32 RGB images and random labels."""
    gen = torch.Generator().manual_seed(42)
    images = torch.randn(n_samples, 3, 32, 32, generator=gen)
    labels = torch.randint(0, 10, (n_samples,), generator=gen)
    return TensorDataset(images, labels)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=100, help="Max training steps")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model",
        choices=list(MODEL_REGISTRY),
        default="small-cnn",
        help="Model architecture to train",
    )
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs (informational)")
    args = parser.parse_args()

    L.seed_everything(args.seed)

    print(f"Using accelerator: {'gpu' if torch.cuda.is_available() else 'cpu'}")
    print(f"Num GPUs (metadata): {args.num_gpus}")

    model_cls = MODEL_REGISTRY[args.model]
    backbone = model_cls()
    param_count = sum(p.numel() for p in backbone.parameters())
    print(f"Model: {args.model} ({model_cls.__name__}) | Params: {param_count:,}")

    lit_model = ImageClassifier(backbone)

    dataset = make_synthetic_data()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    cbs: list[L.Callback] = []
    if _lightning_callback_cls is not None:
        cbs.append(_lightning_callback_cls())
        print("alloc.LightningCallback() registered")
    else:
        print("WARN: alloc.LightningCallback not available — training without callback")

    trainer = L.Trainer(
        max_steps=args.max_steps,
        accelerator="auto",
        devices=1,
        callbacks=cbs,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=True,
    )

    trainer.fit(lit_model, train_dataloaders=loader)
    print("Training complete.")


if __name__ == "__main__":
    main()
