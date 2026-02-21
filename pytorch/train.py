#!/usr/bin/env python3
"""Vanilla PyTorch models on synthetic image data.

Trains a configurable model on randomly generated 32x32 images for a few steps.
Fully deterministic (seeded), no downloads, completes in < 60s on CPU.

Signature scenario: baseline training — validates that `alloc run` produces
a correct artifact for a standard PyTorch training loop.

Usage:
    alloc run python pytorch/train.py
    alloc run python pytorch/train.py --model medium-cnn --max-steps 50
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


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


class LargeCNN(nn.Module):
    """6 conv layers, widest channels (~1M params)."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class MLPSmall(nn.Module):
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


class MLPLarge(nn.Module):
    """5-layer MLP (~526K params)."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "small-cnn": SmallCNN,
    "medium-cnn": MediumCNN,
    "large-cnn": LargeCNN,
    "mlp-small": MLPSmall,
    "mlp-large": MLPLarge,
}


def make_synthetic_data(n_samples: int = 2048) -> TensorDataset:
    """Generate synthetic 32x32 RGB images and random labels."""
    gen = torch.Generator().manual_seed(42)
    images = torch.randn(n_samples, 3, 32, 32, generator=gen)
    labels = torch.randint(0, 10, (n_samples,), generator=gen)
    return TensorDataset(images, labels)


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

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Num GPUs (metadata): {args.num_gpus}")

    dataset = make_synthetic_data()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model_cls = MODEL_REGISTRY[args.model]
    model = model_cls().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model} ({model_cls.__name__}) | Params: {param_count:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    step = 0
    model.train()
    while step < args.max_steps:
        for inputs, targets in loader:
            if step >= args.max_steps:
                break
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            step += 1
            if step % 20 == 0:
                print(f"Step {step}/{args.max_steps} | Loss: {loss.item():.4f}")

    print(f"Training complete. {step} steps.")


if __name__ == "__main__":
    main()
