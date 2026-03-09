#!/usr/bin/env python3
"""Well-optimized training script — should produce ZERO diagnose findings.

All best practices applied:
  - num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=4
  - torch.amp.autocast with bf16 (new API)
  - torch.compile
  - cudnn.benchmark = True
  - torch.set_float32_matmul_precision("high")
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gen = torch.Generator().manual_seed(42)
    images = torch.randn(256, 3, 32, 32, generator=gen)
    labels = torch.randint(0, 10, (256,), generator=gen)
    dataset = TensorDataset(images, labels)

    loader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    model = TinyModel().to(device)
    model = torch.compile(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    model.train()
    for step, (inputs, targets) in enumerate(loader):
        if step >= 10:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()

    print("Done.")


if __name__ == "__main__":
    main()
