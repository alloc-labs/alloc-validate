#!/usr/bin/env python3
"""Training script with intentional precision issues.

Triggers: PREC001 (deprecated torch.cuda.amp.autocast API),
          PREC002 (fp16 on Ampere+ where bf16 is better)

Uses the old-style torch.cuda.amp.autocast (deprecated in PyTorch 2.4+)
with fp16 instead of bf16.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


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
    loader = DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True,
                        persistent_workers=True, prefetch_factor=2)

    model = TinyModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    # PREC002: fp16 GradScaler (should use bf16 on Ampere+)
    scaler = torch.cuda.amp.GradScaler()

    model.train()
    for step, (inputs, targets) in enumerate(loader):
        if step >= 10:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        # PREC001: deprecated torch.cuda.amp.autocast (old API)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            loss = criterion(model(inputs), targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    print("Done.")


if __name__ == "__main__":
    main()
