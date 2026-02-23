#!/usr/bin/env python3
"""Training script with intentional DataLoader issues.

Triggers: DL001 (low workers), DL002 (no pin_memory), DL003 (no persistent_workers),
          DL004 (no prefetch_factor), THRU001 (no cudnn.benchmark)

Uses num_workers=1 so DL001/DL003/DL004 fire (they require num_workers > 0).
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

    # DL001: num_workers=1 (too low)
    # DL002: no pin_memory
    # DL003: no persistent_workers
    # DL004: no prefetch_factor
    loader = DataLoader(dataset, batch_size=32, num_workers=1)

    model = TinyModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    model.train()
    for step, (inputs, targets) in enumerate(loader):
        if step >= 10:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()

    print("Done.")


if __name__ == "__main__":
    main()
