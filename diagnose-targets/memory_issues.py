#!/usr/bin/env python3
"""Training script with intentional memory/throughput issues.

Triggers: MEM002 (no mixed precision), MEM005 (no torch.compile)

NOTE: This is a diagnose target (AST analysis only). Uses .cuda() explicitly
so the AST analyzer detects GPU training. Not meant to be executed on CPU.
cudnn.benchmark is set (so THRU001 doesn't suppress MEM002/MEM005).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

torch.backends.cudnn.benchmark = True


class SmallCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 8 * 8, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def main() -> None:
    torch.manual_seed(42)

    gen = torch.Generator().manual_seed(42)
    images = torch.randn(256, 3, 32, 32, generator=gen)
    labels = torch.randint(0, 10, (256,), generator=gen)
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True,
                        persistent_workers=True, prefetch_factor=2)

    # No mixed precision → MEM002
    # No torch.compile → MEM005
    model = SmallCNN()
    model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    model.train()
    for step, (inputs, targets) in enumerate(loader):
        if step >= 10:
            break
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()

    print("Done.")


if __name__ == "__main__":
    main()
