#!/usr/bin/env python3
"""Ghost target: SmallCNN (~26K params).

Usage:
    alloc ghost scan-only/ghost_target_small_cnn.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim


class SmallCNN(nn.Module):
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


def main() -> None:
    model = SmallCNN()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    x = torch.randn(32, 3, 32, 32)
    y = torch.randint(0, 10, (32,))
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()


if __name__ == "__main__":
    main()
