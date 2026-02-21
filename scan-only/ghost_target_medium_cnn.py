#!/usr/bin/env python3
"""Ghost target: MediumCNN (~200K params).

Usage:
    alloc ghost scan-only/ghost_target_medium_cnn.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim


class MediumCNN(nn.Module):
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


def main() -> None:
    model = MediumCNN()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    x = torch.randn(32, 3, 32, 32)
    y = torch.randint(0, 10, (32,))
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()


if __name__ == "__main__":
    main()
