#!/usr/bin/env python3
"""Ghost target: MLPLarge (~526K params).

Usage:
    alloc ghost scan-only/ghost_target_mlp_large.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim


class MLPLarge(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1024, 128),
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


def main() -> None:
    model = MLPLarge()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    x = torch.randn(32, 1024)
    y = torch.randint(0, 10, (32,))
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()


if __name__ == "__main__":
    main()
