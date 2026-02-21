#!/usr/bin/env python3
"""Target script for `alloc ghost` static VRAM analysis.

This is a minimal training script that Ghost can statically analyze to
produce a VRAM breakdown (weights, gradients, optimizer, activations, total).

It is NOT meant to be run directly — it's the input for:
    alloc ghost scan-only/ghost_target.py

Usage:
    alloc ghost scan-only/ghost_target.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim


class MediumModel(nn.Module):
    """A model large enough for Ghost to produce meaningful VRAM estimates."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def main() -> None:
    model = MediumModel()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Single forward/backward pass for Ghost to trace
    x = torch.randn(32, 1024)
    y = torch.randint(0, 10, (32,))
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()


if __name__ == "__main__":
    main()
