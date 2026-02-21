"""Shared model definitions for distributed topology workloads.

Contains synthetic transformer architectures and model configs used by all
distributed training scripts (DDP, FSDP, PP, TP, and hybrid topologies).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset


# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------

MODEL_CONFIGS: dict[str, dict[str, int]] = {
    # --- Existing configs (CPU-safe) ---
    "small": {"d_model": 512, "nhead": 8, "num_layers": 6, "dim_feedforward": 2048},
    "medium": {"d_model": 1024, "nhead": 16, "num_layers": 12, "dim_feedforward": 4096},
    "large": {"d_model": 2048, "nhead": 16, "num_layers": 24, "dim_feedforward": 8192},
    "xl": {"d_model": 4096, "nhead": 32, "num_layers": 32, "dim_feedforward": 16384},
    # --- Large model configs (GPU-only, shaped like real LLMs) ---
    "1b": {"d_model": 2048, "nhead": 32, "num_layers": 20, "dim_feedforward": 8192},
    "7b": {"d_model": 4096, "nhead": 32, "num_layers": 36, "dim_feedforward": 14336},
    "13b": {"d_model": 5120, "nhead": 40, "num_layers": 40, "dim_feedforward": 20480},
    "30b": {"d_model": 6144, "nhead": 64, "num_layers": 60, "dim_feedforward": 24576},
    "70b": {"d_model": 8192, "nhead": 64, "num_layers": 80, "dim_feedforward": 28672},
}

# Models too large for CPU smoke tests — scripts should exit(0) with SKIP message
GPU_ONLY_MODELS = {"1b", "7b", "13b", "30b", "70b"}

SEQ_LEN = 64


# ---------------------------------------------------------------------------
# Synthetic transformer (used by DDP, FSDP, TP, and hybrid scripts)
# ---------------------------------------------------------------------------


class SyntheticTransformer(nn.Module):
    """Configurable transformer encoder for topology testing.

    Input: random float tensors (batch, seq_len, d_model).
    Output: (batch, num_classes) logits via mean pooling + linear head.
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        pooled = encoded.mean(dim=1)  # mean pool over sequence
        return self.head(pooled)


# ---------------------------------------------------------------------------
# Pipelined transformer (used by PP and hybrid scripts)
# ---------------------------------------------------------------------------


class TransformerStage(nn.Module):
    """A subset of transformer encoder layers — one pipeline stage."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.layers = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class PipelinedTransformer(nn.Module):
    """Transformer split into stages for pipeline parallelism.

    On CPU or single-GPU, all stages run sequentially on one device.
    On multi-GPU, stages are assigned round-robin to devices.
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        num_classes: int = 10,
        num_stages: int = 2,
    ) -> None:
        super().__init__()
        layers_per_stage = max(1, num_layers // num_stages)
        remainder = num_layers - layers_per_stage * num_stages

        self.stages = nn.ModuleList()
        for i in range(num_stages):
            n = layers_per_stage + (1 if i < remainder else 0)
            if n > 0:
                self.stages.append(
                    TransformerStage(d_model, nhead, dim_feedforward, n)
                )

        self.head = nn.Linear(d_model, num_classes)
        self._num_stages = len(self.stages)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            x = stage(x)
        pooled = x.mean(dim=1)
        return self.head(pooled)

    def assign_devices(self, devices: list[torch.device]) -> None:
        """Move each stage to a different device (round-robin)."""
        for i, stage in enumerate(self.stages):
            dev = devices[i % len(devices)]
            stage.to(dev)
        self.head.to(devices[-1])
        self._devices = devices

    @property
    def num_stages(self) -> int:
        return self._num_stages


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------


def build_model(model_name: str) -> SyntheticTransformer:
    """Build a SyntheticTransformer from a named config."""
    cfg = MODEL_CONFIGS[model_name]
    return SyntheticTransformer(**cfg)


def build_pipelined_model(model_name: str, num_stages: int = 2) -> PipelinedTransformer:
    """Build a PipelinedTransformer from a named config."""
    cfg = MODEL_CONFIGS[model_name]
    return PipelinedTransformer(**cfg, num_stages=num_stages)


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------


def make_synthetic_data(d_model: int, n_samples: int = 2048) -> TensorDataset:
    """Generate synthetic sequence data and random labels."""
    gen = torch.Generator().manual_seed(42)
    inputs = torch.randn(n_samples, SEQ_LEN, d_model, generator=gen)
    labels = torch.randint(0, 10, (n_samples,), generator=gen)
    return TensorDataset(inputs, labels)
