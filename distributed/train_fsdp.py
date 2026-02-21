#!/usr/bin/env python3
"""FSDP (FullyShardedDataParallel) training on synthetic transformer data.

Trains a configurable synthetic transformer using PyTorch FSDP via torchrun.
Fully deterministic (seeded), no downloads, completes in < 60s on CPU.

Signature scenario: FSDP topology — validates that `alloc run` detects FSDP
strategy and reports topology fields in the artifact. Uses a larger default
model (medium ~150M params) to make FSDP sharding meaningful.

Usage:
    alloc run -- torchrun --nproc_per_node=4 distributed/train_fsdp.py
    alloc run -- torchrun --nproc_per_node=1 distributed/train_fsdp.py --model small
    alloc run -- python distributed/train_fsdp.py  # single-process fallback
"""

from __future__ import annotations

import argparse
import functools
import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, DistributedSampler

from models import MODEL_CONFIGS, GPU_ONLY_MODELS, build_model, make_synthetic_data


# ---------------------------------------------------------------------------
# FSDP setup
# ---------------------------------------------------------------------------


def setup_fsdp() -> tuple[int, int]:
    """Initialize process group for FSDP. Returns (rank, world_size)."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="gloo")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def wrap_fsdp(model: nn.Module, device: torch.device) -> nn.Module:
    """Wrap model with FSDP using transformer auto-wrap policy."""
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={nn.TransformerEncoderLayer},
    )
    return FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        device_id=device if device.type == "cuda" else None,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=5, help="Max training steps")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model",
        choices=list(MODEL_CONFIGS),
        default="medium",
        help="Model size variant",
    )
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs (informational)")
    args = parser.parse_args()

    if args.model in GPU_ONLY_MODELS and not torch.cuda.is_available():
        print(f"SKIP: model '{args.model}' requires GPU")
        sys.exit(0)

    rank, world_size = setup_fsdp()
    torch.manual_seed(args.seed + rank)

    device = torch.device("cuda", rank) if torch.cuda.is_available() else torch.device("cpu")
    if rank == 0:
        print(f"Strategy: FSDP | World size: {world_size} | Device: {device}")

    cfg = MODEL_CONFIGS[args.model]
    model = build_model(args.model).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"Model: {args.model} (SyntheticTransformer) | Params: {param_count:,}")

    # FSDP requires CUDA — skip wrapping on CPU/MPS to allow CPU smoke tests
    if torch.cuda.is_available() and (world_size > 1 or dist.is_initialized()):
        model = wrap_fsdp(model, device)
    elif not torch.cuda.is_available():
        if rank == 0:
            print("WARN: FSDP wrapping skipped (requires CUDA) — running unwrapped on CPU")

    dataset = make_synthetic_data(cfg["d_model"])
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

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
            if rank == 0 and step % 2 == 0:
                print(f"Step {step}/{args.max_steps} | Loss: {loss.item():.4f}")

    if rank == 0:
        print(f"FSDP training complete. {step} steps, {world_size} processes.")

    cleanup()


if __name__ == "__main__":
    main()
