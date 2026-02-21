#!/usr/bin/env python3
"""DDP (DistributedDataParallel) training on synthetic transformer data.

Trains a configurable synthetic transformer using PyTorch DDP via torchrun.
Fully deterministic (seeded), no downloads, completes in < 60s on CPU.

Signature scenario: DDP topology — validates that `alloc run` detects DDP
strategy and reports dp_degree, num_nodes, gpus_per_node in the artifact.

Usage:
    alloc run -- torchrun --nproc_per_node=2 distributed/train_ddp.py
    alloc run -- torchrun --nproc_per_node=1 distributed/train_ddp.py --model medium
    alloc run -- python distributed/train_ddp.py  # single-process fallback
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from models import MODEL_CONFIGS, GPU_ONLY_MODELS, build_model, make_synthetic_data


# ---------------------------------------------------------------------------
# DDP setup
# ---------------------------------------------------------------------------


def setup_ddp() -> tuple[int, int]:
    """Initialize DDP process group. Returns (rank, world_size)."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="gloo")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def cleanup() -> None:
    """Destroy DDP process group if initialized."""
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=5, help="Max training steps")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model",
        choices=list(MODEL_CONFIGS),
        default="small",
        help="Model size variant",
    )
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs (informational)")
    args = parser.parse_args()

    if args.model in GPU_ONLY_MODELS and not torch.cuda.is_available():
        print(f"SKIP: model '{args.model}' requires GPU")
        sys.exit(0)

    rank, world_size = setup_ddp()
    torch.manual_seed(args.seed + rank)

    device = torch.device("cuda", rank) if torch.cuda.is_available() else torch.device("cpu")
    if rank == 0:
        print(f"Strategy: DDP | World size: {world_size} | Device: {device}")

    cfg = MODEL_CONFIGS[args.model]
    model = build_model(args.model).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"Model: {args.model} (SyntheticTransformer) | Params: {param_count:,}")

    if world_size > 1:
        model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)

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
        print(f"DDP training complete. {step} steps, {world_size} processes.")

    cleanup()


if __name__ == "__main__":
    main()
