#!/usr/bin/env python3
"""TP+DP (Tensor Parallel + Data Parallel) training on synthetic transformer data.

Combines tensor parallelism within sub-groups with data parallelism across
sub-groups. TP group handles tensor sharding (gradient all-reduce within TP
group), DP group handles data parallelism (gradient all-reduce within DP group).

Fully deterministic (seeded), no downloads, completes in < 60s on CPU.

Signature scenario: TP+DP hybrid topology — validates that `alloc run` detects
the combined strategy and reports tp_degree and dp_degree in the artifact.

Usage:
    alloc run -- torchrun --nproc_per_node=4 distributed/train_tp_dp.py
    alloc run -- torchrun --nproc_per_node=4 distributed/train_tp_dp.py --tp-degree 2
    alloc run -- python distributed/train_tp_dp.py  # single-process fallback
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler

from models import MODEL_CONFIGS, GPU_ONLY_MODELS, build_model, make_synthetic_data


# ---------------------------------------------------------------------------
# Process group setup
# ---------------------------------------------------------------------------


def setup_process_group() -> tuple[int, int]:
    """Initialize process group. Returns (rank, world_size)."""
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
    parser.add_argument("--num-gpus", type=int, default=4, help="Number of GPUs (informational)")
    parser.add_argument("--tp-degree", type=int, default=0, help="TP degree (0 = auto: min(world_size, 2))")
    args = parser.parse_args()

    if args.model in GPU_ONLY_MODELS and not torch.cuda.is_available():
        print(f"SKIP: model '{args.model}' requires GPU")
        sys.exit(0)

    rank, world_size = setup_process_group()
    torch.manual_seed(args.seed + rank)

    device = torch.device("cuda", rank) if torch.cuda.is_available() else torch.device("cpu")

    # Compute TP and DP degrees
    tp_degree = args.tp_degree if args.tp_degree > 0 else min(world_size, 2)
    dp_degree = world_size // tp_degree if tp_degree > 0 else 1

    if rank == 0:
        print(f"Strategy: TP+DP | World size: {world_size} | TP degree: {tp_degree} | DP degree: {dp_degree} | Device: {device}")

    # Create TP and DP sub-groups when distributed
    tp_group = None
    dp_group = None
    if world_size > 1 and tp_degree > 1:
        # TP group: consecutive ranks (e.g., [0,1], [2,3] for tp_degree=2)
        for i in range(0, world_size, tp_degree):
            tp_ranks = list(range(i, min(i + tp_degree, world_size)))
            group = dist.new_group(tp_ranks)
            if rank in tp_ranks:
                tp_group = group

        # DP group: same position within each TP group (e.g., [0,2], [1,3] for tp_degree=2)
        for i in range(tp_degree):
            dp_ranks = list(range(i, world_size, tp_degree))
            group = dist.new_group(dp_ranks)
            if rank in dp_ranks:
                dp_group = group

    cfg = MODEL_CONFIGS[args.model]
    model = build_model(args.model).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"Model: {args.model} (SyntheticTransformer) | Params: {param_count:,}")

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

            # TP gradient sync within TP group
            if tp_group is not None:
                for param in model.parameters():
                    if param.grad is not None:
                        dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, group=tp_group)

            # DP gradient sync within DP group
            if dp_group is not None:
                for param in model.parameters():
                    if param.grad is not None:
                        dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, group=dp_group)

            optimizer.step()

            step += 1
            if rank == 0 and step % 2 == 0:
                print(f"Step {step}/{args.max_steps} | Loss: {loss.item():.4f}")

    if rank == 0:
        print(f"TP+DP training complete. {step} steps, tp_degree={tp_degree}, dp_degree={dp_degree}.")

    cleanup()


if __name__ == "__main__":
    main()
