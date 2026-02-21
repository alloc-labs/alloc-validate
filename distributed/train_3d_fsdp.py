#!/usr/bin/env python3
"""3D Parallelism with FSDP (TP+PP+FSDP) training on synthetic transformer data.

Same as 3D parallelism (TP+PP+DP) but replaces the DP layer with FSDP wrapping
for memory-efficient data parallelism. On CPU, falls back to single-process
sequential execution (FSDP wrapping skipped).

Fully deterministic (seeded), no downloads, completes in < 60s on CPU.

Signature scenario: TP+PP+FSDP topology — validates that `alloc run` detects
the combined strategy and reports tp_degree, pp_degree in the artifact.

Usage:
    alloc run -- torchrun --nproc_per_node=8 distributed/train_3d_fsdp.py
    alloc run -- torchrun --nproc_per_node=8 distributed/train_3d_fsdp.py --tp-degree 2 --pp-degree 2
    alloc run -- python distributed/train_3d_fsdp.py  # single-process fallback
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

from models import (
    MODEL_CONFIGS,
    GPU_ONLY_MODELS,
    TransformerStage,
    build_pipelined_model,
    make_synthetic_data,
)


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
        default="large",
        help="Model size variant",
    )
    parser.add_argument("--num-gpus", type=int, default=8, help="Number of GPUs (informational)")
    parser.add_argument("--tp-degree", type=int, default=2, help="Tensor parallelism degree")
    parser.add_argument("--pp-degree", type=int, default=2, help="Pipeline parallelism degree")
    args = parser.parse_args()

    if args.model in GPU_ONLY_MODELS and not torch.cuda.is_available():
        print(f"SKIP: model '{args.model}' requires GPU")
        sys.exit(0)

    rank, world_size = setup_process_group()
    torch.manual_seed(args.seed + rank)

    device = torch.device("cuda", rank) if torch.cuda.is_available() else torch.device("cpu")

    tp_degree = args.tp_degree
    pp_degree = args.pp_degree
    dp_degree = max(1, world_size // (tp_degree * pp_degree))

    if rank == 0:
        print(f"Strategy: TP+PP+FSDP (3D) | World size: {world_size}")
        print(f"  TP degree: {tp_degree} | PP degree: {pp_degree} | DP/FSDP degree: {dp_degree}")
        print(f"  Device: {device}")

    # Build pipelined model with pp_degree stages
    cfg = MODEL_CONFIGS[args.model]
    model = build_pipelined_model(args.model, num_stages=pp_degree)
    param_count = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"Model: {args.model} (PipelinedTransformer) | Params: {param_count:,}")
        print(f"Pipeline stages: {model.num_stages}")

    model.to(device)

    # FSDP wrapping — requires CUDA
    if torch.cuda.is_available() and (world_size > 1 or dist.is_initialized()):
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={TransformerStage},
        )
        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            device_id=device if device.type == "cuda" else None,
        )
        if rank == 0:
            print("FSDP wrapping applied to pipeline stages")
    elif not torch.cuda.is_available():
        if rank == 0:
            print("WARN: FSDP wrapping skipped (requires CUDA) — running unwrapped on CPU")

    # Create TP sub-groups for gradient sync when distributed
    tp_group = None
    if world_size > 1 and tp_degree > 1:
        for i in range(0, world_size, tp_degree):
            tp_ranks = list(range(i, min(i + tp_degree, world_size)))
            group = dist.new_group(tp_ranks)
            if rank in tp_ranks:
                tp_group = group

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

            # TP gradient sync within TP group (FSDP handles DP sync)
            if tp_group is not None:
                for param in model.parameters():
                    if param.grad is not None:
                        dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, group=tp_group)

            optimizer.step()

            step += 1
            if rank == 0 and step % 2 == 0:
                print(f"Step {step}/{args.max_steps} | Loss: {loss.item():.4f}")

    if rank == 0:
        print(f"3D+FSDP training complete. {step} steps, tp={tp_degree}, pp={pp_degree}, fsdp_degree={dp_degree}.")

    cleanup()


if __name__ == "__main__":
    main()
