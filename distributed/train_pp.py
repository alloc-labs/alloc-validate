#!/usr/bin/env python3
"""Pipeline Parallelism (PP) training on synthetic transformer data.

Trains a configurable synthetic transformer using manual pipeline parallelism.
Splits transformer layers across devices and runs micro-batches through the
pipeline. Falls back to single-device sequential execution on CPU.

Fully deterministic (seeded), no downloads, completes in < 60s on CPU.

Signature scenario: PP topology — validates that `alloc run` detects pipeline
parallelism strategy and reports pp_degree in the artifact.

Usage:
    alloc run -- torchrun --nproc_per_node=4 distributed/train_pp.py
    alloc run -- python distributed/train_pp.py  # single-device fallback
    alloc run -- python distributed/train_pp.py --model large --max-steps 3
"""

from __future__ import annotations

import argparse
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models import MODEL_CONFIGS, GPU_ONLY_MODELS, build_pipelined_model, make_synthetic_data


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
    parser.add_argument("--num-gpus", type=int, default=4, help="Number of GPUs / pipeline stages")
    parser.add_argument("--num-stages", type=int, default=0, help="Pipeline stages (0 = auto)")
    args = parser.parse_args()

    if args.model in GPU_ONLY_MODELS and not torch.cuda.is_available():
        print(f"SKIP: model '{args.model}' requires GPU")
        sys.exit(0)

    torch.manual_seed(args.seed)

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    num_stages = args.num_stages if args.num_stages > 0 else max(2, num_gpus)
    device = torch.device("cuda:0") if num_gpus > 0 else torch.device("cpu")

    print(f"Strategy: PP | Stages: {num_stages} | GPUs available: {num_gpus} | Device: {device}")

    cfg = MODEL_CONFIGS[args.model]
    model = build_pipelined_model(args.model, num_stages=num_stages)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model} (PipelinedTransformer) | Params: {param_count:,}")
    print(f"Pipeline stages: {model.num_stages}")

    if num_gpus > 1:
        # Assign stages to GPUs round-robin
        devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]
        model.assign_devices(devices)
        print(f"Stages assigned across {num_gpus} GPUs")
    else:
        model.to(device)

    dataset = make_synthetic_data(cfg["d_model"])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    step = 0
    model.train()
    while step < args.max_steps:
        for inputs, targets in loader:
            if step >= args.max_steps:
                break
            # For multi-GPU PP, tensors flow through stages on different devices
            # Input goes to first device, output comes from last device
            if num_gpus > 1:
                input_dev = next(model.stages[0].parameters()).device
                output_dev = next(model.head.parameters()).device
                inputs = inputs.to(input_dev)
                targets = targets.to(output_dev)
            else:
                inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            # Forward through stages (data moves between devices)
            x = inputs
            for stage in model.stages:
                stage_dev = next(stage.parameters()).device
                x = x.to(stage_dev)
                x = stage(x)

            head_dev = next(model.head.parameters()).device
            x = x.to(head_dev)
            pooled = x.mean(dim=1)
            outputs = model.head(pooled)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            step += 1
            if step % 2 == 0:
                print(f"Step {step}/{args.max_steps} | Loss: {loss.item():.4f}")

    print(f"PP training complete. {step} steps, {num_stages} stages.")


if __name__ == "__main__":
    main()
