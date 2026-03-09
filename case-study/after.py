#!/usr/bin/env python3
"""Case study: HuggingFace 7B fine-tuning — AFTER optimization.

Same training task as before.py with all alloc diagnose recommendations
applied. Every change maps to a specific rule ID.

Fixes applied:
  - DL001: num_workers=8 (was 2)
  - DL002: pin_memory=True (was missing)
  - DL003: persistent_workers=True (was missing)
  - DL004: prefetch_factor=4 (was default 2)
  - DIST005: DDP via torchrun (was DataParallel)
  - MEM002: bf16 autocast (was fp32)
  - MEM005: torch.compile (was missing)
  - PREC002: bf16 on Ampere+ (was fp32)
  - THRU001: cudnn.benchmark=True (was missing)
  - UPG002: torch.set_float32_matmul_precision("high") for TF32-capable GPUs
  - Gradient checkpointing enabled for 7B model

Usage:
    alloc diagnose case-study/after.py
    torchrun --nproc_per_node=4 case-study/after.py
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

import alloc


# ── Config ──────────────────────────────────────────────────────────────

MODEL_NAME = "meta-llama/Llama-2-7b-hf"
BATCH_SIZE = 16          # 4x larger — bf16 frees VRAM headroom
MAX_STEPS = 1000
LR = 2e-5
SEQ_LEN = 512

# THRU001: enable cudnn autotuner
torch.backends.cudnn.benchmark = True
# UPG002: enable TF32 matmul precision where supported
torch.set_float32_matmul_precision("high")


# ── Dataset ─────────────────────────────────────────────────────────────

class TextDataset(Dataset):
    def __init__(self, tokenizer, n_samples=10000, seq_len=SEQ_LEN):
        self.data = torch.randint(0, tokenizer.vocab_size, (n_samples, seq_len))
        self.labels = self.data.clone()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"input_ids": self.data[idx], "labels": self.labels[idx]}


# ── Training ────────────────────────────────────────────────────────────

def main():
    # DIST005: proper DDP initialization (launch via torchrun)
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # PREC002 / MEM002: load in bf16 instead of fp32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
    )

    # Enable gradient checkpointing — trades compute for memory on 7B model
    model.gradient_checkpointing_enable()

    model = model.cuda(local_rank)

    # MEM005: torch.compile for kernel fusion
    model = torch.compile(model)

    # DIST005: DistributedDataParallel instead of DataParallel
    model = DDP(model, device_ids=[local_rank])

    dataset = TextDataset(tokenizer)
    sampler = DistributedSampler(dataset)

    # DL001-004: optimized DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=8,              # DL001: match CPU cores
        pin_memory=True,            # DL002: async H2D transfer
        persistent_workers=True,    # DL003: skip fork overhead
        prefetch_factor=4,          # DL004: keep GPU fed
        drop_last=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # MEM002 / PREC002: bf16 autocast for forward/backward
    scaler = None  # bf16 doesn't need GradScaler

    model.train()
    step = 0
    for epoch in range(10):
        sampler.set_epoch(epoch)
        for batch in dataloader:
            input_ids = batch["input_ids"].cuda(local_rank, non_blocking=True)
            labels = batch["labels"].cuda(local_rank, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            step += 1
            if step % 100 == 0 and local_rank == 0:
                print(f"Step {step}, Loss: {loss.item():.4f}")
            if step >= MAX_STEPS:
                break
        if step >= MAX_STEPS:
            break

    if local_rank == 0:
        print("Training complete.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
