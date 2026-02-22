#!/usr/bin/env python3
"""Case study: HuggingFace 7B fine-tuning — BEFORE optimization.

Realistic fine-tuning script with common performance mistakes that
alloc diagnose catches. Represents what most users ship on day one.

Deliberate issues:
  - DataLoader: num_workers=2, no pin_memory, no persistent_workers
  - Precision: fp32 (no mixed precision on Ampere+)
  - Distribution: DataParallel instead of DDP on multi-GPU
  - No torch.compile
  - No gradient checkpointing despite large model
  - Small batch size wastes GPU compute
  - cudnn.benchmark not set

Usage:
    alloc diagnose case-study/before.py
    alloc ghost case-study/before.py --param-count-b 7.0
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

import alloc


# ── Config ──────────────────────────────────────────────────────────────

MODEL_NAME = "meta-llama/Llama-2-7b-hf"
BATCH_SIZE = 4
MAX_STEPS = 1000
LR = 2e-5
SEQ_LEN = 512


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
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)

    # Multi-GPU: use DataParallel (suboptimal — should use DDP)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.cuda()

    dataset = TextDataset(tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    step = 0
    for epoch in range(10):
        for batch in dataloader:
            input_ids = batch["input_ids"].cuda()
            labels = batch["labels"].cuda()

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            if isinstance(loss, tuple):
                loss = loss[0]

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            step += 1
            if step % 100 == 0:
                print(f"Step {step}, Loss: {loss.item():.4f}")
            if step >= MAX_STEPS:
                break
        if step >= MAX_STEPS:
            break

    print("Training complete.")


if __name__ == "__main__":
    main()
