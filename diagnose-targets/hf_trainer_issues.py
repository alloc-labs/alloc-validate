#!/usr/bin/env python3
"""HF Trainer script without bf16 or torch_compile.

Triggers: MEM002 (no mixed precision), MEM005 (no torch.compile)

Uses HuggingFace Trainer with TrainingArguments but omits performance settings.
NOTE: Uses .cuda() for AST detection. Diagnose target only.
cudnn.benchmark set so THRU001 doesn't suppress MEM002/MEM005.
"""

from __future__ import annotations

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

torch.backends.cudnn.benchmark = True


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    config = BertConfig(
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        num_labels=2,
    )
    model = BertForSequenceClassification(config)
    model = model.cuda()

    rng = np.random.default_rng(42)
    dataset = Dataset.from_dict({
        "input_ids": rng.integers(0, 1000, size=(200, 32)).tolist(),
        "attention_mask": np.ones((200, 32), dtype=int).tolist(),
        "labels": rng.integers(0, 2, size=200).tolist(),
    })

    # No bf16=True, no torch_compile=True
    training_args = TrainingArguments(
        output_dir="./hf_output",
        max_steps=10,
        per_device_train_batch_size=16,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
    print("Done.")


if __name__ == "__main__":
    main()
