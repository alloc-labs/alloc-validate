#!/usr/bin/env python3
"""HuggingFace Trainer with alloc callback.

Uses tiny in-code model configurations (no model download required).
Trains a small text classifier on synthetic data to validate
the alloc HuggingFace integration.

Fully deterministic (seeded), no downloads, completes in < 60s on CPU.

Signature scenario: callback integration — validates that
`alloc.HuggingFaceCallback()` correctly hooks into the Trainer loop
and that callback events appear in the artifact.

Usage:
    alloc run python huggingface/train.py
    alloc run python huggingface/train.py --model gpt2-tiny --max-steps 50
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    GPT2Config,
    GPT2ForSequenceClassification,
    Trainer,
    TrainingArguments,
)

import alloc

try:
    _hf_callback_cls = alloc.HuggingFaceCallback  # type: ignore[attr-defined]
except AttributeError:
    _hf_callback_cls = None


def _build_distilbert_tiny() -> tuple:
    config = DistilBertConfig(
        vocab_size=1000,
        dim=64,
        n_layers=2,
        n_heads=2,
        hidden_dim=128,
        num_labels=2,
    )
    return config, DistilBertForSequenceClassification(config)


def _build_distilbert_small() -> tuple:
    config = DistilBertConfig(
        vocab_size=1000,
        dim=128,
        n_layers=4,
        n_heads=4,
        hidden_dim=256,
        num_labels=2,
    )
    return config, DistilBertForSequenceClassification(config)


def _build_gpt2_tiny() -> tuple:
    config = GPT2Config(
        vocab_size=1000,
        n_embd=64,
        n_layer=2,
        n_head=2,
        n_inner=128,
        num_labels=2,
    )
    config.pad_token_id = config.eos_token_id
    return config, GPT2ForSequenceClassification(config)


def _build_bert_tiny() -> tuple:
    config = BertConfig(
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        num_labels=2,
    )
    return config, BertForSequenceClassification(config)


MODEL_REGISTRY: dict[str, callable] = {
    "distilbert-tiny": _build_distilbert_tiny,
    "distilbert-small": _build_distilbert_small,
    "gpt2-tiny": _build_gpt2_tiny,
    "bert-tiny": _build_bert_tiny,
}


def make_synthetic_dataset(
    n_samples: int = 500, seq_len: int = 32, seed: int = 42,
) -> Dataset:
    """Create a synthetic dataset of random token IDs and binary labels."""
    rng = np.random.default_rng(seed)
    return Dataset.from_dict({
        "input_ids": rng.integers(0, 1000, size=(n_samples, seq_len)).tolist(),
        "attention_mask": np.ones((n_samples, seq_len), dtype=int).tolist(),
        "labels": rng.integers(0, 2, size=n_samples).tolist(),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=100, help="Max training steps")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model",
        choices=list(MODEL_REGISTRY),
        default="distilbert-tiny",
        help="Model architecture to train",
    )
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs (informational)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Num GPUs (metadata): {args.num_gpus}")

    # Build model from registry — no download needed
    builder = MODEL_REGISTRY[args.model]
    config, model = builder()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model} | Params: {param_count:,}")

    dataset = make_synthetic_dataset(seed=args.seed)

    training_args = TrainingArguments(
        output_dir="./hf_output",
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        use_cpu=(device == "cpu"),
        seed=args.seed,
    )

    cbs = [_hf_callback_cls()] if _hf_callback_cls is not None else []
    if not cbs:
        print("WARN: alloc.HuggingFaceCallback not available yet — training without callback")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        callbacks=cbs,
    )

    trainer.train()
    print("Training complete.")


if __name__ == "__main__":
    main()
