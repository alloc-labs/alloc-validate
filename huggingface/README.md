# HuggingFace — Trainer with alloc Callback

HuggingFace Transformers workload using `alloc.HuggingFaceCallback()`.

## Signature Scenario

**Callback integration** — validates that `alloc.HuggingFaceCallback()` hooks
into the HF Trainer loop and that `step_count` appears in the generated artifact.
The callback currently captures `global_step` count only (not step timing or
samples/sec — those require Phase 3g framework timing).

## Model Variants

| Key | Config Source | Layers | Hidden | Heads |
|-----|-------------|--------|--------|-------|
| `distilbert-tiny` (default) | AutoConfig distilbert | 2 | 64 | 2 |
| `distilbert-small` | AutoConfig distilbert | 4 | 128 | 4 |
| `gpt2-tiny` | GPT2Config (no download) | 2 | 64 | 2 |
| `bert-tiny` | BertConfig (no download) | 2 | 64 | 2 |

All models use in-code config constructors — no network calls or model downloads.

## What this tests

- `alloc run` wrapping a HuggingFace Trainer script
- `alloc.HuggingFaceCallback()` integration
- `step_count` field present in artifact
- Tiny in-code model config — no model or dataset download

## Usage

```bash
pip install -e ".[huggingface]"

# Default model (distilbert-tiny)
alloc run -- python huggingface/train.py

# Specific model variant
alloc run -- python huggingface/train.py --model gpt2-tiny --max-steps 30

# All variants via matrix runner
python scripts/run_matrix.py --framework huggingface
```

## Properties

- **Deterministic**: seeded with `--seed 42`
- **No downloads**: tiny in-code model config + synthetic token data
- **Fast**: 100 steps default, < 60s CPU / < 30s GPU
- **Configurable**: `--max-steps`, `--batch-size`, `--seed`, `--model`, `--num-gpus`
