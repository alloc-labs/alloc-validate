# Ray

Ray Train workload using `TorchTrainer` — validates that `alloc run` correctly profiles Ray Train scripts.

## Status

**Active** — validates `alloc run` wraps a Ray Train `TorchTrainer` script and produces a correct artifact. No Ray-specific callback (Ray doesn't have one); the value is proving the generic `alloc run` profiling works on Ray workers.

## What It Tests

- `alloc run` wraps a Ray Train script and produces a correct artifact
- Single-worker `TorchTrainer` runs without requiring a cluster
- `alloc ghost` static VRAM analysis on the training script

## Model Variants

| Name | Architecture | Params |
|------|-------------|--------|
| `small-cnn` (default) | 2 conv + 2 dense | ~26K |
| `medium-cnn` | 4 conv layers | ~200K |
| `mlp` | 3-layer MLP | ~50K |

## Usage

```bash
# Via make
make ray

# Direct
alloc run -- python ray/train.py
alloc run -- python ray/train.py --model medium-cnn --max-steps 50
```

## CLI Args

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `small-cnn` | Model architecture |
| `--max-steps` | `100` | Training steps |
| `--batch-size` | `64` | Batch size |
| `--seed` | `42` | Random seed |
| `--num-gpus` | `1` | GPU count (metadata only) |
