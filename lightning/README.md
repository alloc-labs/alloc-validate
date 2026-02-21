# Lightning

PyTorch Lightning training with `alloc.LightningCallback()` integration.

## Status

**Active** — validates `alloc.LightningCallback()` (shipped in alloc v0.4.0+) hooks into the Lightning Trainer loop correctly and that step timing data appears in the callback sidecar.

## What It Tests

- `alloc run` wraps a Lightning Trainer script and produces a correct artifact
- `alloc.LightningCallback()` captures step timing (p50/p90), throughput (samples/sec), dataloader wait %
- Callback writes `.alloc_callback.json` sidecar with timing data
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
make lightning

# Direct
alloc run -- python lightning/train.py
alloc run -- python lightning/train.py --model medium-cnn --max-steps 50
```

## CLI Args

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `small-cnn` | Model architecture |
| `--max-steps` | `100` | Training steps |
| `--batch-size` | `64` | Batch size |
| `--seed` | `42` | Random seed |
| `--num-gpus` | `1` | GPU count (metadata only) |
