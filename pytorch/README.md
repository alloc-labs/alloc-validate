# PyTorch — Baseline Training

Vanilla PyTorch workload: trains configurable models on synthetic 32x32 images.

## Signature Scenario

**Baseline training** — the primary validation workload. Validates the core
`alloc run` → artifact loop and `alloc ghost` static VRAM analysis on a
standard PyTorch training script. No framework abstractions, no callbacks.

## Model Variants

| Key | Architecture | ~Params |
|-----|-------------|---------|
| `small-cnn` (default) | 2 conv + 2 dense | 269K |
| `medium-cnn` | 4 conv layers, wider channels | 200K |
| `large-cnn` | 6 conv layers, widest channels | 1M |
| `mlp-small` | 3-layer MLP (3072→16→16→10) | 50K |
| `mlp-large` | 5-layer MLP (3072→128→256→256→128→10) | 526K |

## What this tests

- `alloc run -- python train.py --model X` — produces `alloc_artifact.json.gz` with peak VRAM, GPU util samples, power draw, duration
- `alloc ghost train.py` — static VRAM breakdown (weights, gradients, optimizer, activations)
- Artifact structure validation against expected keys

## Usage

```bash
pip install -e ".[pytorch]"

# Default model (small-cnn)
alloc run -- python pytorch/train.py

# Specific model variant
alloc run -- python pytorch/train.py --model large-cnn --max-steps 30

# All variants via matrix runner
python scripts/run_matrix.py --framework pytorch
```

## Properties

- **Deterministic**: seeded with `--seed 42`
- **No downloads**: synthetic data generated in-code
- **Fast**: 100 steps default, < 60s CPU / < 30s GPU
- **Configurable**: `--max-steps`, `--batch-size`, `--seed`, `--model`, `--num-gpus`
