# scan-only — Ghost + Remote Scan

Tests `alloc ghost` (static VRAM analysis) and `alloc scan` (remote API scan)
without any training.

## Signature Scenario

**Scan validation** — validates that the analysis tools work independently
of training. Ghost performs static VRAM estimation on a script. Scan queries
the API for model/GPU feasibility and cost.

## Ghost Targets

Six standalone target scripts for `alloc ghost`:

| File | Model | ~Params |
|------|-------|---------|
| `ghost_target.py` | MediumModel MLP (1024→2048→2048→1024→10) | 8M |
| `ghost_target_small_cnn.py` | SmallCNN (2 conv + 2 dense) | 269K |
| `ghost_target_medium_cnn.py` | MediumCNN (4 conv layers) | 200K |
| `ghost_target_large_cnn.py` | LargeCNN (6 conv layers) | 1M |
| `ghost_target_mlp_small.py` | MLPSmall (3-layer MLP) | 50K |
| `ghost_target_mlp_large.py` | MLPLarge (5-layer MLP) | 526K |

Each file is self-contained (model definition + forward/backward pass) since
`alloc ghost` takes a script path and doesn't pass CLI args through.

## Scan Combos (requires authentication)

The validate script loops over:
- **GPUs**: A100-80GB, H100-80GB, T4-16GB, V100-32GB
- **Models**: llama-3-8b, llama-3-70b, mistral-7b
- **num-gpus**: 1, 4

Total: 24 scan combos with 0.5s delay between API calls.

## What this tests

- `alloc ghost script.py` — produces VRAM breakdown (weights, gradients, optimizer, activations, total)
- `alloc scan --model X --gpu Y --num-gpus N` — returns VRAM + strategy feasibility + cost estimate

## Usage

```bash
# Ghost scan (no token needed)
alloc ghost scan-only/ghost_target.py
alloc ghost scan-only/ghost_target_large_cnn.py

# Remote scan (requires authentication — see alloc login)
alloc scan --model llama-3-8b --gpu A100-80GB
alloc scan --model llama-3-70b --gpu H100-80GB --num-gpus 4

# Run all via validate script
cd scan-only && bash validate.sh

# Or via matrix runner
python scripts/run_matrix.py --framework scan-only
```

## Auth

- `alloc ghost` is fully local, no token required
- `alloc scan` calls the Alloc API, requires authentication (`alloc login`)
- `validate.sh` skips remote scan gracefully when no token is set
