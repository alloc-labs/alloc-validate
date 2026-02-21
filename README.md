# alloc-validate

Sample ML training scripts you can use to try [alloc](https://pypi.org/project/alloc/) — training pipeline intelligence for ML engineers.

Every script uses synthetic data, runs on CPU, and finishes in under a minute. No downloads, no datasets, no GPU required.

## Quickstart

```bash
git clone https://github.com/alloc-labs/alloc-validate.git
cd alloc-validate
python3 bootstrap.py          # creates venv, installs deps
source .venv/bin/activate
alloc run -- python pytorch/train.py
```

The bootstrap script auto-detects Python 3.9+ (even if `python3` points to an older version). If no suitable Python is found, use Docker instead:

```bash
make docker-build && make docker-test
```

The `alloc run` command wraps a training script and produces an `alloc_artifact.json.gz` containing GPU metrics, hardware context, and VRAM usage.

**Without login**, results are local only — you'll see a summary in the terminal and an artifact file on disk. To see your runs on the [Alloc dashboard](https://www.alloclabs.com) with analysis, right-sizing suggestions, and cost savings, see [Authentication & Dashboard](#authentication--dashboard).

## What You Can Try

### Wrap a training script (`alloc run`)

`alloc run` monitors any training script and produces an artifact with GPU metrics — without modifying your code. It auto-stops once metrics stabilize (typically 20-60 seconds), then terminates the training process.

Pick a framework and run it:

```bash
# PyTorch
alloc run -- python pytorch/train.py --model large-cnn

# HuggingFace (with step-timing callback)
alloc run -- python huggingface/train.py --model gpt2-tiny

# Lightning (with step-timing callback)
alloc run -- python lightning/train.py --model medium-cnn

# Ray Train
alloc run -- python ray/train.py --model mlp
```

Each produces `alloc_artifact.json.gz` — a compressed JSON with peak VRAM, GPU utilization, power draw, hardware info, and environment context.

Use `--full` to monitor the entire training run instead of auto-stopping after calibration.

Add `--verbose` to see a detailed breakdown, or `--json` for machine-readable output:

```bash
alloc run --verbose -- python pytorch/train.py
alloc run --json -- python pytorch/train.py
```

### Estimate VRAM without training (`alloc ghost`)

Point `alloc ghost` at any Python file that defines a model. It extracts the architecture and estimates VRAM for weights, gradients, optimizer state, and activations — without running training.

```bash
alloc ghost scan-only/ghost_target.py --json
alloc ghost pytorch/train.py --verbose
```

### Compare GPU configurations (`alloc scan`)

Query which GPUs fit your workload, with cost and strategy feasibility. Requires authentication (see below).

```bash
alloc scan --model llama-3-8b --gpu A100-80GB --num-gpus 4
```

### Framework callbacks

The HuggingFace and Lightning training scripts include `alloc` callbacks that capture step-level timing (p50/p90), throughput (samples/sec), and dataloader wait percentage. This data is written to a `.alloc_callback.json` sidecar and merged into the artifact.

```bash
# HuggingFace — uses alloc.HuggingFaceCallback()
alloc run -- python huggingface/train.py

# Lightning — uses alloc.LightningCallback()
alloc run -- python lightning/train.py
```

### Distributed training

The `distributed/` directory has examples for 8 parallelism strategies. Each script sets topology metadata (strategy, degrees, interconnect) that `alloc` detects automatically.

```bash
# Data parallel (DDP)
alloc run -- torchrun --nproc_per_node=1 distributed/train_ddp.py --model small

# Fully sharded data parallel (FSDP)
alloc run -- torchrun --nproc_per_node=1 distributed/train_fsdp.py --model medium

# Pipeline parallelism
alloc run -- torchrun --nproc_per_node=1 distributed/train_pp.py --model large

# Tensor parallelism
alloc run -- torchrun --nproc_per_node=1 distributed/train_tp.py --model small
```

Hybrid strategies are also available: `train_tp_dp.py`, `train_pp_dp.py`, `train_3d.py` (TP+PP+DP), and `train_3d_fsdp.py` (TP+PP+FSDP).

## Available Workloads

All scripts accept `--model`, `--max-steps`, `--batch-size`, and `--seed` flags.

| Framework | Script | Models |
|-----------|--------|--------|
| PyTorch | `pytorch/train.py` | `small-cnn` (default), `medium-cnn`, `large-cnn`, `mlp-small`, `mlp-large` |
| HuggingFace | `huggingface/train.py` | `distilbert-tiny` (default), `distilbert-small`, `gpt2-tiny`, `bert-tiny` |
| Lightning | `lightning/train.py` | `small-cnn` (default), `medium-cnn`, `mlp` |
| Ray | `ray/train.py` | `small-cnn` (default), `medium-cnn`, `mlp` |
| Distributed | `distributed/train_*.py` | `small` (default), `medium`, `large`, `xl` |
| Ghost targets | `scan-only/ghost_target*.py` | 6 standalone model files for `alloc ghost` |

## Authentication & Dashboard

Everything above works without authentication. To see your runs on the dashboard with analysis, right-sizing suggestions, and auto-proposals, log in:

```bash
alloc login --browser    # opens browser for Google/Microsoft sign-in
alloc run -- python pytorch/train.py   # auto-uploads when logged in
```

Once logged in, every `alloc run` automatically uploads the artifact. Your runs appear at [alloclabs.com](https://www.alloclabs.com) with GPU utilization, VRAM breakdown, bottleneck diagnosis, right-sizing recommendations, and cost analysis.

Use `--no-upload` to skip auto-upload for a specific run, or `alloc upload <artifact>` to upload a previous artifact manually.

For CI or non-interactive environments, set the token as an environment variable:

```bash
export ALLOC_TOKEN=<your-token>
```

## GPU Testing

CPU tests validate the artifact contract. GPU tests validate real VRAM measurements, bottleneck classification, and timing metrics.

```bash
# GCP
bash scripts/gpu/launch-gcp-l4.sh       # 1x L4 (~$0.21/hr spot)
bash scripts/gpu/launch-gcp-4xl4.sh     # 4x L4 (~$0.84/hr spot)

# AWS
bash scripts/gpu/launch-aws-t4.sh       # 1x T4 (~$0.16/hr spot)
bash scripts/gpu/launch-aws-4xt4.sh     # 4x T4 (~$1.17/hr spot)
```

See [GPU_TESTING.md](GPU_TESTING.md) for instance recommendations, cost estimates, and multi-GPU topology testing.

<details>
<summary><h2>Contributing</h2></summary>

### Build & Install

```bash
pip install -e ".[all]"           # all framework deps
pip install -e ".[dev]"           # linting/testing tools
ruff check .
mypy scripts/
```

### CI Tiers

**PR CI (CPU-only):** `make validate-free` runs all workloads without `ALLOC_TOKEN`. Validates artifact contract (top-level keys, nested paths, types, non-null checks).

**Nightly (GPU):** `make validate-full` runs on GPU with `ALLOC_TOKEN`. Validates real GPU metrics, ingest, analysis, and bottleneck classification.

### Make Targets Reference

| Target | Description |
|--------|-------------|
| `make validate-free` | All workloads, no API key |
| `make validate-full` | All workloads, requires `ALLOC_TOKEN` |
| `make pytorch` | PyTorch workload only |
| `make huggingface` | HuggingFace workload only |
| `make scan-only` | Ghost + scan tests only |
| `make lightning` | Lightning workload only |
| `make ray` | Ray workload only |
| `make distributed` | Distributed (DDP smoke test) |
| `make validate-topology` | All 8 distributed strategies |
| `make validate-fleet` | Fleet context (`.alloc.yaml`) validation |
| `make validate-upload` | Upload + analysis pipeline (requires `ALLOC_TOKEN`) |
| `make matrix` | Full model/GPU variation matrix |
| `make matrix-quick` | 1 model per framework (smoke test) |
| `make docker-build` | Build Docker image |
| `make docker-test` | Run free-tier validation in Docker |
| `make clean` | Remove generated artifacts |
| `make setup` | Create venv + install deps (`bootstrap.py`) |

### Project Structure

```
alloc-validate/
├── pyproject.toml              # Dependencies (extras per workload)
├── Makefile                    # Task orchestration
├── Dockerfile                  # Reproducible CI/local runs
├── scripts/
│   ├── check_artifact.py       # Shared artifact validator
│   ├── run_matrix.py           # Model/GPU variation matrix runner
│   ├── validate_fleet.sh       # .alloc.yaml + catalog validation
│   ├── validate_upload.sh      # Upload + analysis pipeline
│   ├── validate_topology.sh    # Distributed topology validation
│   └── gpu/                    # GPU instance launch scripts (AWS + GCP)
├── pytorch/                    # 5 model variants
├── huggingface/                # 4 model variants (HF Callback)
├── lightning/                  # 3 model variants (Lightning Callback)
├── ray/                        # 3 model variants (Ray Train)
├── distributed/                # 8 strategies, 4+ model sizes
└── scan-only/                  # 6 ghost targets + scan combos
```

Each workload directory contains `train.py`, `validate.sh`, and `expected/schema.json`.

### Adding a Workload

1. Create a directory with `train.py`, `expected/schema.json`, `validate.sh`
2. Add a dependency extra to `pyproject.toml`
3. Add a `make` target to `Makefile`
4. Add to the CI workflow matrix

</details>
