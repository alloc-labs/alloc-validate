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

Bootstrap behavior is deterministic and repeatable across machines:
- auto-selects a working Python 3.9+ interpreter with SSL support
- creates `.venv` with stdlib `venv` and falls back to `virtualenv` if `ensurepip` is unavailable
- installs `alloc-validate` dependencies in editable mode

If you need a specific interpreter (recommended for workstations/CI), pin it explicitly:

```bash
python3 bootstrap.py --python python3.10
# or
ALLOC_VALIDATE_PYTHON=python3.10 python3 bootstrap.py
```

If no suitable Python is found, use Docker instead:

```bash
make docker-build && make docker-test
```

The `alloc run` command wraps a training script and produces an `alloc_artifact.json.gz` containing GPU metrics, hardware context, and VRAM usage.

**Without login**, results are local only — you'll see a summary in the terminal and an artifact file on disk. To see your runs on the [Alloc dashboard](https://www.alloclabs.com) with analysis, right-sizing suggestions, and cost savings, see [Authentication & Dashboard](#authentication--dashboard).

For private founder notes or internal context, use `.private/` (git-ignored by default).

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

Query which GPUs fit your workload, with cost and strategy feasibility. Works without login (`/scans/cli` path); login unlocks org-aware and Pro-gated enrichments.

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

## Running on a GPU

The quickstart commands work on CPU and produce a valid artifact, but GPU metrics (peak VRAM, GPU utilization, power draw) will be zeros or absent. To get real measurements, run on a machine with an NVIDIA GPU.

### What to expect

| Environment | What you get |
|---|---|
| **CPU only** | Valid artifact structure, but GPU metrics are zeros. Good for trying the CLI. |
| **GPU** | Real peak VRAM (MB), GPU utilization (%), power draw (W), hardware detection. |
| **GPU + callbacks** (HF/Lightning) | All of the above, plus step timing p50/p90, throughput (samples/sec), dataloader wait %. |
| **GPU + login** | All of the above. Add `--upload` (or `ALLOC_UPLOAD=1`) to send artifacts to the [Alloc dashboard](https://www.alloclabs.com) for analysis and right-sizing. |

### Local NVIDIA GPU

If you have a local GPU (RTX 3090, 4090, A6000, etc.), the exact same quickstart commands produce real metrics automatically — no extra setup:

```bash
source .venv/bin/activate
alloc run -- python pytorch/train.py --model large-cnn    # real VRAM + utilization
alloc run -- python huggingface/train.py --model gpt2-tiny # adds step timing via callback
```

Verify your GPU is detected:

```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Local Dual-L4 Workstation (Recommended Before Cloud)

If your machine already has 2x L4 GPUs, run the local stress lane first to validate alloc end-to-end with zero cloud spend:

```bash
make dual-l4-stress
```

This runs calibrated + full monitoring paths across DDP/FSDP and callback workloads using `alloc` CLI, then validates artifact schemas.
It also includes a high-pressure lane (expected OOM or very high VRAM) to validate low-memory behavior.

See [LOCAL_DUAL_L4_TESTING.md](LOCAL_DUAL_L4_TESTING.md) for the full runbook and expected outcomes.

### Cloud GPU via GCP

If you don't have a local GPU, the fastest path is a GCP spot instance. A full test cycle takes under 10 minutes and costs ~$0.04.

**Step 1 — Set up GCP (one-time):**

```bash
make setup-gcp
```

This interactive wizard installs the `gcloud` CLI if needed, authenticates, creates or selects a project, enables the Compute Engine API, and checks GPU quota. It's idempotent — re-run anytime to verify your setup.

**Step 2 — Launch a spot GPU instance:**

```bash
bash scripts/gpu/launch-gcp-l4.sh       # 1x L4 (24 GB VRAM, ~$0.21/hr spot)
```

The instance clones this repo, installs dependencies, runs the validation suite, and **auto-deletes** when done.

Security note: launch scripts do not inject `ALLOC_TOKEN` into cloud startup metadata. They run free-tier validation by default; run `make validate-full` manually after SSH if needed.

**Step 3 — Check progress:**

```bash
gcloud compute ssh alloc-validate-l4 --zone=us-central1-a \
  --command 'tail -f /var/log/alloc-validate.log'
```

To delete the instance early: `gcloud compute instances delete alloc-validate-l4 --zone=us-central1-a --quiet`

### AWS

AWS GPU instances are also supported:

```bash
bash scripts/gpu/launch-aws-t4.sh       # 1x T4 (16 GB VRAM, ~$0.16/hr spot)
```

### Multi-GPU and advanced testing

For distributed strategy validation (DDP, FSDP, TP, PP) and cost tables across instance types, see [GPU_TESTING.md](GPU_TESTING.md).

## Authentication & Dashboard

Everything above works without authentication and keeps results local.
To send runs to dashboard analysis, authenticate and upload explicitly:

```bash
alloc login --browser
alloc run --upload -- python pytorch/train.py
```

`alloc run` is privacy-first and does not upload unless `--upload` (or `ALLOC_UPLOAD=1`) is set.

For the step-by-step GPU test plan with expected outcomes, follow [GCP_ONBOARDING.md](GCP_ONBOARDING.md).

To upload an existing artifact manually:

```bash
alloc upload pytorch/alloc_artifact.json.gz
```

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

**Nightly (GPU):** `make validate-full` runs authenticated GPU workloads with `ALLOC_TOKEN` and validates runtime behavior.

For explicit upload + ingest checks, run `make validate-upload` in the same session.

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
| `make dual-l4-stress` | Local workstation stress test for dual-L4 GPUs |
| `make docker-build` | Build Docker image |
| `make docker-test` | Run free-tier validation in Docker |
| `make test` | Run full pytest suite |
| `make test-diagnose` | Diagnose tests only |
| `make test-callbacks` | Callback tests only |
| `make test-quick` | Fast tests (no training): CLI, diagnose, ghost, schema |
| `make clean` | Remove generated artifacts |
| `make setup` | Create venv + install deps (`bootstrap.py`) |

### Project Structure

```
alloc-validate/
├── pyproject.toml              # Dependencies (extras per workload)
├── Makefile                    # Task orchestration
├── Dockerfile                  # Reproducible CI/local runs
├── tests/                      # pytest suite (new)
│   ├── conftest.py             # Shared fixtures
│   ├── test_cli_smoke.py       # version, whoami, catalog
│   ├── test_diagnose.py        # alloc diagnose on all targets
│   ├── test_ghost.py           # alloc ghost on scan-only targets
│   ├── test_run.py             # alloc run on each framework
│   ├── test_callbacks.py       # Callback-only artifact generation
│   ├── test_scan.py            # alloc scan (unauthenticated + token-enhanced paths)
│   ├── test_artifact_contract.py  # Schema validation
│   └── test_config.py          # .alloc.yaml + catalog
├── diagnose-targets/           # Scripts designed to trigger specific rules (new)
│   ├── dl_issues.py            # Triggers DL001-DL004
│   ├── precision_issues.py     # Triggers PREC001
│   ├── memory_issues.py        # Triggers MEM002, MEM005
│   ├── dist_issues.py          # Triggers DIST005
│   ├── clean_script.py         # Should produce ZERO findings
│   └── hf_trainer_issues.py    # HF Trainer without optimizations
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
