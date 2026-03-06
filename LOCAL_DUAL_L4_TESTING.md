# Local Dual-L4 Test Runbook (Work Laptop, No Cloud)

Purpose: validate your platform end-to-end on a local workstation with 2x L4 GPUs before spending cloud credits.

This path is local-first and cost-effective:
- no VM provisioning
- no cloud launch scripts
- full alloc CLI + artifact + optional upload coverage

---

## 1) Preconditions

On the workstation:

```bash
cd /path/to/alloc-validate
python3 bootstrap.py
source .venv/bin/activate
```

If `python3` points to an old/broken interpreter on your machine:

```bash
python3 bootstrap.py --python python3.10
```

Sanity checks:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

Expected:
- `nvidia-smi` shows two L4 GPUs
- torch reports `True 2` (or higher)

---

## 2) Optional CLI Auth (Only Needed For Upload)

`alloc run` itself does not require login.

If you want dashboard ingest/analysis during this local test:

```bash
alloc login --browser
```

Current CLI behavior (from alloc-labs audit):
- `alloc run` is privacy-first and does not upload by default
- upload is explicit via `--upload` or `alloc upload <artifact>`
- `alloc scan` works without login via `/scans/cli`

---

## 3) Run the New Dual-L4 Stress Test

```bash
make dual-l4-stress
```

This target runs `scripts/local_dual_l4_stress.sh` and executes six lanes:

0. Pre-flight sizing:
- command: `alloc ghost` baseline + optional `alloc scan` recommendation capture
- expectation: ghost JSON always exists; scan JSON exists when network/API is reachable

1. Calibrate DDP run (default mode):
- command: DDP medium model with `torchrun --nproc_per_node=2`
- expectation: run auto-stops after metrics stabilize and writes artifact

2. Full DDP stress run:
- command: DDP large model, full monitoring
- expectation: full run completes with non-zero GPU metrics

3. Full FSDP stress run:
- command: FSDP medium model, full monitoring
- expectation: full run completes and artifact passes distributed schema

4. HF callback path run:
- command: HuggingFace gpt2-tiny, full monitoring
- expectation: callback timing fields are present in artifact

5. High-pressure lane:
- command: DDP xl model with aggressive batch size
- expectation: either non-zero exit from memory pressure (expected) OR successful run with >=85% peak VRAM

Artifacts are written under:
- `baselines/dual-l4-local-<timestamp>/`

---

## 4) Expected Output Signals

Successful run should show:
- `Detected CUDA GPUs: 2`
- ghost pre-flight JSON exists
- all required baseline `alloc run` commands complete without crash
- `scripts/check_artifact.py` succeeds for each generated artifact
- pressure lane either:
  - exits non-zero with expected high-memory failure, or
  - completes and reports high peak VRAM
- optional upload step:
  - `SKIP: ALLOC_TOKEN not set` (no auth case), or
  - upload success when token is set

---

## 5) Manual Validation Checklist

After test completion:

1. Confirm artifact files exist in the output folder.
2. Spot check one artifact for non-zero GPU usage/VRAM.
3. If authenticated, confirm uploaded runs appear in dashboard.
4. Confirm recommendation/bottleneck sections render for uploaded run(s).

---

## 6) Troubleshooting

If script fails with GPU count < 2:
- check Docker/WSL/driver visibility
- re-run `nvidia-smi` and torch CUDA device count check

If `torchrun` is not functional:
- reinstall deps: `.venv/bin/pip install -e '.[all]'`

If upload fails:
- confirm login/token (`alloc whoami --json`)
- retry with explicit upload:

```bash
alloc upload baselines/dual-l4-local-<timestamp>/ddp_full_large.json.gz
```
