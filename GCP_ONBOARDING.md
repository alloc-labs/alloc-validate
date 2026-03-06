# GCP Onboarding + Test-Day Runbook (L4-First, Secure)

Purpose: run a complete, low-cost validation of your platform tomorrow using `alloc-validate` on GCP GPUs.

If you already have local dual-L4 access, run [LOCAL_DUAL_L4_TESTING.md](LOCAL_DUAL_L4_TESTING.md) first and use this guide only when you intentionally move to cloud.

This runbook is designed for:
- founder self-testing
- pre-customer demo confidence checks
- end-to-end platform validation (CLI -> upload -> ingest -> dashboard)

Security model:
- launcher scripts do **not** inject `ALLOC_TOKEN` into startup metadata
- authenticated tests set `ALLOC_TOKEN` only inside an interactive SSH shell

---

## 0) Tomorrow Goals

By end of test day, you should have evidence for all 4 paths:

1. Local contract path: artifacts and CLI behavior are stable.
2. GPU telemetry path: real VRAM/util/power metrics from L4.
3. Upload/ingest path: artifacts reach platform and get analyzed.
4. Multi-GPU path: 4x L4 process discovery and matrix coverage.

---

## 1) Tonight (CPU-Only, 10 Minutes)

From your laptop:

```bash
cd /Users/TZKLHL/Desktop/startup/alloc-validate
python3 bootstrap.py
source .venv/bin/activate
make test-quick
```

If your default `python3` is not suitable on the host:

```bash
python3 bootstrap.py --python python3.10
```

What to expect:
- `test-quick` finishes with all core tests passing (some skips are acceptable).
- No credential prompts required.

Optional preflight sanity:

```bash
alloc version
alloc whoami --json
alloc scan --model llama-3-8b --gpu A100-80GB --json
```

What to expect:
- `alloc version` and `whoami` exit cleanly.
- `alloc scan` works without login on normal internet; if network is restricted, you should still get a clean error (not a crash).

---

## 2) One-Time GCP Setup (Tomorrow Morning)

```bash
cd /Users/TZKLHL/Desktop/startup/alloc-validate
source .venv/bin/activate
make setup-gcp
```

Get project identifiers for quota/support:

```bash
gcloud config get-value project
gcloud projects describe "$(gcloud config get-value project)" --format="value(projectNumber)"
```

What to expect:
- setup summary ends with `RESULT: OK`
- Compute Engine API enabled
- L4 quota detected in your chosen region (`>=1` for 1x L4 lane, `>=4` for 4x L4 lane)

Default zone is `us-central1-a`. Override if needed:

```bash
GCP_ZONE=us-east4-b make setup-gcp
```

---

## 3) Test Matrix (Run In This Order)

### T1. 1x L4 Free Smoke (Auto-Delete)

```bash
bash scripts/gpu/launch-gcp-l4.sh
```

Monitor:

```bash
gcloud compute ssh alloc-validate-l4 --zone=us-central1-a \
  --command 'tail -f /var/log/alloc-validate.log'
```

Validates:
- GCP launch path
- GPU runtime environment
- free-tier validation
- quick matrix generation

Expected signals:
- log contains `CUDA: True` and `GPU: NVIDIA L4`
- `make validate-free` and `make matrix-quick` both run
- run ends with `alloc-validate GPU test complete`
- instance auto-deletes

---

### T2. Authenticated Upload + Analysis (Manual SSH Session)

Use a manual shell on a running GPU instance. This avoids racing with auto-delete.

1) Create a manual 1x L4 VM:

```bash
gcloud compute instances create alloc-validate-manual-l4 \
  --zone=us-central1-a \
  --machine-type=g2-standard-4 \
  --accelerator=type=nvidia-l4,count=1 \
  --image-family=common-cu128-ubuntu-2204-nvidia-570 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --maintenance-policy=TERMINATE
```

2) SSH and run validation:

```bash
gcloud compute ssh alloc-validate-manual-l4 --zone=us-central1-a

# inside VM
set -euo pipefail
sudo apt-get update -qq && sudo apt-get install -y -qq python3-venv git
cd /tmp
git clone https://github.com/alloc-labs/alloc-validate.git
cd alloc-validate
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"

make validate-free

export ALLOC_TOKEN=<your-token>
make validate-upload
alloc run --upload -- python pytorch/train.py --model large-cnn --max-steps 30
unset ALLOC_TOKEN
```

Validates:
- authenticated artifact upload
- `/runs/ingest` analysis pipeline
- dashboard run creation and recommendation rendering

Expected signals:
- `make validate-upload` prints:
  - `OK: alloc upload succeeded (pytorch)`
  - `OK: alloc upload succeeded (huggingface)`
  - `OK: alloc upload succeeded (lightning)`
- `alloc run --upload ...` exits zero and confirms upload/analysis completion
- dashboard shows new runs within ~1-2 minutes

Dashboard spot-check per uploaded run:
- run row exists with recent timestamp
- peak VRAM is non-null (on GPU lanes)
- bottleneck label/recommendation block is present
- HF/Lightning runs include callback timing fields (step-time/throughput) when captured

---

### T3. 4x L4 Stress Lane (Auto-Delete)

```bash
bash scripts/gpu/launch-gcp-4xl4.sh
```

Monitor:

```bash
gcloud compute ssh alloc-validate-4xl4 --zone=us-central1-a \
  --command 'tail -f /var/log/alloc-validate.log'
```

Validates:
- multi-GPU bring-up
- matrix multi-GPU combos
- process-tree and topology stress path

Expected signals:
- log prints `GPUs detected: 4`
- `make matrix-multi` runs to completion
- no unexpected `FAIL` rows in matrix summary
- instance auto-deletes after completion

---

### T4. Optional Topology Deep Check (Manual 4x L4)

Run this only if you want explicit distributed strategy exercise beyond matrix defaults.

```bash
# inside a manual 4x GPU VM with repo + venv set up
cd /tmp/alloc-validate
source .venv/bin/activate
make validate-topology
```

Validates:
- distributed scripts across DDP/FSDP/PP/TP(+hybrids)

Expected signals:
- summary line prints `RESULT: OK`
- each strategy section prints pass/ok status

---

## 4) Pass/Fail Gate For Tomorrow

Treat test day as successful if all below are true:

1. `make test-quick` is green locally.
2. T1 1x L4 free lane completes with no unexpected failures.
3. T2 authenticated upload lane shows successful uploads and dashboard runs.
4. T3 4x L4 lane detects 4 GPUs and completes matrix-multi.
5. No token appears in startup scripts or instance metadata/log bootstrap.

If one lane fails due to transient cloud/network error, rerun once before classifying as product failure.

---

## 5) Cost Guardrails

- default to Spot/Preemptible for launcher lanes
- use manual on-demand VMs only for authenticated debugging
- run T3/T4 only after T1/T2 pass
- delete manual VMs immediately after use

Manual cleanup:

```bash
gcloud compute instances delete alloc-validate-manual-l4 --zone=us-central1-a --quiet
```

Also clean auto lanes early if needed:

```bash
gcloud compute instances delete alloc-validate-l4 --zone=us-central1-a --quiet
gcloud compute instances delete alloc-validate-4xl4 --zone=us-central1-a --quiet
```

---

## 6) Troubleshooting

Quota too low:
- rerun `make setup-gcp`
- request higher L4 quota in selected region

Upload path failing:
- ensure `ALLOC_TOKEN` is set only in SSH shell
- rerun `make validate-upload`
- if needed, verify local auth with `alloc whoami --json`

Matrix drift / regression suspicion:
- run `make test-quick`
- review `tests/test_repo_hygiene.py` and `tests/test_scan.py`
