# GPU Testing Guide

Running alloc-validate on real GPUs to fine-tune analysis accuracy, verify VRAM estimates, and validate bottleneck classification on actual hardware.

## Why GPU Tests Matter

CPU-only CI proves the contract (artifact structure, callback integration, CLI correctness). GPU tests prove the **product** (VRAM estimates match reality, bottleneck classification is accurate, config scoring picks the right GPU, timing metrics are meaningful).

## Recommended GPU Instances

Keep costs low. We need real GPU metrics, not massive compute -- small models on cheap GPUs are ideal.

### Tier 1: Primary (use for every release)

| Provider | Instance | GPUs | GPU Type | VRAM | On-Demand Cost | Spot Cost |
|----------|----------|------|----------|------|----------------|-----------|
| GCP | g2-standard-4 | 1x L4 | L4 | 24 GB | ~$0.70/hr | ~$0.21/hr |
| AWS | g4dn.xlarge | 1x T4 | T4 | 16 GB | ~$0.53/hr | ~$0.16/hr |
| Lambda Labs | gpu_1x_a10 | 1x A10 | A10 | 24 GB | ~$0.75/hr | N/A |

**Pick one.** GCP L4 or AWS T4 are the cheapest options. A single 1-GPU instance running the full matrix takes < 10 minutes and costs < $0.15.

### Tier 2: Multi-GPU (use for strategy planner validation)

| Provider | Instance | GPUs | GPU Type | VRAM | On-Demand Cost | Spot Cost |
|----------|----------|------|----------|------|----------------|-----------|
| GCP | g2-standard-16 | 4x L4 | L4 | 4x 24 GB | ~$2.80/hr | ~$0.84/hr |
| AWS | g4dn.12xlarge | 4x T4 | T4 | 4x 16 GB | ~$3.91/hr | ~$1.17/hr |
| Lambda Labs | gpu_4x_a10 | 4x A10 | A10 | 4x 24 GB | ~$3.00/hr | N/A |

Multi-GPU validates: DDP process-tree discovery, multi-GPU VRAM reporting, strategy planner topology recommendations (DDP, FSDP feasibility). Full matrix on 4 GPUs takes < 15 minutes, costs < $1.

### Tier 3: High-end (use sparingly, for calibration)

| Provider | Instance | GPUs | GPU Type | VRAM |
|----------|----------|------|----------|------|
| Lambda Labs | gpu_1x_a100_sxm4 | 1x A100 | A100 SXM4 | 80 GB |
| GCP | a2-highgpu-1g | 1x A100 | A100 | 40 GB |

Only needed when calibrating VRAM estimates for large models or validating A100-specific analysis paths. Use < 1x/month.

## Setup on a Fresh GPU Instance

```bash
# 1. Clone and install
git clone https://github.com/alloc-labs/alloc-validate.git
cd alloc-validate
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"

# 2. Verify GPU is detected
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

# 3. Run free-tier validation (no token needed)
make validate-free

# 4. Run full validation with token (uploads to dashboard)
export ALLOC_TOKEN=<your-token>
make validate-full

# 5. Run full matrix with all model/GPU combos
make matrix

# 6. Multi-GPU matrix (on 4-GPU instances)
make matrix --include-multi-gpu
```

## What to Look For in GPU Results

### VRAM Accuracy

Compare `probe.peak_vram_mb` in the artifact against `alloc ghost` estimates:

```bash
# Get probe VRAM from artifact
alloc run -- python pytorch/train.py --model large-cnn
python -c "
import gzip, json
data = json.loads(gzip.open('pytorch/alloc_artifact.json.gz').read())
print(f'Probe peak VRAM: {data[\"probe\"][\"peak_vram_mb\"]} MB')
"

# Get ghost estimate
alloc ghost pytorch/train.py --json 2>/dev/null | python -c "
import sys, json
data = json.load(sys.stdin)
print(f'Ghost estimated VRAM: {data.get(\"total_vram_mb\", \"N/A\")} MB')
"
```

If ghost estimate is off by > 30% from probe measurement, that's a calibration issue to investigate.

### Timing Metrics (Callback Workloads)

HuggingFace and Lightning callbacks capture step timing. On GPU, these should show meaningful values:

```bash
# Check callback sidecar
cat huggingface/.alloc_callback.json | python -m json.tool
```

Expected on GPU:
- `step_time_ms_p50`: 5-200ms range (depends on model size)
- `samples_per_sec`: > 0

### Bottleneck Classification

After uploading to dashboard (`make validate-full`), check the run detail page. Alloc classifies each run into one of four categories:
- **underutilized**: GPU is oversized for the workload
- **memory_bound**: VRAM is the constraint
- **compute_bound**: GPU compute is fully engaged
- **balanced**: healthy utilization across dimensions

### Multi-GPU DDP

On 4-GPU instances, verify:
- `hardware.num_gpus_detected` = 4 in artifact
- Process-tree discovery finds all GPU processes
- Per-GPU VRAM metrics are reported

## Cost Optimization Tips

1. **Use spot/preemptible instances.** GPU tests are idempotent and fast -- preemption is fine.
2. **Run the quick matrix first.** `make matrix-quick` takes < 3 minutes. Only run full matrix if quick passes.
3. **Shut down immediately after.** Script the teardown:
   ```bash
   make matrix && sudo shutdown -h now
   ```
4. **Budget ceiling.** A full release validation cycle (1x T4 quick + 4x L4 full matrix) should cost < $2 total.
5. **Use the Docker image on GPU instances** to avoid dependency setup time:
   ```bash
   make docker-build
   docker run --rm --gpus all -e ALLOC_TOKEN=$ALLOC_TOKEN alloc-validate make validate-full
   ```

## Distributed Topology Testing

Multi-GPU instances are required to test distributed strategies (DDP, FSDP, PP, TP).

### Quick topology smoke test (any instance)

```bash
# Single-process fallback — validates script runs and alloc wraps it
make distributed
```

### Full topology validation (4-GPU instance)

```bash
# Runs all 4 strategies with torchrun on available GPUs
make validate-topology

# Or run individual strategies
cd distributed
bash validate.sh ddp
bash validate.sh fsdp
bash validate.sh pp
bash validate.sh tp
```

### Topology matrix

```bash
# All strategy × model combos
make matrix-distributed

# Or via runner directly
python scripts/run_matrix.py --framework distributed
```

### Cost Estimates (per topology run)

| Test Type | Instance | Cost |
|-----------|----------|------|
| Single-GPU smoke | 1x T4 or L4 | ~$0.09 |
| DDP/FSDP (4 GPU) | 4x T4 or 4x L4 | ~$0.70-1.00 |
| TP/PP (4 GPU) | 4x L4 | ~$0.70 |
| Full topology matrix | 4x L4, 30 min | ~$1.40 |
| Monthly budget (daily runs) | | ~$42/mo |

## Cloud Setup

### AWS Setup

1. Create an AWS account, set up an IAM user with `AmazonEC2FullAccess`
2. Install the AWS CLI:
   ```bash
   curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o /tmp/AWSCLIV2.pkg && sudo installer -pkg /tmp/AWSCLIV2.pkg -target /
   ```
3. Configure credentials:
   ```bash
   aws configure  # enter access key, secret key, us-east-1, json
   ```
4. Launch a GPU instance:
   ```bash
   bash scripts/gpu/launch-aws-t4.sh       # 1x T4
   bash scripts/gpu/launch-aws-4xt4.sh     # 4x T4
   ```

### GCP Setup

The fastest way to get started:

```bash
make setup-gcp
```

This interactive wizard handles everything: installs `gcloud` CLI if missing, authenticates, creates/selects a project, enables the Compute Engine API, and checks GPU quota. It's idempotent — re-run it anytime to verify your setup.

Env var overrides: `GCP_PROJECT_ID`, `GCP_ZONE` (default `us-central1-a`), `GCP_SKIP_AUTH` (for CI).

After setup completes, launch a GPU instance:

```bash
bash scripts/gpu/launch-gcp-l4.sh       # 1x L4
bash scripts/gpu/launch-gcp-4xl4.sh     # 4x L4
```

<details>
<summary>Manual setup (fallback)</summary>

1. Create a GCP account (comes with $300 free credits)
2. Apply for Google Cloud for Startups ($2K-$200K credits)
3. Install the gcloud CLI:
   ```bash
   curl https://sdk.cloud.google.com | bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```
4. Enable Compute Engine API and request GPU quota:
   - L4: request 4 GPUs in your preferred region
   - A100: request 8 GPUs (if needed for calibration)
5. Launch a GPU instance:
   ```bash
   bash scripts/gpu/launch-gcp-l4.sh       # 1x L4
   bash scripts/gpu/launch-gcp-4xl4.sh     # 4x L4
   ```

</details>

With GCP startup credits, topology testing is effectively free for 2+ years.

## CI Integration (Future)

When ready, add a GitHub Actions nightly workflow:

```yaml
# .github/workflows/gpu-nightly.yml
name: GPU Validation (Nightly)
on:
  schedule:
    - cron: '0 6 * * *'  # 6 AM UTC daily
  workflow_dispatch: {}

jobs:
  gpu-test:
    runs-on: [self-hosted, gpu]  # or use a cloud GPU runner service
    steps:
      - uses: actions/checkout@v4
      - run: pip install -e ".[all]"
      - run: make validate-full
        env:
          ALLOC_TOKEN: ${{ secrets.ALLOC_TOKEN }}
      - run: make matrix --include-multi-gpu
```

Self-hosted runners on a preemptible instance keep nightly costs at ~$0.50/day. Alternatively, use [Cirun](https://cirun.io) or [RunsOn](https://runs-on.com) for on-demand GPU runners without managing infrastructure.

## Recording Baseline Results

After a GPU test run, save the results for comparison:

```bash
# Save matrix output
make matrix --json > baselines/$(date +%Y%m%d)_$(hostname).json

# Save individual artifacts
mkdir -p baselines/artifacts
cp pytorch/alloc_artifact.json.gz baselines/artifacts/pytorch_$(date +%Y%m%d).json.gz
cp huggingface/alloc_artifact.json.gz baselines/artifacts/hf_$(date +%Y%m%d).json.gz
```

Compare across releases to detect regressions in VRAM estimates, timing accuracy, or bottleneck classification.
