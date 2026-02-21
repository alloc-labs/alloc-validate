# distributed/ — Distributed Topology Workloads

Synthetic transformer models exercising PyTorch distributed training strategies.
Validates that `alloc run` detects topology metadata (strategy, degrees, interconnect)
and that `alloc ghost` can estimate VRAM for distributed training scripts.

## Training Scripts

| Script | Strategy | Default Model | GPUs | PyTorch API |
|--------|----------|--------------|------|-------------|
| `train_ddp.py` | DDP | small (~20M) | 2-4 | `torchrun` + `DistributedDataParallel` |
| `train_fsdp.py` | FSDP | medium (~150M) | 4 | `torchrun` + `FullyShardedDataParallel` |
| `train_pp.py` | PP | large (~600M) | 4 | Manual pipeline stages across devices |
| `train_tp.py` | TP | large (~600M) | 4 | Manual all-reduce simulating tensor parallelism |

## Model Variants

All scripts use `SyntheticTransformer` — a configurable `nn.TransformerEncoder` with a linear head.

| Variant | d_model | nhead | layers | d_ff | ~Params |
|---------|---------|-------|--------|------|---------|
| `small` | 512 | 8 | 6 | 2048 | ~20M |
| `medium` | 1024 | 16 | 12 | 4096 | ~150M |
| `large` | 2048 | 16 | 24 | 8192 | ~600M |
| `xl` | 4096 | 32 | 32 | 16384 | ~3B |

Default models: `small` for DDP, `medium` for FSDP, `large` for PP/TP.
`large` and `xl` are for GPU-only runs.

## Running

```bash
# CPU smoke test (single process)
make distributed

# Individual strategy
alloc run -- torchrun --nproc_per_node=1 distributed/train_ddp.py --model small
alloc run -- python distributed/train_fsdp.py --model small

# Multi-GPU (4x GPU instance)
alloc run -- torchrun --nproc_per_node=4 distributed/train_ddp.py --model medium
alloc run -- torchrun --nproc_per_node=4 distributed/train_fsdp.py --model medium

# Full topology validation
make validate-topology
```

## CPU Fallback

All scripts degrade gracefully on CPU:
- `torchrun --nproc_per_node=1` runs a single process
- Direct `python` invocation skips process group init
- DDP/FSDP wrapping is skipped when world_size=1
- PP runs all stages sequentially on CPU
