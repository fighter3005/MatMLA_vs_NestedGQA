# SLURM scripts

Two scripts wrap `train.py` and `evaluate.py` for cluster submission:

| Script | Purpose |
|---|---|
| `run.sh` | Submit a training job (default: `configs/matmla.yaml`) |
| `eval.sh` | Sweep every sub-model on a saved checkpoint + print the cache-size table |

## Quick start

```bash
# Train MatMLA on one GPU.
sbatch slurm/run.sh

# Train NestedGQA with wandb logging.
sbatch slurm/run.sh configs/nested_gqa.yaml -log wandb -name nested_gqa_test

# Evaluate a finished checkpoint.
sbatch slurm/eval.sh configs/matmla.yaml save/matmla/final.pt

# Evaluate with fewer validation batches (faster).
sbatch slurm/eval.sh configs/nested_gqa.yaml save/nested_gqa/final.pt -max_batches 32
```

Logs land in `slurm/logs/<jobname>-<jobid>.{out,err}`. Checkpoints land in
`save/<run-name>/final.pt`.

## Customising for your cluster

The `#SBATCH` directives at the top of each script are written for a generic
1-GPU node. You will likely need to tweak them:

| Directive | Common values | Notes |
|---|---|---|
| `--partition` | `gpu`, `ml`, `a100`, `h100` | Match the cluster's advertised GPU partitions |
| `--gres` | `gpu:1`, `gpu:a100:1`, `gpu:h100:1` | Some sites want the GPU model spelled out |
| `--account` | (your PI account) | Required on shared clusters that bill accounts |
| `--time` | `04:00:00`, `08:00:00`, `24:00:00` | Job walltime |
| `--mem` | `32G`, `64G`, `128G` | Per-node RAM. The 47M configs need ~16-32 GB during eval due to the `[B,T,V]` logits tensor |
| `--cpus-per-task` | `4`, `8` | DataLoader workers |

The environment setup block in each script:

```bash
module load python/3.11 cuda/12.1   # adjust to your site
source /path/to/your/venv/bin/activate
```

is intentionally generic — uncomment and edit as needed.

## Performance notes

- **47M models**: on a single A100 the configs should fit comfortably with
  `batch_size=8, grad_accum=4`. Each `eval_interval` step evaluates **9
  sub-models × 16 validation batches**, so it's worth setting
  `eval_interval` to a higher value (e.g. 2000) for long runs.
- **10M `configs/smoke_real.yaml`**: useful for a quick cluster smoke test
  (~5 minutes walltime on a single A100) to verify the submission script
  works end-to-end before launching the full 47M run.

## Walltime budgeting

Approximate per-step wallclock on a single A100 (80 GB) at `seq_len=512`:

| Config | Steps/sec | 20k steps | +9-submodel eval × 2000 steps |
|---|---|---|---|
| `base_mha` | ~6 | ~55 min | +20 min |
| `base_gqa` | ~7 | ~48 min | +18 min |
| `nested_gqa` | ~5 | ~67 min | +25 min |
| `matmla` | ~5 | ~67 min | +25 min |

Adjust `--time` accordingly. For full 20k-step runs on a 47M model, plan
on `--time=04:00:00` to be safe.
