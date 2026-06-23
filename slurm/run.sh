#!/bin/bash
#SBATCH --job-name=matmla-train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --export=NONE

# Adjust the partition / gres line to match the cluster you're on:
#   TUHH NCPS: usually --partition=gpu   --gres=gpu:1
#   HLRN:      usually --partition=ml    --gres=gpu:a100:1 (etc.)
# You may also need to set --account=<your_account> on shared clusters.

set -euo pipefail

# --- 1. Setup environment ----------------------------------------------------
# Adjust module loads / venv activation to your site. Examples:
#   module load python/3.11 cuda/12.1
#   source /path/to/your/venv/bin/activate

# Pin the working dir to the repo root no matter where sbatch was invoked from.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

# Make local src and data importable without `pip install -e`.
export PYTHONPATH="${PYTHONPATH:-}:${PWD}"
# (Optional) restrict the visible GPU when --gres gives you one anyway.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p slurm/logs save

# --- 2. Run -----------------------------------------------------------------
# Pick the experiment here. Override on the command line:
#   sbatch slurm/run.sh configs/nested_gqa.yaml -log both -name nested_gqa
CONFIG="${1:-configs/matmla.yaml}"
shift || true   # drop config from the arg list so user flags flow through

echo "[slurm] host=$(hostname) jobid=${SLURM_JOB_ID:-local} start=$(date -Iseconds)"
echo "[slurm] python=$(python3 --version 2>&1) cuda=${CUDA_VISIBLE_DEVICES}"
echo "[slurm] config=${CONFIG}"
echo "[slurm] extra_args=$*"

python3 -u train.py \
    -config "${CONFIG}" \
    "$@"

echo "[slurm] done at $(date -Iseconds)"
