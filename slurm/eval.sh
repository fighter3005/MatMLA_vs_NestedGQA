#!/bin/bash
#SBATCH --job-name=matmla-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --export=NONE

# Usage:
#   sbatch slurm/eval.sh configs/matmla.yaml save/matmla/final.pt
#   sbatch slurm/eval.sh configs/nested_gqa.yaml save/nested_gqa/final.pt -max_batches 32

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONPATH="${PYTHONPATH:-}:${PWD}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p slurm/logs

CONFIG="${1:-configs/matmla.yaml}"
CKPT="${2:-save/matmla/final.pt}"
shift 2 || true

echo "[slurm] eval host=$(hostname) jobid=${SLURM_JOB_ID:-local} start=$(date -Iseconds)"
echo "[slurm] config=${CONFIG} ckpt=${CKPT}"
echo "[slurm] extra_args=$*"

python3 -u evaluate.py -config "${CONFIG}" -ckpt "${CKPT}" "$@"

echo "[slurm] done at $(date -Iseconds)"
