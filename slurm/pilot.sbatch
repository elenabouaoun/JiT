#!/usr/bin/env bash
#SBATCH --job-name=jit-pilot
#SBATCH --account=general
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --open-mode=append

set -euo pipefail
cd "${JIT_REPO:-$HOME/JiT}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2
# Bounded single-GPU training only; never calls main_jit.main or saves checkpoints.
"${JIT_PYTHON:-python}" -u tools/hpc_pilot.py pilot "$@"
