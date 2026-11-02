#!/usr/bin/env bash
#SBATCH --job-name=jit-preflight
#SBATCH --account=general
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --open-mode=append

set -euo pipefail
cd "${JIT_REPO:-$HOME/JiT}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2
# JIT_PYTHON selects an already configured GPU environment; no installs on workers.
"${JIT_PYTHON:-python}" -u tools/hpc_pilot.py preflight "$@"
