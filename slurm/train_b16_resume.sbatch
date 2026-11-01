#!/bin/bash
#SBATCH --job-name=jit-b16-ep2
#SBATCH --account=general
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/ebkmw/jit-experiments/training/jit-b16-128/slurm-%j.log

set -euo pipefail

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29537
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0

cd "$HOME/JiT"

"$HOME/jit-pilot-env/bin/python" -u main_jit.py \
  --model JiT-B/16 \
  --img_size 128 \
  --data_path "$HOME/data/imagenet" \
  --epochs 2 \
  --warmup_epochs 0 \
  --batch_size 4 \
  --output_dir "$HOME/jit-experiments/training/jit-b16-128/run-20261030-072922" \
  --resume "$HOME/jit-experiments/training/jit-b16-128/run-20261030-072922/checkpoint-epoch000000.pth" \
  --save_last_freq 1
