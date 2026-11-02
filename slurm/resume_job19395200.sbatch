#!/usr/bin/env bash
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

cd /home/ebkmw/JiT

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29537
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0

CHECKPOINT=/home/ebkmw/jit-experiments/training/jit-b16-128/run-20261030-072922/checkpoint-epoch000000.pth

test -f "$CHECKPOINT"

RUN_DIR=$(mktemp -d /home/ebkmw/jit-experiments/training/jit-b16-128/resume-epoch1-XXXXXXXX)

exec /home/ebkmw/jit-pilot-env/bin/python -u main_jit.py \
  --model JiT-B/16 \
  --img_size 128 \
  --batch_size 4 \
  --data_path /home/ebkmw/data/imagenet \
  --epochs 2 \
  --warmup_epochs 0 \
  --save_last_freq 1 \
  --resume "$CHECKPOINT" \
  --output_dir "$RUN_DIR/training"
