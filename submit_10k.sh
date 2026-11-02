#!/usr/bin/env bash
set -euo pipefail

DATA="$HOME/data/imagenet-10k"
COUNT=$(find "$DATA/train" -type l | wc -l)
CLASSES=$(find "$DATA/train" -mindepth 1 -maxdepth 1 -type d | wc -l)

echo "Images: $COUNT | Classes: $CLASSES"
test "$COUNT" -eq 10000
test "$CLASSES" -eq 1000

mkdir -p "$HOME/jit-experiments/training/jit-b16-10k"

cat > "$HOME/JiT/slurm/train_10k.sbatch" <<'SLURM'
#!/usr/bin/env bash
#SBATCH --job-name=jit-b16-10k
#SBATCH --account=general
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/ebkmw/jit-experiments/training/jit-b16-10k/slurm-%j.log

set -euo pipefail
cd "$HOME/JiT"

"$HOME/jit-pilot-env/bin/python" -u main_jit.py \
  --model JiT-B/16 \
  --img_size 128 \
  --data_path "$HOME/data/imagenet-10k" \
  --batch_size 4 \
  --epochs 5 \
  --save_last_freq 1 \
  --output_dir "$HOME/jit-experiments/training/jit-b16-10k"
SLURM

sbatch "$HOME/JiT/slurm/train_10k.sbatch"
