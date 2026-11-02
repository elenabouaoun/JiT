# JiT-B/16 — ImageNet-1K Training Experiment

## Experiment configuration

| Parameter | Value |
|---|---|
| Model | JiT-B/16 |
| Dataset | ImageNet-1K (full training set) |
| Training images | Approximately 1,280,297 |
| Classes | 1,000 |
| Image resolution | 128 x 128 |
| Batch size | 4 |
| GPU | NVIDIA A100 (1 GPU) |
| Training environment | Mizzou Hellbender HPC |
| Completed epochs | 2 |
| Warmup epochs | 0 |
| PyTorch compile | Disabled |

## Training results

| Metric | Epoch 1 | Epoch 2 |
|---|---:|---:|
| Mean logged loss | 0.151095 | 0.117053 |
| First 100 logged losses (mean) | 0.418100 | 0.122452 |
| Last 100 logged losses (mean) | 0.129558 | 0.118811 |
| Logged measurements | 3,201 | 3,201 |
| Training duration | 5:45:41 | 6:02:04 |
| Slurm job | 19395200 | 19408884 |

Mean logged training loss decreased by 22.53% between epochs.

These statistics summarize periodically logged training losses,
not exact full-batch epoch averages or validation metrics.

## Checkpoints

Epoch 1:
`/home/ebkmw/jit-experiments/training/jit-b16-128/run-20261030-072922/checkpoint-epoch000000.pth`

Epoch 2:
`/home/ebkmw/jit-experiments/training/jit-b16-128/resume-epoch1-yNCtxWOq/training/checkpoint-epoch000001.pth`

Local backup directory:
`/home/ebkmw/jit-checkpoint-backups/`

Both original checkpoints were readable and SHA-256 hashes were recorded.
The second checkpoint passed its PyTorch ZIP archive integrity check.

## Observations

- Epoch 1 showed substantial loss reduction during training.
- Epoch 2 began at a substantially lower loss.
- Epoch 2 training loss was relatively stable.
- The results are consistent with continued optimization.
- Image generation quality and generalization have not yet been evaluated.

## Next steps

1. Resume from the epoch 2 checkpoint.
2. Train epoch 3.
3. Track training loss across subsequent epochs.
4. Evaluate generated samples and validation performance.
5. Compare JiT-B/2, B/4, B/8, and B/16 under controlled settings.
