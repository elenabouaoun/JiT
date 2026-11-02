# Phase 1: ImageNet-128 patch-size pilots on Hellbender

These scripts run preflight or bounded **single-A100** pilots only. They never call
the full training entry point, load a checkpoint, save a checkpoint, submit another
job, or run denoising/generation evaluation. No Slurm jobs were submitted during
implementation. A successful CPU suite does not establish GPU feasibility.

## Shared model and attention

`JiT-B/2`, `JiT-B/4`, `JiT-B/8`, and `JiT-B/16` use the same 12-block,
768-wide, 12-head transformer, SwiGLU ratio, 128-wide patch embedding intermediate,
and 32 class-context tokens inserted at block index 4. The input/output projections
and position lengths vary with patch size; total parameter counts therefore vary.
The intermediate width expands RGB patches for /2 and /4 and compresses /8 and /16.

PyTorch scaled-dot-product attention replaces explicit FP32 score matrices. The
normal model permits PyTorch backend selection; the pilot forces **FlashAttention
only**, failing rather than silently falling back to dense math attention. Rotary
tensors are nonpersistent buffers so `.to(device)` works without changing old
checkpoint keys. No checkpoint conversion is performed.

At resolution 128, image token counts are 4096, 1024, 256, and 64. Maximum lengths
including context tokens are 4128, 1056, 288, and 96. A single dense FP32 score
tensor at batch 1, 12 heads would occupy roughly 780, 51, 3.8, and 0.42 MiB per
layer. Fused attention avoids storing this full tensor, but not quadratic compute.
In particular /2 feasibility is not established until an actual A100 pilot passes.

## Prerequisites (login node)

Use the existing checkout at `$HOME/JiT` (or set `JIT_REPO`). Activate an existing
GPU environment, then export `JIT_PYTHON` as its **absolute interpreter path**.
Use Python 3.10, PyTorch 2.5.1 with a driver-compatible CUDA build (the repository
specifies CUDA 12.4), torchvision 0.20.1, and the dependencies in `environment.yaml`.
The onboarding cloud environment has CPU wheels and cannot run these GPU pilots.
Do not copy that virtualenv to Hellbender. Workers do not install dependencies.

Before submission, inspect the site's actual GPU GRES labels, for example:

```bash
sinfo -p gpu -N -o '%N %G %m %c'
```

Both scripts specify account `general`, partition `gpu`, QoS `normal`, and
`--gres=gpu:A100:1`. If the configured A100 label has different spelling, override
the GRES on the `sbatch` command using the label reported by Hellbender. Record
whether the allocated A100 has 40 or 80 GB. The scripts request 30 minutes for
preflight and 2 hours for pilots, below the 48-hour job limit. CPU and host-memory
requests should be checked against the selected node.

The expected data paths are `$HOME/data/imagenet/train` and
`$HOME/data/imagenet/val`, each with matching 1,000 ImageFolder class directories.
Flat validation images must be handled in a later dataset-loader change; these
scripts fail clearly and never rearrange the source dataset.

The protected checkpoint must exist at:
`$HOME/JiT/checkpoints/jit-b-16/checkpoint-last.pth`. The scripts read its SHA-256
before and after both successful runs and ordinary exceptions; they do not
deserialize or write it. They also detect changes during hashing. An explicit
`--protected-checkpoint` can select another existing location if the checkout is
elsewhere. Add `--expected-checkpoint-sha256 <trusted-digest>` to verify against a
known baseline as well. Before/after hashes prove stability, not that the original
file has the correct model/schema or was uncorrupted before this task. SIGKILL
cannot run the final integrity check. This phase is not a restart/recovery test.

## CPU validation

From the repository with dependencies activated:

```bash
PYTHONDONTWRITEBYTECODE=1 TORCH_COMPILE_DISABLE=1 \
  python -m unittest discover -s tests -v
bash -n slurm/preflight.sbatch slurm/pilot.sbatch
```

Tests cover FP32 SDPA output/input gradients against explicit attention, dropout
mode, full B backbone specifications, small-model forward/backward at all four
patch sizes, unchanged rotary checkpoint keys, accumulation equivalence to larger
batches including short tails, optimizer/EMA counts, pilot bounds, and exclusive
result writing. Further regressions check warmup/cosine LR against equivalent
large batches, the complete training timing boundary, mismatched dataset class
mappings, failure-path integrity checks, and concurrent checkpoint-publication
collisions. The CUDA-only test compares BF16 FlashAttention outputs and gradients
with explicit FP32 attention using relative L2 tolerances of 2% and 3%.
It covers sequence lengths 96, 288, 1056, and 4128 (12 heads, head width 64), using
16 query rows for the reference to avoid a dense 4128x4128 allocation. The loss
uses those same rows, allowing comparison of the full Q/K/V gradients.
It is skipped without CUDA and **required** during Slurm preflight. Small CPU
models verify computation; full backbone shapes are inspected on the meta device.

Every preflight **and every pilot** additionally runs this validation directly and
saves errors in `attention-validation.json`. It restores the training RNG state.
Both validation and pilot training permit only the FlashAttention backend; lack
of a supported CUDA/BF16 kernel raises an explicit error, never a math fallback.
Passing attention validation alone does not establish full-model memory feasibility.

## First preflight and GPU pilot

The following are instructions for you to execute; they were not submitted by the
implementation task. Choose a quota-approved persistent output location. The
example uses `$HOME/jit-experiments`; switch to approved project storage if needed.

```bash
cd "$HOME/JiT"
# Activate your GPU Python environment first.
export JIT_PYTHON="$(command -v python)"
export JIT_REPO="$PWD"
mkdir -p "$HOME/jit-experiments"
LOG_DIR=$(mktemp -d "$HOME/jit-experiments/slurm-XXXXXXXX")

sbatch --output="$LOG_DIR/preflight-%j.out" slurm/preflight.sbatch \
  --output-root "$HOME/jit-experiments/preflight"
```

Inspect the completed preflight log and its `summary.json` (`status: passed`)
before submitting the first pilot:

```bash
sbatch --output="$LOG_DIR/pilot-p16-%j.out" slurm/pilot.sbatch \
  --patch-size 16 --batch-size 1 --accum-steps 1 \
  --warmup-steps 5 --steps 20 --seed 0 \
  --output-root "$HOME/jit-experiments/pilots"
```

Then submit /8, /4, and /2 individually with the same arguments except patch size,
inspecting each outcome before proceeding. Begin with microbatch 1. Increase the
microbatch only after checking memory headroom. Separately test accumulation, for
example `--accum-steps 4`. A pilot's effective batch may be small: these runs measure
feasibility, not comparative quality. Do not interpret pilot losses as final results.
Submissions are never automatic, and failures are not automatically resubmitted.

Eager mode is the default for isolating memory issues. Use a separate pilot with
`--compile` to include existing torch.compile decorators and measure compiled
behavior; allocate enough warmup for compilation/recompilation. No activation
checkpointing has been added in phase 1. If /2 is too large even at microbatch 1,
retain its failure evidence and assess activation checkpointing as a next change.

## Measurements and safety

Every invocation exclusively creates a random-suffixed directory containing:

- configuration, source commit/dirty status, GPU and dependency versions;
- dataset counts and both train/validation mappings, verified identical with 1,000
  classes; checkpoint hashes before/after success or ordinary failure;
- `attention-validation.json`: BF16/FP32 output and Q/K/V-gradient relative errors;
- `steps.jsonl`: flushed per optimizer update, raw sample-mean training loss,
  learning rate, schedule progress, `training_pipeline_images_per_second`,
  `training_pipeline_seconds`, and peak allocated/reserved bytes;
- `summary.json` after success, or `failure.json` after a Python exception.

Training pilots default to 5 warmup plus 20 measured **optimizer** updates. Measured
updates are capped at 200 and warmup at 50. Warmup updates perform real optimization
but are excluded from throughput/loss summaries. Peak memory includes warmup,
model, EMA, and lazy optimizer allocations. **Training-pipeline throughput** measures
images divided by the sum of timed optimizer windows, not a mean of per-window
rates. Timing begins before fetching data and ends after optimizer/zero-grad/EMA
work and CUDA synchronization. It includes data loading, H2D transfers,
forward/backward, accumulation, optimizer and EMA work. It excludes model/data
setup, attention validation, checkpoint hashing, JSON/TensorBoard/console logging,
and excluded timing-warmup updates. It is not whole-job walltime throughput. Slurm logs
are separate and use unique log directories and job IDs. Hard scheduler kills may
leave only partial JSONL and logs, rather than a failure/summary file.

The pilot uses the original stochastic flow-matching loss and real training images,
not synthetic input or a generation benchmark. It uses a constant nonzero LR
`5e-5 * effective_batch / 256`; the production five-epoch LR warmup is intentionally
not run in a 25-update pilot. Seeds for Python, NumPy, PyTorch, sampling, and loader
workers are reproducible, but fused kernels and changed batch sizes do not promise
bitwise-identical results. Do not compare pilots with different batch protocols as
controlled training experiments.

`--accum_steps` in `main_jit.py` scales effective batch and LR. Gradients are
sample-weighted, including short final batches/windows; optimizer and both EMAs
advance once per window. DDP suppresses gradient synchronization before the
window boundary. DDP ranks must have equal batch sizes, as with the existing
DistributedSampler. The normal training entry point now refuses an existing output
directory; resume input and output must be separate.

### Epoch-based learning-rate progression with accumulation

For `M` microbatches per epoch and accumulation factor `A`, let
`U = ceil(M / A)` be the number of optimizer updates. Before update `u` (zero-based),
the scheduler receives `epoch + u / U`. LR is held fixed across all microbatches
in that update. A final partial accumulation window still counts as one update.
For example, five microbatches with `A=2` use progress `0`, `1/3`, `2/3`, followed
by `1` at the next epoch's first update; they do not use `0`, `2/5`, `4/5`.
Thus changing microbatching while retaining the same effective batches and update
count gives the same LR trajectory, including short tails.

Warmup and cosine budgets remain expressed in epochs. With five warmup epochs,
the first update has LR zero and base LR is reached at the first update of epoch
index 5. Cosine is sampled at update starts, so the final update occurs before
the mathematical end-of-training minimum. Resume at an epoch boundary preserves
this progression when loader length and accumulation are unchanged. Mid-epoch
restart and reproducible RNG restoration remain outside this phase. Pilot
`--warmup-steps` controls timing exclusion only; pilot LR remains constant.

### Checkpoint and legacy-evaluation safety

Pilots remain entirely checkpoint-free. The legacy full-training saver now uses
immutable `checkpoint-epoch000000.pth` names and creates a checkpoint at most once
per selected epoch. It writes a private temporary file, flushes/fsyncs it, and
publishes by an atomic no-clobber hard link. Existing files and symlinks, including
`checkpoint-last.pth`, are never replaced; a collision or unsupported hard-link
filesystem fails safely. Failed saves remove only their private temporary file.
There is no automatic pruning. Select immutable resume files explicitly with
`--resume /path/to/checkpoint-epoch000100.pth`; legacy directories containing
`checkpoint-last.pth` still work. A requested missing checkpoint now fails instead
of silently training from scratch.

The pre-existing evaluation path also reserves a new UUID-suffixed directory per
evaluation, creates image files exclusively, and retains samples after scoring.
It cannot reuse or delete an earlier evaluation's images. This is a safety fix
to legacy behavior, not implementation of the new denoising/generation pipeline.

No full training, denoising metrics/panels, generation, FID, checkpoint loading,
or restart validation is part of this phase.
