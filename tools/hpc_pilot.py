"""Bounded, checkpoint-free Hellbender preflight and single-GPU pilots."""
import argparse
import copy
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def write_json(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, indent=2, default=str, allow_nan=False)
        stream.write('\n')


def checkpoint_hash(path):
    # Read-only: never deserialize, chmod, rename, copy over, or save a checkpoint.
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
        after = os.fstat(stream.fileno())
    identity = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if identity(before) != identity(after) or identity(after) != identity(path.stat()):
        raise RuntimeError('Protected checkpoint changed while hashing')
    return dict(path=str(path.resolve()), bytes=after.st_size, mtime_ns=after.st_mtime_ns,
                sha256=digest.hexdigest())


@contextmanager
def verify_checkpoint_integrity(path, output, expected_sha256=None):
    before = checkpoint_hash(path)
    write_json(output / 'protected-checkpoint-before.json', before)
    if expected_sha256 is not None and before['sha256'] != expected_sha256.lower():
        raise RuntimeError('Protected checkpoint SHA-256 does not match the expected trusted digest')
    try:
        yield
    finally:
        # Also verify after ordinary failures (OOM, missing kernel, dataset errors).
        # SIGKILL/scheduler termination cannot execute this guard.
        after = checkpoint_hash(path)
        write_json(output / 'protected-checkpoint-after.json', after)
        if before != after:
            raise RuntimeError('Protected checkpoint changed during the run')


def verify_class_mappings(train, val, expected_classes=1000):
    train_root, val_root = Path(train.root).resolve(), Path(val.root).resolve()
    if train_root == val_root:
        raise RuntimeError('Training and validation must use separate dataset directories')
    if (len(train.classes) != expected_classes or len(val.classes) != expected_classes or
            train.class_to_idx != val.class_to_idx or not len(train) or not len(val)):
        raise RuntimeError(f'Expected matching nonempty ImageFolder train/val mappings with {expected_classes} classes')
    return dict(train_images=len(train), val_images=len(val), class_mapping_match=True,
                train_root=str(train_root), val_root=str(val_root),
                train_class_to_idx=train.class_to_idx, val_class_to_idx=val.class_to_idx)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['preflight', 'pilot'])
    parser.add_argument('--data-path', type=Path, default=Path.home() / 'data/imagenet')
    parser.add_argument('--output-root', type=Path, default=Path.home() / 'jit-experiments/pilots')
    parser.add_argument('--protected-checkpoint', type=Path,
                        default=Path.home() / 'JiT/checkpoints/jit-b-16/checkpoint-last.pth')
    parser.add_argument('--expected-checkpoint-sha256', help='Optional trusted digest, in addition to before/after verification')
    parser.add_argument('--patch-size', type=int, choices=[2, 4, 8, 16], default=16)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--accum-steps', type=int, default=1)
    parser.add_argument('--steps', type=int, default=20, help='Measured optimizer updates (maximum 200)')
    parser.add_argument('--warmup-steps', type=int, default=5, help='Untimed optimizer updates (maximum 50)')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--compile', action='store_true', help='Enable existing torch.compile decorators')
    cli = parser.parse_args()
    if (cli.batch_size < 1 or cli.accum_steps < 1 or not 1 <= cli.steps <= 200 or
            not 0 <= cli.warmup_steps <= 50 or cli.num_workers < 0):
        parser.error('Invalid batch, accumulation, step, warmup, or worker count')
    if cli.expected_checkpoint_sha256 is not None:
        digest = cli.expected_checkpoint_sha256.lower()
        if len(digest) != 64 or any(character not in '0123456789abcdef' for character in digest):
            parser.error('Expected checkpoint SHA-256 must be 64 hexadecimal characters')
    # Set before importing decorated model modules. Eager is useful for first OOM pilots.
    os.environ['TORCH_COMPILE_DISABLE'] = '0' if cli.compile else '1'
    cli.data_path = cli.data_path.expanduser().resolve()
    cli.protected_checkpoint = cli.protected_checkpoint.expanduser().resolve()
    cli.output_root = cli.output_root.expanduser().resolve()
    protected_tree = cli.protected_checkpoint.expanduser().resolve().parent
    if cli.output_root == protected_tree or protected_tree in cli.output_root.parents:
        parser.error('Output root must be outside the protected checkpoint directory')
    cli.output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=f'{cli.mode}-p{cli.patch_size:02d}-s{cli.seed}-', dir=cli.output_root))
    print(f'Unique output directory: {output}', flush=True)
    write_json(output / 'arguments.json', vars(cli))
    try:
        run(cli, output)
    except BaseException as error:
        failure = dict(type=type(error).__name__, message=str(error))
        torch = sys.modules.get('torch')
        if torch is not None and torch.cuda.is_available() and torch.cuda.is_initialized():
            failure.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                           peak_reserved_bytes=torch.cuda.max_memory_reserved())
        write_json(output / 'failure.json', failure)
        raise


def run(cli, output):
    with verify_checkpoint_integrity(cli.protected_checkpoint, output, cli.expected_checkpoint_sha256):
        summary = run_checks(cli, output)
    write_json(output / 'summary.json', summary)
    print(json.dumps({key: value for key, value in summary.items() if key != 'steps'}, indent=2), flush=True)


def run_checks(cli, output):
    import torch
    import torchvision
    from PIL import Image
    from torchvision import datasets, transforms
    from util.crop import center_crop_arr
    from util.attention_validation import flash_attention_only, validate_bf16_flash_attention

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('A CUDA GPU with BF16 support is required; CPU wheels cannot run this pilot')
    if torch.__version__.split('+')[0] != '2.5.1' or torchvision.__version__.split('+')[0] != '0.20.1':
        raise RuntimeError('Use the pinned torch 2.5.1 and torchvision 0.20.1 GPU environment')
    if torch.cuda.device_count() != 1:
        raise RuntimeError('This bounded pilot expects exactly one visible GPU; request one Slurm GPU')
    if 'A100' not in torch.cuda.get_device_name(0):
        raise RuntimeError('This pilot requires an A100 allocation')
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    torch.manual_seed(cli.seed)
    import random
    import numpy as np
    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    repo = Path(__file__).resolve().parents[1]
    environment = dict(python=platform.python_version(), torch=torch.__version__,
                       torchvision=torchvision.__version__, cuda=torch.version.cuda,
                       gpu=torch.cuda.get_device_name(0), gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
                       attention_backend='FLASH_ATTENTION only', compile=cli.compile,
                       git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
                       git_status=subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo, text=True),
                       nvidia_smi=subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                                                           '--format=csv,noheader'], text=True).strip(),
                       slurm_job_id=os.environ.get('SLURM_JOB_ID'))
    write_json(output / 'environment.json', environment)
    # Required for every pilot, not just when the user ran a separate preflight.
    write_json(output / 'attention-validation.json', validate_bf16_flash_attention())

    transform = transforms.Compose([
        transforms.Lambda(lambda image: center_crop_arr(image, 128)),
        transforms.RandomHorizontalFlip(), transforms.PILToTensor()])
    train = datasets.ImageFolder(cli.data_path / 'train', transform=transform)
    val = datasets.ImageFolder(cli.data_path / 'val')
    dataset_report = verify_class_mappings(train, val)
    for dataset in (train, val):
        with Image.open(dataset.samples[0][0]) as image:
            image.convert('RGB').load()
    write_json(output / 'dataset.json', dataset_report)

    if cli.mode == 'preflight':
        # Includes CUDA BF16 FlashAttention/reference output and gradient checks.
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
                       cwd=repo, check=True, env={**os.environ, 'TORCH_COMPILE_DISABLE': '1',
                                                 'JIT_REQUIRE_CUDA_TESTS': '1'})
        summary = dict(status='passed', training_started=False)
    else:
        from main_jit import get_args_parser, effective_batch_size
        from denoiser import Denoiser
        from engine_jit import train_one_epoch
        from util.misc import add_weight_decay

        args = get_args_parser().parse_args([])
        args.model = f'JiT-B/{cli.patch_size}'
        args.img_size = 128
        args.batch_size = cli.batch_size
        args.accum_steps = cli.accum_steps
        args.seed = cli.seed
        args.data_path = str(cli.data_path)
        args.output_dir = str(output)
        args.num_workers = cli.num_workers
        args.log_freq = 1
        # Constant nonzero LR isolates numerical/memory behavior in a short pilot.
        # The full training five-epoch warmup is not meaningful over 25 updates.
        args.warmup_epochs = 0
        args.lr = args.blr * effective_batch_size(args, 1) / 256
        write_json(output / 'training-config.json', vars(args))
        total_steps = cli.steps + cli.warmup_steps
        needed = total_steps * cli.accum_steps * cli.batch_size
        if needed > len(train):
            raise ValueError('Requested pilot exceeds one training-data pass')
        indices = torch.randperm(len(train), generator=torch.Generator().manual_seed(cli.seed))[:needed].tolist()
        loader = torch.utils.data.DataLoader(torch.utils.data.Subset(train, indices),
                                            batch_size=cli.batch_size, num_workers=cli.num_workers,
                                            pin_memory=True, drop_last=True,
                                            generator=torch.Generator().manual_seed(cli.seed))
        model = Denoiser(args).to(device)
        model.ema_params1 = copy.deepcopy(list(model.parameters()))
        model.ema_params2 = copy.deepcopy(list(model.parameters()))
        optimizer = torch.optim.AdamW(add_weight_decay(model, args.weight_decay),
                                     lr=args.lr, betas=(0.9, 0.95))
        write_json(output / 'model.json', dict(trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                                             image_tokens=(128 // cli.patch_size) ** 2,
                                             effective_batch_size=cli.batch_size * cli.accum_steps))
        with (output / 'steps.jsonl').open('x') as stream:
            def record(step):
                stream.write(json.dumps(step, allow_nan=False) + '\n')
                stream.flush()
            # Never allow a silent quadratic math-attention fallback.
            with flash_attention_only():
                summary = train_one_epoch(model, model, loader, optimizer, device, 0, args=args,
                                          max_optimizer_steps=total_steps, warmup_steps=cli.warmup_steps,
                                          step_callback=record)
        if summary['optimizer_steps'] != total_steps or summary['measured_steps'] != cli.steps:
            raise RuntimeError('Pilot did not complete its requested updates')
        summary.update(status='passed', peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved(), checkpoints_written=False)
    return summary


if __name__ == '__main__':
    main()
