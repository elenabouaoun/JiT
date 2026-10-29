import contextlib
import copy
import io
import math
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault('TORCH_COMPILE_DISABLE', '1')

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import DataLoader, TensorDataset

from engine_jit import train_one_epoch
from main_jit import get_args_parser, effective_batch_size
from model_jit import JiT, JiT_models, Attention, scaled_dot_product_attention

torch.set_num_threads(2)


def reference_attention(q, k, v):
    # Explicit FP32 scoring used by the original implementation (dropout disabled).
    weights = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(q.size(-1))
    return weights.softmax(dim=-1) @ v.float()


class AttentionTests(unittest.TestCase):
    def test_math_attention_output_and_gradients(self):
        torch.manual_seed(7)
        inputs = [torch.randn(2, 3, 17, 16, requires_grad=True) for _ in range(3)]
        refs = [tensor.detach().clone().requires_grad_() for tensor in inputs]
        with sdpa_kernel(SDPBackend.MATH):
            actual = scaled_dot_product_attention(*inputs)
        expected = reference_attention(*refs)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        upstream = torch.randn_like(actual)
        actual.backward(upstream)
        expected.backward(upstream)
        for actual_input, reference_input in zip(inputs, refs):
            torch.testing.assert_close(actual_input.grad, reference_input.grad, atol=2e-6, rtol=2e-5)

    def test_attention_dropout_only_in_training(self):
        attention = Attention(32, num_heads=4, attn_drop=0.5)
        x = torch.randn(1, 8, 32)
        with patch('model_jit.scaled_dot_product_attention', wraps=scaled_dot_product_attention) as call:
            attention.train()(x, lambda tensor: tensor)
            self.assertEqual(call.call_args.kwargs['dropout_p'], 0.5)
            attention.eval()(x, lambda tensor: tensor)
            self.assertEqual(call.call_args.kwargs['dropout_p'], 0.0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable: BF16 FlashAttention needs GPU validation')
    def test_cuda_flash_bfloat16_outputs_and_gradients(self):
        from util.attention_validation import validate_bf16_flash_attention
        results = validate_bf16_flash_attention()
        self.assertEqual([record['sequence_length'] for record in results], [96, 288, 1056, 4128])

    def test_pilots_forbid_fallback_and_explain_unavailable_kernel(self):
        from util.attention_validation import flash_attention_only
        with patch('torch.cuda.is_available', return_value=False), self.assertRaisesRegex(RuntimeError, 'never fall back'):
            with flash_attention_only():
                self.fail('Must fail before running CPU/math attention')
        with (patch('torch.cuda.is_available', return_value=True),
              patch('torch.cuda.is_bf16_supported', return_value=True),
              patch('torch.backends.cuda.is_flash_attention_available', return_value=True),
              patch('util.attention_validation.sdpa_kernel', return_value=contextlib.nullcontext()) as backend,
              self.assertRaisesRegex(RuntimeError, 'no dense fallback')):
            with flash_attention_only():
                raise RuntimeError('No available kernel. Aborting execution.')
        backend.assert_called_once_with(SDPBackend.FLASH_ATTENTION)

    def test_required_cuda_is_not_silently_skipped(self):
        if os.environ.get('JIT_REQUIRE_CUDA_TESTS') == '1':
            self.assertTrue(torch.cuda.is_available(), 'Preflight requires CUDA tests')
            self.assertTrue(torch.cuda.is_bf16_supported(), 'Preflight requires BF16')


class ModelTests(unittest.TestCase):
    def test_registered_backbones_are_identical(self):
        shapes = None
        for size in (2, 4, 8, 16):
            with self.subTest(patch_size=size), torch.device('meta'):
                model = JiT_models[f'JiT-B/{size}'](input_size=128)
                self.assertEqual((len(model.blocks), model.hidden_size, model.num_heads), (12, 768, 12))
                self.assertEqual((model.in_context_len, model.in_context_start), (32, 4))
                self.assertEqual(model.x_embedder.proj1.out_channels, 128)
                self.assertEqual(model.x_embedder.num_patches, (128 // size) ** 2)
                current = {name: tuple(p.shape) for name, p in model.named_parameters()
                           if name.startswith(('blocks.', 't_embedder.', 'y_embedder.'))}
                if shapes is not None:
                    self.assertEqual(current, shapes)
                shapes = current

    def test_all_patch_sizes_forward_backward_and_checkpoint_keys(self):
        for size in (2, 4, 8, 16):
            with self.subTest(patch_size=size):
                model = JiT(input_size=32, patch_size=size, hidden_size=32, depth=2,
                            num_heads=4, bottleneck_dim=8, in_context_len=2, in_context_start=1)
                # Nonzero gates/readout exercise attention gradients despite zero initialization.
                torch.nn.init.normal_(model.final_layer.linear.weight, std=0.02)
                for block in model.blocks:
                    torch.nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.02)
                x = torch.randn(1, 3, 32, 32, requires_grad=True)
                output = model(x, torch.tensor([0.5]), torch.tensor([0]))
                self.assertEqual(output.shape, x.shape)
                output.square().mean().backward()
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(model.blocks[0].attn.qkv.weight.grad.abs().sum().item(), 0)
                self.assertGreater(x.grad.abs().sum().item(), 0)
                self.assertFalse(any('freqs_' in name for name in model.state_dict()))
                model.load_state_dict(model.state_dict(), strict=True)


class ToyLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 1)
        self.ema_updates = 0

    def forward(self, images, targets):
        return (self.linear(images).squeeze(-1) - targets.float()).square().mean()

    def update_ema(self):
        self.ema_updates += 1


class FakeDDP(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.module = model
        self.in_no_sync = False
        self.unsynced_forwards = 0

    @contextlib.contextmanager
    def no_sync(self):
        self.in_no_sync = True
        try:
            yield
        finally:
            self.in_no_sync = False

    def forward(self, *args):
        if self.in_no_sync:
            self.unsynced_forwards += 1
        return self.module(*args)


class AccumulationTests(unittest.TestCase):
    def args(self, accum):
        return SimpleNamespace(accum_steps=accum, lr=0.01, warmup_epochs=0,
                               lr_schedule='constant', log_freq=100)

    def test_matches_large_batch_with_short_final_window_and_microbatch(self):
        torch.manual_seed(1)
        x = torch.randint(0, 256, (11, 3), dtype=torch.uint8)
        y = torch.arange(11) % 3
        dataset = TensorDataset(x, y)
        model = ToyLoss()
        expected = copy.deepcopy(model)
        wrapped = FakeDDP(model)
        optimizer = torch.optim.SGD(wrapped.parameters(), lr=0.01)
        reference_optimizer = torch.optim.SGD(expected.parameters(), lr=0.01)
        with contextlib.redirect_stdout(io.StringIO()):
            metrics = train_one_epoch(wrapped, model, DataLoader(dataset, batch_size=2),
                                      optimizer, 'cpu', 0, args=self.args(4))
            reference = train_one_epoch(expected, expected, DataLoader(dataset, batch_size=8),
                                        reference_optimizer, 'cpu', 0, args=self.args(1))
        for actual, wanted in zip(model.parameters(), expected.parameters()):
            torch.testing.assert_close(actual, wanted, atol=1e-7, rtol=1e-6)
        self.assertEqual(model.ema_updates, 2)
        self.assertEqual(wrapped.unsynced_forwards, 4)
        self.assertEqual(metrics['optimizer_steps'], 2)
        self.assertEqual([step['images'] for step in metrics['steps']], [8, 3])
        for actual, wanted in zip(metrics['steps'], reference['steps']):
            self.assertAlmostEqual(actual['loss'], wanted['loss'], places=6)

    def test_bounded_pilot_excludes_warmup(self):
        model = ToyLoss()
        data = DataLoader(TensorDataset(torch.zeros(30, 3), torch.zeros(30)), batch_size=2)
        with contextlib.redirect_stdout(io.StringIO()):
            result = train_one_epoch(model, model, data, torch.optim.SGD(model.parameters(), lr=0.01),
                                    'cpu', 0, args=self.args(2), max_optimizer_steps=3, warmup_steps=1)
        self.assertEqual((result['optimizer_steps'], result['measured_steps'], model.ema_updates), (3, 2, 3))
        self.assertGreater(result['training_pipeline_images_per_second'], 0)

    def test_epoch_lr_progression_matches_large_batches_with_short_tail(self):
        torch.manual_seed(3)
        dataset = TensorDataset(torch.randint(0, 256, (5, 3), dtype=torch.uint8), torch.arange(5) % 2)
        model = ToyLoss()
        expected = copy.deepcopy(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        reference_optimizer = torch.optim.SGD(expected.parameters(), lr=0.01)
        args = self.args(2)
        args.warmup_epochs, args.epochs, args.lr_schedule, args.min_lr = 1, 4, 'cosine', 1e-5
        reference_args = copy.copy(args)
        reference_args.accum_steps = 1
        with contextlib.redirect_stdout(io.StringIO()):
            for epoch in range(4):
                actual = train_one_epoch(model, model, DataLoader(dataset, batch_size=1), optimizer,
                                         'cpu', epoch, args=args)
                reference = train_one_epoch(expected, expected, DataLoader(dataset, batch_size=2),
                                            reference_optimizer, 'cpu', epoch, args=reference_args)
                self.assertEqual([record['schedule_epoch'] for record in actual['steps']],
                                 [epoch + index / 3 for index in range(3)])
                for a, b in zip(actual['steps'], reference['steps']):
                    self.assertAlmostEqual(a['lr'], b['lr'])
                for a, b in zip(model.parameters(), expected.parameters()):
                    torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-6)

    def test_pipeline_timer_includes_loader_forward_backward_optimizer_and_ema(self):
        clock = [0.0]
        def advance(seconds):
            clock[0] += seconds
        class TimedLoader:
            def __len__(self):
                return 2
            def __iter__(self):
                for _ in range(2):
                    advance(2)  # Data loading.
                    yield torch.zeros(2, 3), torch.zeros(2)
        model = ToyLoss()
        forward, ema = model.forward, model.update_ema
        def timed_forward(*args):
            advance(3)
            return forward(*args)
        def backward_hook(gradient):
            advance(5)
            return gradient
        def timed_ema():
            advance(11)
            ema()
        model.forward, model.update_ema = timed_forward, timed_ema
        model.linear.weight.register_hook(backward_hook)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        original_step = optimizer.step
        def timed_step(*args, **kwargs):
            advance(7)
            return original_step(*args, **kwargs)
        with (patch('engine_jit.time.perf_counter', side_effect=lambda: clock[0]),
              patch.object(optimizer, 'step', side_effect=timed_step),
              contextlib.redirect_stdout(io.StringIO())):
            result = train_one_epoch(model, model, TimedLoader(), optimizer, 'cpu', 0,
                                     args=self.args(2), max_optimizer_steps=1,
                                     step_callback=lambda record: advance(100))
        # 2*(load 2 + forward 3 + backward 5) + optimizer 7 + EMA 11.
        self.assertEqual(result['steps'][0]['training_pipeline_seconds'], 38)
        self.assertAlmostEqual(result['training_pipeline_images_per_second'], 4 / 38)
        self.assertIn('excludes setup and logging', result['timing_scope'])

    def test_invalid_accumulation_rejected(self):
        model = ToyLoss()
        data = DataLoader(TensorDataset(torch.zeros(2, 3), torch.zeros(2)), batch_size=1)
        with self.assertRaises(ValueError):
            train_one_epoch(model, model, data, torch.optim.SGD(model.parameters(), lr=0.01),
                            'cpu', 0, args=self.args(0))

    def test_parser_accumulation_default(self):
        self.assertEqual(get_args_parser().parse_args([]).accum_steps, 1)

    def test_effective_batch_and_learning_rate_scaling(self):
        args = get_args_parser().parse_args(['--batch_size', '8', '--accum_steps', '4'])
        self.assertEqual(effective_batch_size(args, 8), 256)
        self.assertEqual(args.blr * effective_batch_size(args, 8) / 256, 5e-5)

    def test_nonfinite_loss_never_updates_optimizer_or_ema(self):
        model = ToyLoss()
        before = copy.deepcopy(model.state_dict())
        data = DataLoader(TensorDataset(torch.zeros(2, 3), torch.full((2,), float('nan'))))
        with self.assertRaises(FloatingPointError):
            train_one_epoch(model, model, data, torch.optim.SGD(model.parameters(), lr=0.01),
                            'cpu', 0, args=self.args(1))
        self.assertEqual(model.ema_updates, 0)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name])


class SafetyTests(unittest.TestCase):
    def test_checkpoint_publication_never_replaces_files_or_symlinks(self):
        import tempfile
        from util.misc import save_checkpoint_exclusive
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / 'checkpoint-epoch000001.pth'
            save_checkpoint_exclusive({'weight': torch.tensor([1.0])}, checkpoint)
            before = checkpoint.read_bytes()
            with self.assertRaises(FileExistsError):
                save_checkpoint_exclusive({'weight': torch.tensor([2.0])}, checkpoint)
            self.assertEqual(checkpoint.read_bytes(), before)
            alias = root / 'alias.pth'
            alias.symlink_to(checkpoint)
            with self.assertRaises(FileExistsError):
                save_checkpoint_exclusive({}, alias)
            self.assertEqual(checkpoint.read_bytes(), before)
            self.assertEqual(list(root.glob('.checkpoint-*')), [])

    def test_failed_checkpoint_save_removes_only_its_temporary_file(self):
        import tempfile
        from util.misc import save_checkpoint_exclusive
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            previous = root / 'checkpoint-last.pth'
            previous.write_bytes(b'preserve')
            with patch('torch.save', side_effect=RuntimeError('serialization failure')), self.assertRaises(RuntimeError):
                save_checkpoint_exclusive({}, root / 'new.pth')
            self.assertEqual(list(root.iterdir()), [previous])
            self.assertEqual(previous.read_bytes(), b'preserve')

    def test_checkpoint_publication_race_does_not_replace_competing_file(self):
        import tempfile
        from util.misc import save_checkpoint_exclusive
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'new.pth'
            link = os.link
            def concurrent_writer(source, destination):
                Path(destination).write_bytes(b'competing writer fixture')
                return link(source, destination)
            with patch('util.misc.os.link', side_effect=concurrent_writer), self.assertRaises(FileExistsError):
                save_checkpoint_exclusive({}, path)
            self.assertEqual(path.read_bytes(), b'competing writer fixture')
            self.assertEqual(list(Path(temp).glob('.checkpoint-*')), [])

    def test_resume_is_explicit_and_missing_checkpoints_fail(self):
        import tempfile
        from main_jit import resolve_resume_checkpoint
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / 'checkpoint-last.pth'
            checkpoint.write_bytes(b'preserve')
            self.assertEqual(resolve_resume_checkpoint(temp), checkpoint)
            self.assertEqual(resolve_resume_checkpoint(str(checkpoint)), checkpoint)
            with self.assertRaises(FileNotFoundError):
                resolve_resume_checkpoint(str(Path(temp) / 'missing.pth'))
            self.assertEqual(checkpoint.read_bytes(), b'preserve')

    def test_training_rejects_existing_output_before_touching_checkpoint(self):
        import tempfile
        from main_jit import main
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / 'checkpoint-last.pth'
            checkpoint.write_bytes(b'preserve')
            args = get_args_parser().parse_args(['--output_dir', temp])
            args.distributed = False
            with (patch('main_jit.misc.init_distributed_mode'),
                  contextlib.redirect_stdout(io.StringIO()),
                  self.assertRaisesRegex(RuntimeError, 'must be new')):
                main(args)
            self.assertEqual(checkpoint.read_bytes(), b'preserve')
            self.assertEqual(list(Path(temp).iterdir()), [checkpoint])

    def test_pilot_gpu_guard_records_failure_in_unique_directories(self):
        import tempfile
        import json
        from tools.hpc_pilot import main
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            protected = root / 'protected/checkpoint-last.pth'
            protected.parent.mkdir()
            protected.write_bytes(b'protected')
            arguments = ['hpc_pilot.py', 'pilot', '--output-root', str(root / 'runs'),
                         '--protected-checkpoint', str(protected)]
            for _ in range(2):
                with (patch('sys.argv', arguments), patch('torch.cuda.is_available', return_value=False),
                      contextlib.redirect_stdout(io.StringIO()),
                      self.assertRaisesRegex(RuntimeError, 'CUDA GPU')):
                    main()
            outputs = list((root / 'runs').iterdir())
            self.assertEqual(len(outputs), 2)
            for output in outputs:
                self.assertTrue((output / 'arguments.json').is_file())
                self.assertEqual(json.loads((output / 'failure.json').read_text())['type'], 'RuntimeError')
                self.assertEqual(list(output.glob('*.pth')), [])
            self.assertEqual(protected.read_bytes(), b'protected')

    def test_exclusive_json_and_readonly_checkpoint_hash(self):
        import tempfile
        from tools.hpc_pilot import checkpoint_hash, write_json
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / 'checkpoint-last.pth'
            checkpoint.write_bytes(b'protected checkpoint fixture')
            before = checkpoint.read_bytes()
            self.assertEqual(checkpoint_hash(checkpoint), checkpoint_hash(checkpoint))
            self.assertEqual(checkpoint.read_bytes(), before)
            path = root / 'summary.json'
            write_json(path, {'first': True})
            with self.assertRaises(FileExistsError):
                write_json(path, {'first': False})


class PreflightTests(unittest.TestCase):
    def test_validation_class_mapping_matches_actual_imagefolders(self):
        import tempfile
        from PIL import Image
        from torchvision.datasets import ImageFolder
        from tools.hpc_pilot import verify_class_mappings
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for split in ('train', 'val'):
                for name in ('n00000001', 'n00000002'):
                    folder = root / split / name
                    folder.mkdir(parents=True)
                    Image.new('RGB', (8, 8)).save(folder / 'fixture.png')
            train, val = ImageFolder(root / 'train'), ImageFolder(root / 'val')
            report = verify_class_mappings(train, val, expected_classes=2)
            self.assertTrue(report['class_mapping_match'])
            self.assertEqual(report['train_class_to_idx'], report['val_class_to_idx'])
            with self.assertRaisesRegex(RuntimeError, 'separate'):
                verify_class_mappings(train, train, expected_classes=2)
            with self.assertRaisesRegex(RuntimeError, '1000 classes'):
                verify_class_mappings(train, val)
            val.class_to_idx = {'n00000001': 1, 'n00000002': 0}
            with self.assertRaisesRegex(RuntimeError, 'matching'):
                verify_class_mappings(train, val, expected_classes=2)

    def test_checkpoint_verified_after_failure_and_expected_digest_checked(self):
        import tempfile
        from tools.hpc_pilot import checkpoint_hash, verify_checkpoint_integrity
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / 'fixture.pth'
            checkpoint.write_bytes(b'protected fixture')
            output = root / 'first'
            output.mkdir()
            digest = checkpoint_hash(checkpoint)['sha256']
            with self.assertRaisesRegex(RuntimeError, 'pilot failure'):
                with verify_checkpoint_integrity(checkpoint, output, digest):
                    raise RuntimeError('pilot failure')
            self.assertEqual((output / 'protected-checkpoint-before.json').read_bytes(),
                             (output / 'protected-checkpoint-after.json').read_bytes())
            second = root / 'second'
            second.mkdir()
            with self.assertRaisesRegex(RuntimeError, 'trusted digest'):
                with verify_checkpoint_integrity(checkpoint, second, '0' * 64):
                    self.fail('Wrong trusted digest must stop the workload')
            self.assertEqual(checkpoint.read_bytes(), b'protected fixture')

    def test_checkpoint_guard_detects_an_external_change(self):
        import tempfile
        from tools.hpc_pilot import verify_checkpoint_integrity
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / 'fixture.pth'
            checkpoint.write_bytes(b'before fixture')
            with self.assertRaisesRegex(RuntimeError, 'changed during'):
                with verify_checkpoint_integrity(checkpoint, root):
                    # Simulate an external writer using a disposable test fixture only.
                    checkpoint.write_bytes(b'changed fixture')


if __name__ == '__main__':
    unittest.main()
