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
        torch.manual_seed(7)
        inputs = [torch.randn(2, 12, 96, 64, device='cuda', dtype=torch.bfloat16,
                              requires_grad=True) for _ in range(3)]
        refs = [tensor.detach().float().requires_grad_() for tensor in inputs]
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            actual = scaled_dot_product_attention(*inputs)
        expected = reference_attention(*refs)
        error = (actual.float() - expected).norm() / expected.norm()
        self.assertLess(error.item(), 0.02)
        upstream = torch.randn_like(actual)
        actual.backward(upstream)
        expected.backward(upstream.float())
        for actual_input, reference_input in zip(inputs, refs):
            self.assertTrue(torch.isfinite(actual_input.grad).all())
            relative = (actual_input.grad.float() - reference_input.grad).norm() / reference_input.grad.norm()
            self.assertLess(relative.item(), 0.03)

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
        self.assertGreater(result['images_per_second'], 0)

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


if __name__ == '__main__':
    unittest.main()
