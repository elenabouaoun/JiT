"""Strict CUDA FlashAttention validation, shared by preflight and pilots."""
from contextlib import contextmanager
import math

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from model_jit import scaled_dot_product_attention


@contextmanager
def flash_attention_only():
    if (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported() or
            not torch.backends.cuda.is_flash_attention_available()):
        raise RuntimeError('BF16 CUDA FlashAttention is required; pilots never fall back to dense attention')
    try:
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            yield
    except RuntimeError as error:
        if isinstance(error, torch.cuda.OutOfMemoryError):
            raise
        if any(message in str(error).lower() for message in ('no available kernel', 'no viable backend', 'no suitable kernel')):
            raise RuntimeError('FlashAttention cannot execute this workload; no dense fallback is permitted') from error
        raise


def validate_bf16_flash_attention(lengths=(96, 288, 1056, 4128)):
    """Compare full-length FlashAttention with FP32 reference on 16 query rows.

    Sampling query rows avoids constructing a dense 4128x4128 reference matrix.
    Loss uses those same rows, so full Q/K/V gradients are still comparable.
    RNG state is restored, keeping pilots' training randomness unchanged.
    """
    results = []
    with flash_attention_only(), torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(17)
        for length in lengths:
            inputs = [torch.randn(1, 12, length, 64, device='cuda', dtype=torch.bfloat16,
                                  requires_grad=True) for _ in range(3)]
            references = [tensor.detach().float().requires_grad_() for tensor in inputs]
            rows = torch.linspace(0, length - 1, min(16, length), device='cuda').long()
            actual = scaled_dot_product_attention(*inputs).index_select(-2, rows)
            q, k, v = references
            scores = q.index_select(-2, rows) @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
            expected = scores.softmax(dim=-1) @ v
            output_error = ((actual.float() - expected).norm() / expected.norm().clamp_min(1e-12)).item()
            upstream = torch.randn_like(actual)
            actual.backward(upstream)
            expected.backward(upstream.float())
            gradient_errors = []
            for tensor, reference in zip(inputs, references):
                if not torch.isfinite(tensor.grad).all():
                    raise RuntimeError(f'Nonfinite FlashAttention gradient at sequence length {length}')
                gradient_errors.append(((tensor.grad.float() - reference.grad).norm() /
                                        reference.grad.norm().clamp_min(1e-12)).item())
            if (not math.isfinite(output_error) or output_error > 0.02 or
                    any(not math.isfinite(error) or error > 0.03 for error in gradient_errors)):
                raise RuntimeError(f'BF16 FlashAttention numerical validation failed at length {length}: '
                                   f'output={output_error}, Q/K/V gradients={gradient_errors}')
            results.append(dict(sequence_length=length, dtype='bfloat16', backend='FLASH_ATTENTION only',
                                output_relative_l2_error=output_error,
                                qkv_gradient_relative_l2_errors=gradient_errors))
    torch.cuda.synchronize()
    return results
