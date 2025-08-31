# PyTorch has its own NVRTC, which may have a lower version than the system
# So try to disable PyTorch's NVRTC, or import NVRTC before PyTorch
import cuda.bindings.nvrtc as nvrtc
print(f'NVRTC version: {nvrtc.nvrtcVersion()[1:]}')

import random
import torch
from typing import List, Tuple

import deep_gemm
from deep_gemm import bench_kineto, calc_diff, ceil_div, get_col_major_tma_aligned_tensor
from deep_gemm.jit_kernels.utils import get_m_alignment_for_contiguous_layout


def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    pad_size = (128 - (n % 128)) % 128
    x = torch.nn.functional.pad(x, (0, pad_size), value=0) if pad_size > 0 else x
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    fp8_data = (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn)
    return fp8_data.view(m, n + pad_size)[:, :n], (x_amax / 448.0).view(m, -1)


def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros((ceil_div(m, 128) * 128, ceil_div(n, 128) * 128), dtype=x.dtype, device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / 448.0).view(x_view.size(0), x_view.size(2))


def construct(m: int, k: int, n: int) -> \
        Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
    x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
    out = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    ref_out = x @ y.t()

    x_fp8, y_fp8 = per_token_cast_to_fp8(x), per_block_cast_to_fp8(y)
    # Transpose earlier so that the testing will not trigger transposing kernels
    x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))
    return x_fp8, y_fp8, out, ref_out


def construct_masked_grouped(num_groups: int, max_m: int, expected_m_per_group: int, k: int, n: int) -> \
        Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.randn((num_groups, max_m, k), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((num_groups, n, k), device='cuda', dtype=torch.bfloat16)
    out = torch.empty((num_groups, max_m, n), device='cuda', dtype=torch.bfloat16)
    ref_out = torch.einsum('gmk,gnk->gmn', x, y)

    assert max_m % 4 == 0, f'TMA alignment error: {max_m}'
    x_fp8 = (torch.empty_like(x, dtype=torch.float8_e4m3fn), torch.empty((num_groups, max_m, k // 128), device='cuda', dtype=torch.float))
    y_fp8 = (torch.empty_like(y, dtype=torch.float8_e4m3fn), torch.empty((num_groups, ceil_div(n, 128), k // 128), device='cuda', dtype=torch.float))
    for i in range(num_groups):
        x_fp8[0][i], x_fp8[1][i] = per_token_cast_to_fp8(x[i])
        y_fp8[0][i], y_fp8[1][i] = per_block_cast_to_fp8(y[i])

    # Transpose earlier so that the testing will not trigger transposing kernels
    x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))

    # Construct mask
    masked_m = torch.empty((num_groups, ), device='cuda', dtype=torch.int)
    for j in range(num_groups):
        masked_m[j] = int(expected_m_per_group * random.uniform(0.7, 1.3))
    assert masked_m.amax().item() <= max_m
    return x_fp8, y_fp8, masked_m, out, ref_out

def test_m_grouped_gemm_masked() -> None:
    print('Testing grouped masked GEMM:')
    max_m = 2048

    for num_groups, expected_m_per_group in ((16, 32), (16, 64)):
        for k, n in ((2048, 7168), ):
            # Test correctness
            for i in range(2):
                x_fp8, y_fp8, masked_m, out, ref_out = construct_masked_grouped(num_groups, max_m, expected_m_per_group, k, n)
                deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(x_fp8, y_fp8, out, masked_m, expected_m_per_group)
                for j in range(num_groups):
                    diff = calc_diff(out[j, :masked_m[j].item()], ref_out[j, :masked_m[j].item()])
                    assert diff < 0.001, f'{expected_m_per_group=}, {k=}, {n=}, {j=}, masked_m={masked_m[j]}, {num_groups=}, {diff:.5f}'
    print()

def test_m_grouped_gemm_signal() -> None:
    print('Testing grouped signal GEMM:')
    max_m = 2048

    for num_groups, expected_m_per_group in ((16, 32), (16, 64)):
        for k, n in ((2048, 7168), ):
            # Test correctness
            for i in range(2):
                x_fp8, y_fp8, masked_m, out, ref_out = construct_masked_grouped(num_groups, max_m, expected_m_per_group, k, n)
                deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_signal(x_fp8, y_fp8, out, masked_m, expected_m_per_group)
                for j in range(num_groups):
                    diff = calc_diff(out[j, :masked_m[j].item()], ref_out[j, :masked_m[j].item()])
                    assert diff < 0.001, f'{expected_m_per_group=}, {k=}, {n=}, {j=}, masked_m={masked_m[j]}, {num_groups=}, {diff:.5f}'
    print()