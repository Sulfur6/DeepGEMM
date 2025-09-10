import copy
import random
import time
import torch

import json
import re

import deep_gemm
from deep_gemm.testing import (
    bench, bench_kineto,
    calc_diff, count_bytes,
    check_signal,
)

from generators import (
    KernelType, get_ue8m0_usage,
    enumerate_normal, enumerate_m_grouped_contiguous, enumerate_m_grouped_masked, enumerate_k_grouped_contiguous,
    generate_normal, generate_m_grouped_contiguous, generate_m_grouped_masked, generate_k_grouped_contiguous
)

def parse_fp8_gemm_info_from_trace(trace_path: str):
    """
    Parses a trace file to find the 'sm90_fp8_gemm_1d2d_impl' kernel
    and extracts its BLOCK_N template parameter and gridDim.x.

    Args:
        trace_path (str): Path to the trace.json file.

    Returns:
        A tuple (block_n, grid_dim_x) or (None, None) if not found.
    """
    try:
        with open(trace_path, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None, None

    for event in data.get('traceEvents', []):
        if event.get('cat') == 'kernel' and 'sm90_fp8_gemm_1d2d_impl' in event.get('name', ''):
            kernel_name = event.get('name', '')
            args = event.get('args', {})
            
            # Extract gridDim.x
            grid_dim_x = args.get('grid', [None])[0]

            # Extract BLOCK_N from the 7th template argument (index 6)
            block_n = None
            match = re.search(r'<(.+)>', kernel_name)
            if match:
                content = match.group(1)
                template_args = []
                balance = 0
                current_arg = ""
                for char in content:
                    if char == '<': balance += 1
                    elif char == '>': balance -= 1
                    if char == ',' and balance == 0:
                        template_args.append(current_arg.strip())
                        current_arg = ""
                    else:
                        current_arg += char
                if current_arg:
                    template_args.append(current_arg.strip())
                
                if len(template_args) > 5:
                    try:
                        block_n = int(template_args[5].replace('u', ''))
                    except (ValueError, IndexError):
                        block_n = None

            # Return the first one we find
            if block_n is not None and grid_dim_x is not None:
                return block_n, grid_dim_x
                
    return None, None


def test_m_grouped_gemm_masked(max_block_n=256) -> None:
    print('Testing m-grouped masked GEMM with max_block_n={}:'.format(max_block_n))

    # TODO: when the actual `m` is greater than `expected_m_per_group`, efficiency may significantly decrease.
    for kernel_type, enable_overlap, num_groups, max_m, expected_m_per_group, n, k in enumerate_m_grouped_masked():
        kernel_opt = f'1D1D' if kernel_type.is_1d1d() else '1D2D'
        use_ue8m0 = get_ue8m0_usage(kernel_type)
        disable_ue8m0_cast = not use_ue8m0

        # Test correctness (this part remains unchanged)
        for i in range(10):
            a, b, masked_m, d, ref_d, signal = generate_m_grouped_masked(num_groups, max_m, expected_m_per_group, n, k, use_ue8m0=use_ue8m0, enable_overlap=enable_overlap)
            result = deep_gemm.m_grouped_fp8_gemm_nt_masked(a, b, d, masked_m, expected_m_per_group, disable_ue8m0_cast=disable_ue8m0_cast, enable_overlap=enable_overlap, signal=signal, max_block_n=max_block_n)

            if enable_overlap:
                block_m, threshold = result
                check_signal(num_groups, max_m, block_m, threshold, signal, masked_m)

            for j in range(num_groups):
                diff = calc_diff(d[j, :masked_m[j].item()], ref_d[j, :masked_m[j].item()])
                assert diff < 0.001, f'{max_m=}, {n=}, {k=}, {j=}, masked_m={masked_m[j]}, {kernel_opt}, {num_groups=}, {diff:.5f}'

        # Construct full cases
        a, b, masked_m, d, ref_d, signal = generate_m_grouped_masked(num_groups, max_m, expected_m_per_group, n, k, use_ue8m0=use_ue8m0, enable_overlap=enable_overlap)

        # noinspection PyShadowingNames
        def test_func():
            if not enable_overlap:
                deep_gemm.m_grouped_fp8_gemm_nt_masked(a, b, d, masked_m, expected_m_per_group, disable_ue8m0_cast=disable_ue8m0_cast, enable_overlap=enable_overlap, signal=signal, max_block_n=max_block_n)
            else:
                origin_sms = deep_gemm.get_num_sms()
                deep_gemm.set_num_sms(origin_sms - 3)
                deep_gemm.m_grouped_fp8_gemm_nt_masked(a, b, d, masked_m, expected_m_per_group, disable_ue8m0_cast=disable_ue8m0_cast, enable_overlap=enable_overlap, signal=signal, max_block_n=max_block_n)
                deep_gemm.set_num_sms(origin_sms)

        # --- MODIFICATION STARTS HERE ---

        # Test performance with fixed shapes
        valid_m = masked_m.sum().item()
        
        # 1. Run benchmark and generate trace.json (bench_kineto is NOT modified)
        trace_file = 'trace.json'
        t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True, trace_path=trace_file)
        
        # 2. Parse the generated trace file to get extra info
        block_n_info, num_sms_info = parse_fp8_gemm_info_from_trace(trace_file)

        # 3. Format the performance string
        perf_string = (f' > Perf ({num_groups=}, expected_m_per_group={expected_m_per_group:4}, n={n:4}, k={k:4}, {kernel_opt}, enable_overlap={enable_overlap}): '
                       f'{t * 1e6:4.0f} us | '
                       f'{2 * valid_m * n * k / t / 1e12:4.0f} TFLOPS | '
                       f'{(count_bytes(a, d) * valid_m / (max_m * num_groups) + count_bytes(b)) / 1e9 / t:4.0f} GB/s')

        # 4. Append the extra info to the end of the string
        extra_info_string = ""
        if block_n_info is not None:
            extra_info_string += f' | BLOCK_N={block_n_info:<4}'
        if num_sms_info is not None:
            extra_info_string += f' | NUM_SMS={num_sms_info:<4}' # Renamed from grid.x
        
        print(perf_string + extra_info_string)
        
        # --- MODIFICATION ENDS HERE ---

    print()


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')

    test_m_grouped_gemm_masked()
    test_m_grouped_gemm_masked(max_block_n=160)
