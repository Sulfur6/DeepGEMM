from dataclasses import dataclass
from typing import List, Tuple


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def align(a: int, b: int) -> int:
    return ceil_div(a, b) * b


@dataclass
class MulticastConfig:
    """Multicast configuration"""
    num_multicast: int
    is_multicast_on_a: bool


@dataclass
class BlockConfig:
    """Block size configuration"""
    block_m: int
    block_n: int
    block_k: int
    num_stages: int
    num_sms: int
    multicast_config: MulticastConfig


class SM90ArchSpec:

    @staticmethod
    def get_block_n_candidates(cd_dtype_is_fp32: bool, max_block_n: int) -> List[int]:
        # Avoid bank conflicts for FP32 output
        start = 8 if cd_dtype_is_fp32 else 16
        candidates = []
        for i in range(start, max_block_n + 1, 16):
            candidates.append(i)
        return candidates
    
    @staticmethod
    def is_block_size_legal(cd_dtype_is_fp32: bool, 
                       ab_dtype_size: int,
                       block_m: int, 
                       block_n: int, 
                       block_k: int,
                       kernel_type: str = "Kernel1D2D") -> bool:
        # SM90 FP32 output does not support `block_m == 256`
        if cd_dtype_is_fp32 and block_m == 256:
            return False
    
        # Avoid large C/D shared memory for FP32 output
        # Ensure `num_stages >= 4` (for 1D1D Kernel), `num_stages >= 3` (for No SF kernel)
        if block_n > 128 and cd_dtype_is_fp32:
            if kernel_type == "Kernel1D1D" and block_n > 152:
                return False
            if kernel_type == "KernelNoSF" and block_n > 200:
                return False
        
        # Too many scaling factors in a single block: for Kernel1D2D
        # Only allow specific block_n values when block_n > 128
        if block_n > 128 and kernel_type == "Kernel1D2D" and (block_n != 144 and block_n != 160 and block_n != 192):
            return False
    
        # Avoid bank conflicts for FP32 output
        if cd_dtype_is_fp32 and block_n % 16 == 0:
            return False
    
        # The block sizes cannot be too large (for enough registers), so at least one dim less than 128
        return block_m <= 128 or block_n <= 128
    

def get_best_config(
    m: int,
    n: int,
    k: int,
    num_sms: int,
    ab_dtype_size: int = 1,  # 1 for FP8, 2 for BF16
    cd_dtype_is_fp32: bool = False,  # Whether output is FP32
    max_block_n: int = 256
) -> BlockConfig:

    # Select M/N block sizes
    block_ms = [64, 128, 256]
    block_ns = SM90ArchSpec.get_block_n_candidates(cd_dtype_is_fp32, max_block_n)
    
    # Apply max_block_n constraint
    block_ns = [bn for bn in block_ns if bn <= max_block_n]
    
    # K block size is selected in a fixed manner
    block_k = 128 // ab_dtype_size  # FP8: 128, BF16: 64
    
    # Some util functions
    def get_num_blocks(block_m: int, block_n: int) -> int:
        return ceil_div(m, block_m) * ceil_div(n, block_n)
    
    def get_num_waves(block_m: int, block_n: int) -> int:
        return ceil_div(get_num_blocks(block_m, block_n), num_sms)
    
    def get_last_wave_util(block_m: int, block_n: int) -> int:
        num_last_blocks = get_num_blocks(block_m, block_n) % num_sms
        return num_sms if num_last_blocks == 0 else num_last_blocks
    
    # Decide block sizes by waves
    best_block_m = 0
    best_block_n = 0
    best_num_waves = 0
    best_last_util = 0
    
    for block_m in block_ms:
        for block_n in block_ns:
            num_waves = get_num_waves(block_m, block_n)
            last_util = get_last_wave_util(block_m, block_n)
            
            # Check if block size is legal
            if not SM90ArchSpec.is_block_size_legal(
                cd_dtype_is_fp32, ab_dtype_size, block_m, block_n, block_k):
                continue
            
            success = False
            
            if best_block_m == 0 or best_block_n == 0:
                success = True
            elif num_waves < best_num_waves:
                success = True
            elif num_waves == best_num_waves:
                # Check last wave utilization
                if last_util > best_last_util:
                    success = True
                elif last_util == best_last_util:
                    # Case 1: same `block_m`, smaller `block_n` (wasted)
                    if block_m == best_block_m and block_n < best_block_n:
                        success = True
                    # Case 2: same `block_n`, smaller `block_m` (wasted)
                    elif block_n == best_block_n and block_m < best_block_m:
                        success = True
                    # Case 3: different for both `block_m` and `block_n`, larger `block_n` is better
                    # NOTES: don't pick `block_m/block_n` larger than shape `m/n` in this case
                    elif (block_m != best_block_m and block_n > best_block_n 
                          and block_n <= n and block_m <= m):
                        success = True
            
            # Replace with the new config if successful
            if success:
                best_block_m = block_m
                best_block_n = block_n
                best_num_waves = num_waves
                best_last_util = last_util
    
    assert best_block_m > 0 and best_block_n > 0, "No valid block configuration found"
    
    # Decide the number of TMA multicasts and whether broadcast on A
    def is_multicast_legal_on_a() -> bool:
        """Check if multicast on A is legal"""
        num_m_blocks = ceil_div(m, best_block_m)
        # Need at least 2 blocks for multicast
        return num_m_blocks >= 2
    
    def is_multicast_legal_on_b() -> bool:
        """Check if multicast on B is legal"""
        num_n_blocks = ceil_div(n, best_block_n)
        return num_n_blocks >= 2
    
    is_legal_on_a = is_multicast_legal_on_a()
    is_legal_on_b = is_multicast_legal_on_b()
    
    best_multicast_config = MulticastConfig(num_multicast=1, is_multicast_on_a=False)
    
    is_legal = [is_legal_on_b, is_legal_on_a]
    order = [False, True]
    if best_block_m > best_block_n:
        order = [True, False]
    
    for is_multicast_on_a in order:
        if m >= 512 and is_legal[int(is_multicast_on_a)]:
            best_multicast_config = MulticastConfig(
                num_multicast=2,
                is_multicast_on_a=is_multicast_on_a
            )
            break
    
    # Always pick the largest number of stage
    # Simplified: use heuristic based on dtype
    if ab_dtype_size == 1:  # FP8
        best_num_stages = 7
    else:  # BF16
        best_num_stages = 5
    
    # Recompute the minimal number of SMs required
    # NOTES: less L2 cache usage and less GPU frequency drop
    num_min_sms = ceil_div(
        ceil_div(m, best_block_m) * ceil_div(n, best_block_n),
        best_num_waves
    )
    num_min_sms = align(num_min_sms, best_multicast_config.num_multicast)
    num_min_sms = min(num_min_sms, num_sms)
    
    return BlockConfig(
        block_m=best_block_m,
        block_n=best_block_n,
        block_k=block_k,
        num_stages=best_num_stages,
        num_sms=num_min_sms,
        multicast_config=best_multicast_config
    )


def get_num_1d_blocks_per_group(BLOCK_M: int, BLOCK_N: int, kNumSMs: int, kIsMulticastOnA: bool) -> int:
    # same heuristic as scheduler.cuh for candidates {8, 16}
    num_best_blocks = 0
    min_usage = 2**31 - 1
    for candidate in (8, 16):
        if kIsMulticastOnA:
            # grouping on N
            usage = candidate * BLOCK_N + ceil_div(kNumSMs, candidate) * BLOCK_M
        else:
            # grouping on M
            usage = candidate * BLOCK_M + ceil_div(kNumSMs, candidate) * BLOCK_N
        if usage < min_usage:
            min_usage = usage
            num_best_blocks = candidate
    return num_best_blocks


@dataclass
class SchedulerNormalSim:
    shape_m: int
    shape_n: int
    BLOCK_M: int
    BLOCK_N: int
    kNumSMs: int
    kNumMulticast: int          # kNumTMAMulticast
    kIsMulticastOnA: bool       # kIsTMAMulticastOnA
    sm90_odd_fix: bool = True   # __CUDA_ARCH__ < 1000 branch enabled on SM90

    def __post_init__(self):
        self.num_m_blocks = ceil_div(self.shape_m, self.BLOCK_M)
        self.num_n_blocks = ceil_div(self.shape_n, self.BLOCK_N)
        self.num_blocks = self.num_m_blocks * self.num_n_blocks
        self.kNum1DBlocksPerGroup = get_num_1d_blocks_per_group(
            self.BLOCK_M, self.BLOCK_N, self.kNumSMs, self.kIsMulticastOnA
        )

        if self.kNum1DBlocksPerGroup % self.kNumMulticast != 0:
            raise ValueError("kNum1DBlocksPerGroup must be divisible by kNumMulticast")

    def get_swizzled_block_idx(self, block_idx: int) -> Tuple[int, int]:
        primary_num_blocks   = self.num_n_blocks if self.kIsMulticastOnA else self.num_m_blocks
        secondary_num_blocks = self.num_m_blocks if self.kIsMulticastOnA else self.num_n_blocks

        num_blocks_per_group = secondary_num_blocks * self.kNum1DBlocksPerGroup

        group_idx = block_idx // num_blocks_per_group
        first_block_idx = group_idx * self.kNum1DBlocksPerGroup
        in_group_idx = block_idx % num_blocks_per_group

        num_blocks_in_group = min(self.kNum1DBlocksPerGroup, primary_num_blocks - first_block_idx)

        # SM90 odd fix: when multicast>1 and group primary count is odd, split into (even part) + (tail 1)
        if self.sm90_odd_fix and self.kNumMulticast > 1 and (num_blocks_in_group % 2 == 1):
            # same logic as scheduler.cuh
            if in_group_idx < ((num_blocks_in_group ^ 1) * secondary_num_blocks):
                num_blocks_in_group = (num_blocks_in_group ^ 1)
            else:
                in_group_idx = in_group_idx - ((num_blocks_in_group ^ 1) * secondary_num_blocks)
                first_block_idx += (num_blocks_in_group ^ 1)
                num_blocks_in_group = 1

        # convert to (m,n)
        if self.kIsMulticastOnA:
            m_block_idx = in_group_idx // num_blocks_in_group
            n_block_idx = first_block_idx + (in_group_idx % num_blocks_in_group)
        else:
            m_block_idx = first_block_idx + (in_group_idx % num_blocks_in_group)
            n_block_idx = in_group_idx // num_blocks_in_group

        return m_block_idx, n_block_idx


if __name__ == "__main__":
    shape_m, shape_n, shape_k = 4096, 7168, 2048
    kNumSMs = 70

    blockconfig = get_best_config(shape_m, shape_n, shape_k, kNumSMs)
    BLOCK_M, BLOCK_N, BLOCK_K = blockconfig.block_m, blockconfig.block_n, blockconfig.block_k
    kNumMulticast = blockconfig.multicast_config.num_multicast
    kIsMulticastOnA = blockconfig.multicast_config.is_multicast_on_a

    scheduler = SchedulerNormalSim(
        shape_m=shape_m, shape_n=shape_n,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        kNumSMs=kNumSMs,
        kNumMulticast=kNumMulticast,
        kIsMulticastOnA=kIsMulticastOnA
    )

    for tile_id in range(scheduler.num_blocks):
        pid_m, pid_n = scheduler.get_swizzled_block_idx(tile_id)
        offset = pid_m * scheduler.num_n_blocks + pid_n
        print(tile_id, pid_m, pid_n, offset)
