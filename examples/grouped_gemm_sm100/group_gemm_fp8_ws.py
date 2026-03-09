# Warp-specialized FP8 Grouped GEMM on SM100

import torch
import tilelang
import tilelang.language as T
from tilelang.utils.tensor import map_torch_type
from tilelang.profiler import do_bench


@tilelang.jit
def _group_gemm_fp8_kernel(
    A,
    B,
    offsets,
    pid_m_to_eid,
    pid_m_to_local,
    block_M,
    block_N,
    block_K,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
    use_tma_store=True,
):
    M_total, N, K, E, E1, MB = T.const("M_total, N, K, E, E1, MB")
    A: T.Tensor[[M_total, K], in_dtype]
    B: T.Tensor[[E, N, K], in_dtype]
    offsets: T.Tensor[[E1], "int32"]
    pid_m_to_eid: T.Tensor[[MB], "int32"]
    pid_m_to_local: T.Tensor[[MB], "int32"]
    C = T.empty((M_total, N), out_dtype)

    n_blocks = T.ceildiv(N, block_N)
    k_blocks = T.ceildiv(K, block_K)

    with T.Kernel(n_blocks, MB, threads=128) as (pid_n, pid_m):
        A_shared = T.alloc_shared((num_stages, block_M, block_K), in_dtype)
        B_shared = T.alloc_shared((num_stages, block_N, block_K), in_dtype)
        C_tmem = T.alloc_tmem([block_M, block_N], accum_dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
        C_shared = T.alloc_shared((block_M, block_N), out_dtype)
        C_local_cast = T.alloc_fragment((block_M, block_N), out_dtype)
        loaded = T.alloc_barrier([32] * num_stages)
        consumed = T.alloc_barrier([1] * num_stages)
        tmem_full = T.alloc_barrier([1])
        local_pid_m = T.alloc_var(T.int32)

        eid = pid_m_to_eid[pid_m]
        local_pid_m = pid_m_to_local[pid_m]
        tx = T.get_thread_binding()

        start_m = offsets[eid]
        end_m = offsets[eid + 1]

        # Swizzle M/N tile mapping for better L2 locality
        group_size_m = 8
        num_pid_m = (end_m - start_m + block_M - 1) // block_M
        linear_id = local_pid_m * n_blocks + pid_n
        num_pid_in_group = group_size_m * n_blocks
        group_id = linear_id // num_pid_in_group
        first_pid_m = group_id * group_size_m
        group_m = T.min(num_pid_m - first_pid_m, group_size_m)
        local_pid_m = first_pid_m + (linear_id % num_pid_in_group) % group_m
        pid_n = (linear_id % num_pid_in_group) // group_m

        tile_m = start_m + local_pid_m * block_M

        if tx < 32:  # warp 0: TMA loads
            for k in T.serial(k_blocks):
                stage = k % num_stages
                T.mbarrier_wait_parity(consumed[stage], ((k // num_stages) & 1) ^ 1)
                T.copy(
                    A[tile_m : tile_m + block_M, k * block_K : (k + 1) * block_K],
                    A_shared[stage, :, :],
                )
                T.copy(
                    B[eid, pid_n * block_N : (pid_n + 1) * block_N, k * block_K : (k + 1) * block_K],
                    B_shared[stage, :, :],
                )
                T.mbarrier_arrive(loaded[stage])
        elif tx < 64:  # warp 1: tcgen5 MMA
            for k in T.serial(k_blocks):
                stage = k % num_stages
                T.mbarrier_wait_parity(loaded[stage], (k // num_stages) & 1)
                T.gemm(
                    A_shared[stage, :, :],
                    B_shared[stage, :, :],
                    C_tmem,
                    transpose_B=True,
                    mbar=consumed[stage],
                    wg_wait=-1,
                    clear_accum=k == 0,
                )
            T.tcgen05_mma_arrive(tmem_full)

        T.mbarrier_wait_parity(tmem_full, 0)
        T.sync_threads()
        T.copy(C_tmem, C_local)
        if tile_m + block_M <= end_m:
            if use_tma_store:
                T.copy(C_local, C_shared)
                T.copy(C_shared, C[tile_m, pid_n * block_N])
            else:
                T.copy(C_local, C_local_cast)
                T.copy(C_local_cast, C[tile_m, pid_n * block_N])
        else:
            T.copy(C_local, C_local_cast)
            actual_rows = end_m - tile_m
            for i, j in T.Parallel(block_M, block_N):
                if i < actual_rows and pid_n * block_N + j < N:
                    C[tile_m + i, pid_n * block_N + j] = C_local_cast[i, j]

    return C


def group_gemm_fp8(
    A: torch.Tensor,    # [M_total, K] fp8
    B: torch.Tensor,    # [E, N, K] fp8
    cnt: torch.Tensor,  # [E+1] cumulative offsets
):
    block_M, block_N, block_K = 128, 256, 128
    in_dtype, out_dtype, accum_dtype = T.float8_e4m3fn, T.bfloat16, T.float32
    num_stages = 4

    expert_m_blocks = (cnt[1:] - cnt[:-1] + block_M - 1) // block_M
    eid_base = torch.arange(expert_m_blocks.numel(), device=cnt.device, dtype=torch.int32)
    pid_m_to_eid = torch.repeat_interleave(eid_base, expert_m_blocks)
    block_cumsum = torch.zeros(eid_base.numel() + 1, device=cnt.device, dtype=torch.int32)
    block_cumsum[1:] = expert_m_blocks.cumsum(0)
    pid_m_to_local = torch.arange(pid_m_to_eid.numel(), device=cnt.device, dtype=torch.int32) - block_cumsum[pid_m_to_eid]

    return _group_gemm_fp8_kernel(
        A, B, cnt, pid_m_to_eid, pid_m_to_local,
        block_M, block_N, block_K,
        in_dtype, out_dtype, accum_dtype, num_stages,
    )


def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def main():
    E = 16
    N, K = 1024, 4096
    tokens_per_expert = 4096
    M_total = E * tokens_per_expert

    tokens_per_expert_vec = torch.full((E,), tokens_per_expert, dtype=torch.int32, device="cuda")
    offsets_noise = torch.tensor([-3, -2, -1, 0, 1, 2, 3, 0, -3, -2, -1, 0, 1, 2, 3, 0], dtype=torch.int32, device="cuda")
    tokens_per_expert_vec = tokens_per_expert_vec + offsets_noise
    tokens_per_expert_vec[-1] = tokens_per_expert_vec[-1] - offsets_noise.sum()

    torch_fp8_dtype = map_torch_type(T.float8_e4m3fn)
    A = torch.randn(M_total, K, device="cuda", dtype=torch.float16).to(torch_fp8_dtype)
    B = torch.randn(E, N, K, device="cuda", dtype=torch.float16).to(torch_fp8_dtype)
    cnt = torch.cat((
        torch.tensor([0], device="cuda", dtype=torch.int32),
        torch.cumsum(tokens_per_expert_vec, dim=0, dtype=torch.int32),
    ))

    # --- Correctness ---
    tl_c = group_gemm_fp8(A, B, cnt)

    ref_c = torch.zeros(M_total, N, device="cuda", dtype=torch.float32)
    B_kn = B.transpose(1, 2).contiguous()
    for i in range(E):
        s, e = cnt[i].item(), cnt[i + 1].item()
        ref_c[s:e] = A[s:e].float() @ B_kn[i].float()

    diff = calc_diff(tl_c.float(), ref_c)
    print(f"diff = {diff}")
    print("Correctness check passed. ✅" if diff < 1e-2 else f"Correctness check FAILED ❌ (diff={diff})")

    # --- Performance ---
    total_flops = 2 * M_total * N * K
    tl_latency = do_bench(lambda: group_gemm_fp8(A, B, cnt), backend="cupti")
    print(f"Latency: {tl_latency:.3f} ms")
    print(f"TFLOPS:  {total_flops / tl_latency * 1e3 / 1e12:.2f}")


if __name__ == "__main__":
    main()
