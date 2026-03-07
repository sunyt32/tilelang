import torch
import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench


@tilelang.jit
def _group_gemm_kernel(
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

    # Keep axis-0 as N blocks to match tcgen5 gemm kernel launch pattern.
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

        T.use_swizzle(8)

        eid = pid_m_to_eid[pid_m]
        local_pid_m = pid_m_to_local[pid_m]
        tx = T.get_thread_binding()

        start_m = offsets[eid]
        end_m = offsets[eid + 1]
        tile_m = start_m + local_pid_m * block_M

        if tile_m < end_m:

            if tx < 32:  # warp 0: stage tiles into shared memory
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
            elif tx < 64:  # warp 1: issue tcgen5 mma
                for k in T.serial(k_blocks):
                    stage = k % num_stages
                    T.mbarrier_wait_parity(loaded[stage], (k // num_stages) & 1)
                    T.gemm(
                        A_shared[stage, :, :],
                        B_shared[stage, :, :],
                        C_tmem,
                        mbar=consumed[stage],
                        transpose_B=True,
                        wg_wait=-1,
                        clear_accum=k == 0,
                    )
                T.tcgen05_mma_arrive(tmem_full) 

            # Wait for tcgen5 mma completion before reading C_tmem.
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


def group_gemm(
    A: torch.Tensor,    # [M_total, K]
    B: torch.Tensor,    # [E, N, K]
    cnt: torch.Tensor,  # [E, ] cumulative token counts
):
    block_M, block_N, block_K = 128, 256, 64
    in_dtype, out_dtype, accum_dtype = T.bfloat16, T.bfloat16, T.float
    num_stages = 4

    # Precompute flattened M-block -> (expert id, local M-block id in expert).
    expert_m_blocks = (cnt[1:] - cnt[:-1] + block_M - 1) // block_M
    eid_base = torch.arange(expert_m_blocks.numel(), device=cnt.device, dtype=torch.int32)
    pid_m_to_eid = torch.repeat_interleave(eid_base, expert_m_blocks)
    local_block_chunks = [
        torch.arange(int(b.item()), device=cnt.device, dtype=torch.int32)
        for b in expert_m_blocks
    ]
    pid_m_to_local = torch.cat(local_block_chunks)

    return _group_gemm_kernel(
        A, B, cnt, pid_m_to_eid, pid_m_to_local,
        block_M, block_N, block_K,
        in_dtype, out_dtype, accum_dtype, num_stages,
    )


def main():
    from examples.grouped_gemm_sm100.group_gemm import Mgemm as triton_Mgemm

    E = 8
    N, K = 1024, 3072
    tokens_per_expert = 4096
    M_total = E * tokens_per_expert

    tokens_per_expert_vec = torch.full((E,), tokens_per_expert, dtype=torch.int32, device="cuda")
    offsets = torch.tensor([-3, -2, -1, 0, 1, 2, 3, 0], dtype=torch.int32, device="cuda")
    tokens_per_expert_vec = tokens_per_expert_vec + offsets
    tokens_per_expert_vec[-1] = tokens_per_expert_vec[-1] - offsets.sum()

    A = torch.randn(M_total, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(E, N, K, device="cuda", dtype=torch.bfloat16)
    cnt = torch.cat((torch.tensor([0], device="cuda", dtype=torch.int32), torch.cumsum(tokens_per_expert_vec, dim=0, dtype=torch.int32)))
    max_M_per_E = int(tokens_per_expert_vec.max().item())

    # --- Correctness ---
    tl_c = group_gemm(A, B, cnt)

    ref_c = torch.zeros(M_total, N, device="cuda", dtype=torch.bfloat16)
    triton_c = triton_Mgemm(A, B, cnt, max_M_per_E, True)
    torch_mm_c = torch._grouped_mm(A, B.transpose(-1, -2), offs=cnt[1:])
    B_kn = B.transpose(1, 2).contiguous()
    for i in range(E):
        s, e = cnt[i].item(), cnt[i + 1].item()
        ref_c[s:e] = (A[s:e].float() @ B_kn[i].float()).bfloat16()

    torch.testing.assert_close(tl_c, ref_c, rtol=1e-2, atol=1e-2)
    print("TileLang correctness check passed. ✅")
    torch.testing.assert_close(triton_c, ref_c, rtol=1e-2, atol=1e-2)
    print("Triton correctness check passed. ✅")
    torch.testing.assert_close(torch_mm_c, ref_c, rtol=1e-2, atol=1e-2)
    print("PyTorch grouped_mm correctness check passed. ✅")

    # --- TFLOPS Comparison ---
    total_flops = 2 * M_total * N * K
    torch_latency = do_bench(
        lambda: torch._grouped_mm(A, B.transpose(-1, -2), offs=cnt[1:]),
        backend="cupti",
    )

    tl_latency = do_bench(
        lambda: group_gemm(A, B, cnt),
        backend="cupti",
    )
    triton_latency = do_bench(
        lambda: triton_Mgemm(A, B, cnt, max_M_per_E, True),
        backend="cupti",
    )

    print(f"\n{'Impl':<20} {'Latency (ms)':>12} {'TFLOPS':>10}")
    print("-" * 44)
    print(f"{'PyTorch':<20} {torch_latency:>12.3f} {total_flops / torch_latency * 1e3 / 1e12:>10.2f}")
    print(f"{'Tilelang':<20} {tl_latency:>12.3f} {total_flops / tl_latency * 1e3 / 1e12:>10.2f}")
    print(f"{'Triton':<20} {triton_latency:>12.3f} {total_flops / triton_latency * 1e3 / 1e12:>10.2f}")


if __name__ == "__main__":
    main()
