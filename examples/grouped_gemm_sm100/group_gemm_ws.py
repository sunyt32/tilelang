import torch
import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench


@tilelang.jit
def _group_gemm_kernel(
    A,
    B,
    offsets,
    max_M_per_E,
    block_M,
    block_N,
    block_K,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
):
    M_total, N, K, E, E1 = T.const("M_total, N, K, E, E1")
    A: T.Tensor[[M_total, K], in_dtype]
    B: T.Tensor[[E, K, N], in_dtype]
    offsets: T.Tensor[[E1], "int64"]
    C = T.empty((M_total, N), out_dtype)

    m_blocks = T.ceildiv(max_M_per_E, block_M)
    n_blocks = T.ceildiv(N, block_N)
    k_blocks = T.ceildiv(K, block_K)

    with T.Kernel(m_blocks * n_blocks, E, threads=128) as (pid, eid):
        A_shared = T.alloc_shared((block_M, block_K), in_dtype)
        B_shared = T.alloc_shared((block_K, block_N), in_dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
        C_local_cast = T.alloc_fragment((block_M, block_N), out_dtype)

        pid_m = pid // n_blocks
        pid_n = pid % n_blocks

        start_m = offsets[eid]
        end_m = offsets[eid + 1]

        if start_m + pid_m * block_M < end_m:
            T.clear(C_local)
            for k in T.serial(k_blocks):
                T.copy(
                    A[start_m + pid_m * block_M : start_m + (pid_m + 1) * block_M,
                      k * block_K : (k + 1) * block_K],
                    A_shared,
                )
                T.copy(
                    B[eid, k * block_K : (k + 1) * block_K,
                      pid_n * block_N : (pid_n + 1) * block_N],
                    B_shared,
                )
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C_local_cast)
            T.copy(C_local_cast, C[start_m + pid_m * block_M, pid_n * block_N])

    return C


def group_gemm(
    A: torch.Tensor,    # [M_total, K]
    B: torch.Tensor,    # [E, K, N] or [E, N, K] if transpose_B
    cnt: torch.Tensor,  # [E, ] cumulative token counts
    M: int,
    N: int,
    K: int,
    E: int,
    max_M_per_E: int,
    transpose_B: bool = False,
):
    """Tilelang Group GEMM, interface aligned with triton groupedM."""
    if transpose_B:
        B = B.transpose(1, 2).contiguous()

    offsets = torch.zeros(E + 1, dtype=torch.int64, device=A.device)
    offsets[1:] = cnt.to(torch.int64)

    block_M, block_N, block_K = 128, 128, 64
    in_dtype, out_dtype, accum_dtype = T.bfloat16, T.bfloat16, T.float
    num_stages = 2

    return _group_gemm_kernel(
        A, B, offsets, max_M_per_E,
        block_M, block_N, block_K,
        in_dtype, out_dtype, accum_dtype, num_stages,
    )


def main():
    from examples.grouped_gemm_sm100.group_gemm import Mgemm

    E = 8
    N, K = 4096, 4096
    tokens_per_expert = 512
    M_total = E * tokens_per_expert

    A = torch.randn(M_total, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(E, K, N, device="cuda", dtype=torch.bfloat16)
    cnt = torch.cumsum(
        torch.full((E,), tokens_per_expert, dtype=torch.int64, device="cuda"), dim=0
    )

    # --- Correctness ---
    tl_c = group_gemm(A, B, cnt, M_total, N, K, E, tokens_per_expert)

    ref_c = torch.zeros(M_total, N, device="cuda", dtype=torch.bfloat16)
    for i in range(E):
        s = 0 if i == 0 else cnt[i - 1].item()
        e = cnt[i].item()
        ref_c[int(s):int(e)] = (A[int(s):int(e)].float() @ B[i].float()).bfloat16()

    torch.testing.assert_close(tl_c, ref_c, rtol=1e-2, atol=1e-2)
    print("Correctness check passed. ✅")

    # --- TFLOPS Comparison ---
    total_flops = 2 * M_total * N * K

    tl_latency = do_bench(
        lambda: group_gemm(A, B, cnt, M_total, N, K, E, tokens_per_expert),
        backend="cupti",
    )
    triton_latency = do_bench(
        lambda: triton_Mgemm(A, B, cnt, M_total, N, K, E, tokens_per_expert, False),
        backend="cupti",
    )

    print(f"\n{'Impl':<20} {'Latency (ms)':>12} {'TFLOPS':>10}")
    print("-" * 44)
    print(f"{'Tilelang':<20} {tl_latency:>12.3f} {total_flops / tl_latency * 1e3 / 1e12:>10.2f}")
    print(f"{'Triton':<20} {triton_latency:>12.3f} {total_flops / triton_latency * 1e3 / 1e12:>10.2f}")


if __name__ == "__main__":
    main()
