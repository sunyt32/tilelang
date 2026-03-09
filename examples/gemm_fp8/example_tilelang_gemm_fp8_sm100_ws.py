# Warp-specialized FP8 GEMM on SM100 (non-persistent, 1-SM)

import torch
import tilelang
import tilelang.language as T
from tilelang.utils.tensor import map_torch_type


def matmul_ws(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
):
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)

    k_iters = T.ceildiv(K, block_K)

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((num_stages, *A_shared_shape), in_dtype)
            B_shared = T.alloc_shared((num_stages, *B_shared_shape), in_dtype)
            C_tmem = T.alloc_tmem([block_M, block_N], accum_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_N), out_dtype)
            loaded = T.alloc_barrier([32] * num_stages)
            consumed = T.alloc_barrier([1] * num_stages)
            tmem_full = T.alloc_barrier([1])

            tx = T.get_thread_binding()

            T.use_swizzle(8)

            if tx < 32:  # warp 0: issue tma
                for k in T.serial(k_iters):
                    T.mbarrier_wait_parity(consumed[k % num_stages], ((k // num_stages) & 1) ^ 1)
                    if trans_A:
                        T.copy(
                            A[k * block_K:(k + 1) * block_K, by * block_M:(by + 1) * block_M],
                            A_shared[k % num_stages, :, :],
                        )
                    else:
                        T.copy(
                            A[by * block_M:(by + 1) * block_M, k * block_K:(k + 1) * block_K],
                            A_shared[k % num_stages, :, :],
                        )
                    if trans_B:
                        T.copy(
                            B[bx * block_N:(bx + 1) * block_N, k * block_K:(k + 1) * block_K],
                            B_shared[k % num_stages, :, :],
                        )
                    else:
                        T.copy(
                            B[k * block_K:(k + 1) * block_K, bx * block_N:(bx + 1) * block_N],
                            B_shared[k % num_stages, :, :],
                        )
                    T.mbarrier_arrive(loaded[k % num_stages])
            elif tx < 64:  # warp 1: issue tcgen5 mma
                for k in T.serial(k_iters):
                    T.mbarrier_wait_parity(loaded[k % num_stages], (k // num_stages) & 1)
                    T.gemm(
                        A_shared[k % num_stages, :, :],
                        B_shared[k % num_stages, :, :],
                        C_tmem,
                        transpose_A=trans_A,
                        transpose_B=trans_B,
                        mbar=consumed[k % num_stages],
                        wg_wait=-1,
                        clear_accum=k == 0,
                    )
                T.tcgen05_mma_arrive(tmem_full)

            # Wait for all tcgen5 to finish
            T.mbarrier_wait_parity(tmem_full, 0)

            T.sync_threads()
            T.copy(C_tmem, C_local)
            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return main


def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


M, N, K = 8192, 8192, 8192
block_M, block_N, block_K = 128, 256, 128
trans_A, trans_B = False, True
num_stages = 4

in_dtype = T.float8_e4m3fn  # mxfp8
out_dtype = T.bfloat16
accum_dtype = T.float32

torch_in_dtype = map_torch_type(in_dtype)

func = matmul_ws(
    M, N, K,
    block_M, block_N, block_K,
    trans_A, trans_B,
    in_dtype, out_dtype, accum_dtype,
    num_stages,
)
jit_kernel = tilelang.compile(
    func,
    out_idx=[2],
    target="cuda",
)

a = torch.randn(M, K, device="cuda", dtype=torch.float16).to(torch_in_dtype)
b = torch.randn(N, K, device="cuda", dtype=torch.float16).to(torch_in_dtype)

c = jit_kernel(a, b)
ref_c = (a.to(torch.half) @ b.T.to(torch.half)).float()
c = c.float()
diff = calc_diff(c, ref_c)
print(f"[{in_dtype} -> acc:{accum_dtype} -> out:{out_dtype}] diff = {diff}")

profiler = jit_kernel.get_profiler()
latency = profiler.do_bench()
print(f"Latency: {latency} ms")
print(f"Flops: {2 * M * N * K / (latency / 1e3) / 1e12} TFLOPS")
