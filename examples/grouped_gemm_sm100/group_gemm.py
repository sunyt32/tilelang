import torch
import triton
from triton import Config
import triton.language as tl

from triton.tools.tensor_descriptor import TensorDescriptor
from triton.tools.ragged_tma import create_ragged_descriptor, load_ragged, store_ragged


@triton.jit
def compute_grouped_pid(tile_id, num_pid_in_group, num_pid_m, super_group_m):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * super_group_m
    group_size_m = min(num_pid_m - first_pid_m, super_group_m)
    pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
    # pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n

def check_tma_alignment(tensor, name):
    """Check if tensor meets TMA alignment requirements."""
    elem_bytes = tensor.element_size()
    strides = tensor.stride()
    # All strides except the last must be 16-byte aligned
    for i, stride in enumerate(strides[:-1]):
        alignment = stride * elem_bytes
        if alignment % 16 != 0:
            raise ValueError(
                f"Tensor '{name}' stride[{i}]={stride} with element_size={elem_bytes} "
                f"is not 16-byte aligned (alignment={alignment}). "
                f"Consider padding the tensor or using a different data layout."
            )
    # Base pointer must be 16-byte aligned
    if tensor.data_ptr() % 16 != 0:
        raise ValueError(
            f"Tensor '{name}' base pointer is not 16-byte aligned. "
            f"data_ptr()={tensor.data_ptr()}"
        )


@triton.jit
def Mgemm_kernel(
    X_desc,
    W_desc,
    C_desc,
    slice_offs,
    N, K,
    TRANSPOSE_B: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    eid = tl.program_id(axis=1)
    m_start = tl.load(slice_offs + eid, cache_modifier=".ca")
    m_end = tl.load(slice_offs + eid + 1, cache_modifier=".ca")
    dtype: tl.dtype = C_desc.dtype
    
    if m_start < m_end: 
        m_size = m_end - m_start
        pid = tl.program_id(axis=0)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_m = tl.num_programs(axis=0) // num_pid_n
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        pid_m, pid_n = compute_grouped_pid(pid, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
        offs_m = pid_m * BLOCK_SIZE_M
        offs_n = pid_n * BLOCK_SIZE_N
        if offs_m >= m_size:
            return
        accumulator = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
        for offs_k in range(0, K, BLOCK_SIZE_K):
            a = load_ragged(X_desc, m_start, m_size, [offs_m, offs_k], ragged_dim=0).reshape(BLOCK_SIZE_M, BLOCK_SIZE_K)
            if TRANSPOSE_B:
                b = W_desc.load([eid, offs_n, offs_k]).reshape(BLOCK_SIZE_N, BLOCK_SIZE_K)
                accumulator += tl.dot(a, b.T)
            else:
                b = W_desc.load([eid, offs_k, offs_n]).reshape(BLOCK_SIZE_K, BLOCK_SIZE_N)
                accumulator += tl.dot(a, b)
        c = accumulator.to(dtype)
        store_ragged(C_desc, m_start, m_size, [offs_m, offs_n], c, ragged_dim=0)


def Mgemm_descriptors3d(x, w, y, block_m, block_n, block_k, transpose_B):
    check_tma_alignment(x, "x")
    check_tma_alignment(w, "w")
    check_tma_alignment(y, "y")
    
    x_desc = create_ragged_descriptor(
        x, 
        block_shape=[block_m, block_k],
        ragged_dim=0
    )
    y_desc = create_ragged_descriptor(
        y, 
        block_shape=[block_m, block_n],
        ragged_dim=0
    )
    if transpose_B:
        w_desc = TensorDescriptor(
            w,
            shape=list(w.shape),
            strides=list(w.stride()),
            block_shape=[1, block_n, block_k],
        )
    else:
        w_desc = TensorDescriptor(
            w,
            shape=list(w.shape),
            strides=list(w.stride()),
            block_shape=[1, block_k, block_n],
        )
    return x_desc, w_desc, y_desc

def Mgemm(
    x: torch.Tensor,   # [M, K]
    w: torch.Tensor,   # [E, N, K]
    cnt: torch.Tensor, # accumulative prefix sum, 0 padding at front
    max_M_per_E: int,
    transpose_B: bool,
):
    M, K = x.shape
    N = w.shape[1] if transpose_B else w.shape[2]
    E = w.shape[0]
    assert cnt.shape[0] - 1 == E
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    META = {
        'BLOCK_SIZE_M': 256,
        'BLOCK_SIZE_N': 256,
        'BLOCK_SIZE_K': 64,
        'GROUP_SIZE_M': 8,
        'num_warps': 8,
        'num_stages': 3,
    }
    assert N % META['BLOCK_SIZE_N'] == 0
    assert K % META['BLOCK_SIZE_K'] == 0
    x_desc, w_desc, y_desc = Mgemm_descriptors3d(x, w, y, 
                                                 META['BLOCK_SIZE_M'], META['BLOCK_SIZE_N'], META['BLOCK_SIZE_K'], 
                                                 transpose_B)
    grid = lambda META: (
        triton.cdiv(max_M_per_E, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        E,
    )
    Mgemm_kernel[grid](
        x_desc,
        w_desc,
        y_desc,
        cnt,
        N, K,
        TRANSPOSE_B=transpose_B,
        **META,
    )
    return y


@triton.jit
def Kgemm_kernel(
    grad,
    X,
    c_ptr,
    cnt_ptr,
    M, N, K,
    stride_ce, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    eid = tl.program_id(axis=1)
    start_k = 0 if eid == 0 else tl.load(cnt_ptr + eid - 1).to(tl.int32)
    end_k = tl.load(cnt_ptr + eid).to(tl.int32)
    k_size = end_k - start_k
    # NOTE: No early return here! We always need to store the accumulator (even zeros)    
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    pid_m, pid_n = compute_grouped_pid(pid, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
    offs_m = pid_m * BLOCK_SIZE_M
    offs_n = pid_n * BLOCK_SIZE_N
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    n_steps = tl.cdiv(k_size, BLOCK_SIZE_K)
    offs_k = 0
    for k in range(n_steps):
        a = load_ragged(grad, start_k, k_size, [offs_k, offs_m], ragged_dim=0)
        a = a.reshape(BLOCK_SIZE_K, BLOCK_SIZE_M)
        
        b = load_ragged(X, start_k, k_size, [offs_k, offs_n], ragged_dim=0)
        b = b.reshape(BLOCK_SIZE_K, BLOCK_SIZE_N)
        accumulator += tl.dot(a.T, b)
        offs_k += BLOCK_SIZE_K
        
    # Always store the accumulator (zeros for empty experts)
    # TODO: use TMA store can eliminate register usage for c_ptrs. 
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    c_ptrs = c_ptr + eid * stride_ce + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, accumulator)

def Kgemm_descriptors(grad, X, block_m, block_n, block_k):
    check_tma_alignment(grad, "grad")
    check_tma_alignment(X, "X")
    
    grad_desc = create_ragged_descriptor(
        grad,
        block_shape=[block_k, block_m],
        ragged_dim=0
    )
    x_desc = create_ragged_descriptor(
        X,
        block_shape=[block_k, block_n],
        ragged_dim=0,   
    )
    return grad_desc, x_desc

def Kgemm(
    grad: torch.Tensor, # [M_total, ]
    X: torch.Tensor,    # [M_total, ]
    cnt: torch.Tensor,
    descriptor: str,
):
    K, M = grad.shape
    N = X.shape[1]
    E = cnt.shape[0]
    grad_w = torch.empty((E, M, N), device=grad.device, dtype=torch.float32)
    if descriptor == 'gw13':
        META = {
        'BLOCK_SIZE_M': 256,
        'BLOCK_SIZE_N': 128,
        'BLOCK_SIZE_K': 32,
        'GROUP_SIZE_M': 8,
        'num_warps': 4,
        'num_stages': 4,
    }
    if descriptor == 'gw2':
        META = {
        'BLOCK_SIZE_M': 128,
        'BLOCK_SIZE_N': 256,
        'BLOCK_SIZE_K': 32,
        'GROUP_SIZE_M': 8,
        'num_warps': 4,
        'num_stages': 4,
    }
    assert M % META['BLOCK_SIZE_M'] == 0
    assert N % META['BLOCK_SIZE_N'] == 0
    grad_desc, x_desc = Kgemm_descriptors(grad, X, META['BLOCK_SIZE_M'], META['BLOCK_SIZE_N'], META['BLOCK_SIZE_K'])
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        E,
    )
    Kgemm_kernel[grid](
        grad_desc,
        x_desc,
        grad_w,
        cnt,
        M,
        N,
        K,
        grad_w.stride(0), grad_w.stride(1), grad_w.stride(2),
        **META,
    )
    return grad_w
