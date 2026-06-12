"""
GEMM (General Matrix Multiply) 算子的 Triton 实现。
C = A @ B，其中 A 形状 (M, K)，B 形状 (K, N)，C 形状 (M, N)。
使用二维分块 + 共享内存优化，是 LLM 推理中最核心的算子。

核心优化思路：
1. 分块计算：将大矩阵划分为 BLOCK_M x BLOCK_N 的子块，每个 program 独立计算一个子块
2. 沿 K 维度累加：acc += A_sub @ B_sub，避免一次性加载整个矩阵
3. tl.dot 调用 GPU Tensor Core 硬件加速
4. GROUP_M 交错调度：让相邻 program 读取相近的内存区域，提高 L2 cache 命中率
"""

import torch
import triton
import triton.language as tl


# autotune 尝试 9 种不同的分块策略，选最快的：
#   BLOCK_M/N/K: 控制子块大小，越大显存访问越合并，但寄存器压力越大
#   GROUP_M: 相邻 program 交错组大小，提高 L2 cache 复用
#   num_warps: 每个 program 的线程数（1 warp = 32 线程）
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 4}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 4}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 4}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 4}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 4}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 4}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 4}, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    矩阵乘法 C = A @ B，二维分块实现。

    每个 program 负责计算输出矩阵 C 的一个 BLOCK_M x BLOCK_N 子块。
    核心思路：
    1. 通过二维 grid 将 C 矩阵划分为多个子块
    2. 每个子块沿 K 维度分块累加：acc += A_sub @ B_sub
    3. 使用 tl.dot 调用 Tensor Core 加速矩阵乘法
    """
    # ── 将一维 program ID 映射到二维网格 ──────────────────────────────
    # Triton 的 grid 是一维的，但 GEMM 需要二维分块（M 方向 x N 方向）。
    # GROUP_M 优化：将 program 按组交错排列，使得连续编号的 program
    # 访问 A 矩阵的同一区域，从而提高 L2 cache 命中率。
    #
    # 举例：GROUP_M=4, num_pid_m=8, num_pid_n=4
    #   正常顺序: (0,0),(0,1),(0,2),(0,3),(1,0),(1,1),...
    #   交错后:   (0,0),(0,1),(0,2),(0,3),(1,0),(1,1),(1,2),(1,3),
    #             (4,0),(4,1),...  ← 第 4 行的 program 和第 0 行靠近，复用 cache
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)   # M 方向有多少个 program
    num_pid_n = tl.cdiv(N, BLOCK_N)   # N 方向有多少个 program
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ── 计算当前 program 负责的 C 子块起始位置 ──────────────────────
    # offs_am: 当前 program 要处理的 A 矩阵的 M 方向行索引 [pid_m*BLOCK_M, ..., (pid_m+1)*BLOCK_M-1]
    # offs_bn: 当前 program 要处理的 B 矩阵的 N 方向列索引 [pid_n*BLOCK_N, ..., (pid_n+1)*BLOCK_N-1]
    # 取模 % M 和 % N 是为了防止越界（当 M/N 不是 BLOCK 的整数倍时）
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N

    # ── 累加器初始化为 0 ─────────────────────────────────────────────
    # acc 形状为 [BLOCK_M, BLOCK_N]，存储当前 program 计算的 C 子块结果
    # 用 float32 累加避免精度损失（即使输入是 float16）
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ── 沿 K 维度分块累加 ────────────────────────────────────────────
    # C[i,j] = Σ A[i,k] * B[k,j]，k 从 0 到 K-1
    # 这里把 k 循环拆成多个 BLOCK_K 大小的块，每次只加载一小段到 SRAM
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # 当前 K 块的起始偏移
        offs_k = (k * BLOCK_K + tl.arange(0, BLOCK_K)) % K

        # 从 A 加载 [BLOCK_M, BLOCK_K] 子块
        # offs_am[:, None] → [BLOCK_M, 1] 行索引（每个元素重复 BLOCK_K 次）
        # offs_k[None, :]  → [1, BLOCK_K] 列索引（每个元素重复 BLOCK_M 次）
        # 广播后得到 [BLOCK_M, BLOCK_K] 的二维地址矩阵
        A_ptrs = A_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
        A_mask = (offs_am[:, None] < M) & (offs_k[None, :] < K)
        A = tl.load(A_ptrs, mask=A_mask, other=0.0)

        # 从 B 加载 [BLOCK_K, BLOCK_N] 子块
        # 注意 B 是按行主序存储的，所以 offs_k 是行索引，offs_bn 是列索引
        B_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        B_mask = (offs_k[:, None] < K) & (offs_bn[None, :] < N)
        B = tl.load(B_ptrs, mask=B_mask, other=0.0)

        # Tensor Core 矩阵乘法：acc += A @ B
        # tl.dot 调用 GPU 的 Tensor Core 硬件指令，比逐元素乘法快数倍
        # out_dtype=tl.float32 保证累加精度
        acc += tl.dot(A, B, out_dtype=tl.float32)

    # ── 写回结果到 C ─────────────────────────────────────────────────
    # 将累加器 acc（[BLOCK_M, BLOCK_N] 的子块结果）写回显存
    # 注意这里不用取模，因为写回时 pid_m * BLOCK_M 不会超出 M 的范围
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C_ptrs = C_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    C_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(C_ptrs, acc, mask=C_mask)


def gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Triton GEMM 的宿主端封装。"""
    assert a.shape[1] == b.shape[0], "A 的列数必须等于 B 的行数"
    assert a.is_contiguous() and b.is_contiguous(), "输入 tensor 必须连续存储"

    M, K = a.shape
    K_b, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    gemm_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


# ─── 性能测试工具函数 ───────────────────────────────────────────────────────

def bench(label, fn, *args, warmup=25, rep=100):
    """测量 CUDA kernel 平均耗时（毫秒）。"""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / rep
    print(f"{label}: {ms:.3f} ms")


# ─── 主程序 ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── 正确性验证 ───────────────────────────────────────────────────────────
    M, K, N = 256, 512, 256
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)

    c_triton = gemm(a, b)
    c_ref = a @ b
    torch.testing.assert_close(c_triton, c_ref, rtol=1e-2, atol=1e-2)
    print("gemm 正确性验证通过！\n")

    # ── 性能对比 ─────────────────────────────────────────────────────────────
    torch_compile_fn = torch.compile(lambda a, b: a @ b)
    for M, K, N in [
        (64, 64, 64),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (8192, 8192, 8192),
    ]:
        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)
        bench(f"triton          (M={M:>5}, K={K:>5}, N={N:>5})", gemm, a, b)
        bench(f"torch eager     (M={M:>5}, K={K:>5}, N={N:>5})", torch.matmul, a, b)
        bench(f"torch.compile   (M={M:>5}, K={K:>5}, N={N:>5})", torch_compile_fn, a, b)
        print()
