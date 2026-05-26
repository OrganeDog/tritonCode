"""
RMSNorm 算子的 Triton 实现。
公式：y = x / sqrt(mean(x²) + eps) * w
每个 program 处理一行，一次读写完成，中间计算全在 SRAM 中。
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_cols'],
)
@triton.jit
def rmsnorm_kernel(
    x_ptr,
    w_ptr,
    output_ptr,
    stride_row,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    逐行 RMSNorm：y = x / sqrt(mean(x²) + eps) * w

    每个 program 处理一行，步骤：
    1. 加载整行数据和对应的权重
    2. 计算均方根 sqrt(mean(x²) + eps)
    3. 归一化后乘权重，写回结果
    """
    # ── 当前 program 对应第几行 ───────────────────────────────────
    row_idx = tl.program_id(axis=0)
    row_start_ptr = x_ptr + row_idx * stride_row
    out_row_start_ptr = output_ptr + row_idx * stride_row

    # ── 加载整行数据 ──────────────────────────────────────────────
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    x = tl.load(row_start_ptr + col_offsets, mask=mask, other=0.0)
    w = tl.load(w_ptr + col_offsets, mask=mask, other=1.0)

    # ── 第一步：计算均方根 ─────────────────────────────────────────
    x_sq = x * x
    mean_sq = tl.sum(x_sq, axis=0) / n_cols
    rms = tl.sqrt(mean_sq + eps)

    # ── 第二步：归一化 + 乘权重 ─────────────────────────────────────
    output = (x / rms) * w

    # ── 写回 ──────────────────────────────────────────────────────
    tl.store(out_row_start_ptr + col_offsets, output, mask=mask)


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Triton 逐行 RMSNorm 的宿主端封装。"""
    assert x.ndim == 2, "输入必须是 2D tensor"
    assert w.ndim == 1, "权重必须是 1D tensor"
    assert x.shape[-1] == w.shape[0], "输入列数必须等于权重长度"

    output = torch.empty_like(x)
    n_rows, n_cols = x.shape

    grid = (n_rows,)
    rmsnorm_kernel[grid](
        x, w, output,
        stride_row=x.stride(0),
        n_cols=n_cols,
        eps=eps,
    )
    return output


# ─── 性能测试工具函数 ───────────────────────────────────────────────────────

def bench(label, fn, *args, warmup=10, rep=100):
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
    n_rows, n_cols = 128, 256
    x = torch.randn(n_rows, n_cols, device="cuda")
    w = torch.ones(n_cols, device="cuda")
    eps = 1e-5

    output = rmsnorm(x, w, eps)

    # PyTorch 参考实现
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    ref = x / rms * w
    torch.testing.assert_close(output, ref)
    print("rmsnorm 正确性验证通过！\n")

    # ── 性能对比 ─────────────────────────────────────────────────────────────
    for n_rows, n_cols in [(128, 256), (512, 512), (1024, 1024), (2048, 2048), (4096, 4096), (8192, 8192)]:
        x = torch.randn(n_rows, n_cols, device="cuda")
        w = torch.ones(n_cols, device="cuda")
        bench(f"triton  (shape={n_rows}x{n_cols})", rmsnorm, x, w, eps)
        bench(f"torch   (shape={n_rows}x{n_cols})",
              lambda a, b, e: (a / torch.sqrt(torch.mean(a ** 2, dim=-1, keepdim=True) + e)) * b,
              x, w, eps)
        print()
