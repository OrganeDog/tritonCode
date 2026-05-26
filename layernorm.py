"""
LayerNorm 算子的 Triton 实现。
公式：y = (x - mean) / sqrt(var + eps) * w + b
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
def layernorm_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    output_ptr,
    stride_row,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    逐行 LayerNorm：y = (x - mean) / sqrt(var + eps) * w + b

    每个 program 处理一行，步骤：
    1. 加载整行数据和对应的权重、偏置
    2. 计算均值 mean(x) 和方差 var(x)
    3. 标准化后乘权重加偏置，写回结果
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
    b = tl.load(b_ptr + col_offsets, mask=mask, other=0.0)

    # ── 第一步：计算均值 ───────────────────────────────────────────
    mean = tl.sum(x, axis=0) / n_cols

    # ── 第二步：计算方差 ───────────────────────────────────────────
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / n_cols

    # ── 第三步：标准化 + 缩放 + 偏置 ───────────────────────────────
    output = x_centered * tl.rsqrt(var + eps) * w + b

    # ── 写回 ──────────────────────────────────────────────────────
    tl.store(out_row_start_ptr + col_offsets, output, mask=mask)


def layernorm(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Triton 逐行 LayerNorm 的宿主端封装。"""
    assert x.ndim == 2, "输入必须是 2D tensor"
    assert w.ndim == 1, "权重必须是 1D tensor"
    assert b.ndim == 1, "偏置必须是 1D tensor"
    assert x.shape[-1] == w.shape[0], "输入列数必须等于权重长度"

    output = torch.empty_like(x)
    n_rows, n_cols = x.shape

    grid = (n_rows,)
    layernorm_kernel[grid](
        x, w, b, output,
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
    import torch.nn.functional as F

    # ── 正确性验证 ───────────────────────────────────────────────────────────
    n_rows, n_cols = 128, 256
    x = torch.randn(n_rows, n_cols, device="cuda")
    w = torch.ones(n_cols, device="cuda")
    b = torch.zeros(n_cols, device="cuda")
    eps = 1e-5

    output = layernorm(x, w, b, eps)
    ref = F.layer_norm(x, (n_cols,), weight=w, bias=b, eps=eps)
    torch.testing.assert_close(output, ref)
    print("layernorm 正确性验证通过！\n")

    # ── 性能对比 ─────────────────────────────────────────────────────────────
    for n_rows, n_cols in [(128, 256), (512, 512), (1024, 1024), (2048, 2048), (4096, 4096), (8192, 8192)]:
        x = torch.randn(n_rows, n_cols, device="cuda")
        w = torch.ones(n_cols, device="cuda")
        b = torch.zeros(n_cols, device="cuda")
        bench(f"triton  (shape={n_rows}x{n_cols})", layernorm, x, w, b, eps)
        bench(f"torch   (shape={n_rows}x{n_cols})",
              lambda a, ww, bb, e: F.layer_norm(a, (a.shape[-1],), weight=ww, bias=bb, eps=e),
              x, w, b, eps)
        print()
