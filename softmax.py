"""
Softmax 算子的 Triton 实现。
每行独立计算：先求行内最大值，再指数化，最后归一化。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    stride_row,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    逐行 Softmax：softmax(x) = exp(x - max) / sum(exp(x - max))

    每个 program 处理一行数据，分三步：
    1. 加载一行数据，求最大值（用于数值稳定性）
    2. 对每个元素计算 exp(x - max)，并累加求和
    3. 将 exp(x - max) 除以总和，写回结果
    """
    # ── 当前 program 对应第几行 ───────────────────────────────────
    row_idx = tl.program_id(axis=0)
    row_start_ptr = input_ptr + row_idx * stride_row
    out_row_start_ptr = output_ptr + row_idx * stride_row

    # ── 加载整行数据 ──────────────────────────────────────────────
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float("inf"))

    # ── 第一步：求行最大值（数值稳定性） ──────────────────────────
    row_max = tl.max(row, axis=0)

    # ── 第二步：减去最大值后求指数，并累加 ────────────────────────
    row = row - row_max
    numerator = tl.exp(row)
    denominator = tl.sum(numerator, axis=0)

    # ── 第三步：归一化 ────────────────────────────────────────────
    softmax_out = numerator / denominator

    # ── 写回 ──────────────────────────────────────────────────────
    tl.store(out_row_start_ptr + col_offsets, softmax_out, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    """Triton 逐行 Softmax 的宿主端封装。"""
    assert x.ndim == 2, "输入必须是 2D tensor"
    output = torch.empty_like(x)
    n_rows, n_cols = x.shape

    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = min(BLOCK_SIZE, 4096)

    grid = (n_rows,)
    softmax_kernel[grid](
        x, output,
        stride_row=x.stride(0),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
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
    x = torch.randn(128, 256, device="cuda")
    output = softmax(x)
    torch.testing.assert_close(output, F.softmax(x, dim=-1))
    print("softmax 正确性验证通过！\n")

    # ── 性能对比 ─────────────────────────────────────────────────────────────
    for n_rows, n_cols in [(128, 256), (512, 512), (1024, 1024), (2048, 2048)]:
        x = torch.randn(n_rows, n_cols, device="cuda")
        bench(f"triton  (shape={n_rows}x{n_cols})", softmax, x)
        bench(f"torch   (shape={n_rows}x{n_cols})", F.softmax, x, dim=-1)
        print()
