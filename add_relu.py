"""
Add + ReLU 融合算子的 Triton 实现。
单次读写显存完成 x + y 和 ReLU，避免 PyTorch 分两步的中间回写。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def add_relu_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    融合算子：output = max(x + y, 0)。

    与分开调用 add 和 relu 相比，只需一次加载和一次写回，
    减少显存访问次数，提升带宽利用率和整体性能。
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = tl.maximum(x + y, 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)


def add_relu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Triton 融合 Add + ReLU 的宿主端封装。"""
    assert x.shape == y.shape, "两个输入 tensor 形状必须一致"
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_relu_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=4096)
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
    size = 100_000
    x = torch.randn(size, device="cuda")
    y = torch.randn(size, device="cuda")
    output = add_relu(x, y)
    torch.testing.assert_close(output, F.relu(x + y))
    print("add_relu 正确性验证通过！\n")

    # ── 性能对比：融合算子 vs PyTorch 分步调用 ────────────────────────────────
    torch_compile_fn = torch.compile(lambda a, b: F.relu(a + b))
    for size in [10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]:
        x = torch.randn(size, device="cuda")
        y = torch.randn(size, device="cuda")
        bench(f"triton fused    (N={size:>12})", add_relu, x, y)
        bench(f"torch eager     (N={size:>12})", lambda a, b: F.relu(a + b), x, y)
        bench(f"torch.compile   (N={size:>12})", torch_compile_fn, x, y)
        print()
