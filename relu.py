"""
ReLU 算子的 Triton 实现。
与 PyTorch 原生 F.relu 进行正确性和性能对比。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def relu_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    逐元素 ReLU 激活函数：output = max(x, 0)。

    每个程序实例处理一个 BLOCK_SIZE 大小的数据块：
    1. 计算当前 program 的全局起始偏移
    2. 生成当前块的元素索引 offsets
    3. 用 mask 处理越界元素（防止访问未初始化内存）
    4. 加载数据、应用 max(x, 0)、写回结果
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.maximum(x, 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)


def relu(x: torch.Tensor) -> torch.Tensor:
    """Triton ReLU kernel 的宿主端封装。"""
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    relu_kernel[grid](x, output, n_elements, BLOCK_SIZE=4096)
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
    output = relu(x)
    torch.testing.assert_close(output, F.relu(x))
    print("relu 正确性验证通过！\n")

    # ── 性能对比 ─────────────────────────────────────────────────────────────
    for size in [10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]:
        x = torch.randn(size, device="cuda")
        bench(f"triton  (N={size:>12})", relu, x)
        bench(f"torch   (N={size:>12})", F.relu, x)
        print()
