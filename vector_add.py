import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr,
               y_ptr,
               output_ptr,
               n_elements,
               BLOCK_SIZE: tl.constexpr,
               ):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = x.numel()
    # 小数据量用一个 block 减少启动开销
    BLOCK_SIZE = min(4096, triton.next_power_of_2(n_elements))
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output


def bench_fn(label, fn, *args, warmup=10, rep=100):
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
    elapsed_ms = start.elapsed_time(end) / rep
    print(f"{label}: {elapsed_ms:.3f} ms")


if __name__ == "__main__":
    size = 100000
    x = torch.randn(size, device="cuda")
    y = torch.randn(size, device="cuda")
    output = add(x, y)
    torch.testing.assert_close(output, x + y)
    print("vector_add passed!")

    for size in [10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]:
        x = torch.randn(size, device="cuda")
        y = torch.randn(size, device="cuda")
        bench_fn(f"triton  (N={size:>12})", add, x, y)
        bench_fn(f"torch   (N={size:>12})", torch.add, x, y)
        print()

