def _all_to_all_optimized(
    local_input: Tensor,
    scatter_dim: int,
    gather_dim: int,
    group: Optional[dist.ProcessGroup],
    async_op: bool = False,
):
    seq_world_size = dist.get_world_size(group)
    
    # 如果 world_size 为 1，直接返回（无需通信）
    if seq_world_size == 1:
        return local_input.contiguous()

    # 确保输入是连续的
    local_input = local_input.contiguous()
    
    # 计算每个 rank 要 scatter 的 chunk 大小
    scatter_chunk_size = local_input.size(scatter_dim) // seq_world_size
    
    # 计算 output 的 shape
    output_shape = list(local_input.shape)
    output_shape[scatter_dim] = scatter_chunk_size  # scatter 后每个 chunk 的大小
    output_shape[gather_dim] = output_shape[gather_dim] * seq_world_size  # gather 后该维度扩大
    
    # 预分配 output 缓冲区（一次性分配，避免 List 开销）
    output = torch.empty(output_shape, dtype=local_input.dtype, device=local_input.device)
    
    # 调用高效的 all_to_all_single
    comm = dist.all_to_all_single(
        output,           # 输出张量（已预分配好）
        local_input,      # 输入张量
        scatter_dim=scatter_dim,
        gather_dim=gather_dim,
        group=group,
        async_op=async_op,
    )
    
    if async_op:
        def wait():
            comm.wait()
            return output.contiguous()
        return wait
    
    return output.contiguous()