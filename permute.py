import torch
import torch.nn as nn   
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_ROWS': 128, 'BLOCK_SIZE_COLS': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_ROWS': 256, 'BLOCK_SIZE_COLS': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE_ROWS': 256, 'BLOCK_SIZE_COLS': 128}, num_warps=8, num_stages=3),
        # 可以加入更多配置
    ],
    key=['num_cols', 'num_topK', 'N_ELEMENTS'],
)
@triton.jit
def permute_fused_gather_and_map_kernel_v3(
    # 指针
    input_ptr, sorted_row_id_ptr, output_ptr, row_id_map_ptr,
    # 维度信息
    num_cols, num_topK, num_tokens, num_out_tokens,
    # --- 新增参数 ---
    num_negative_one_in_indices,
    # Triton 元参数
    N_ELEMENTS: tl.constexpr,
    BLOCK_SIZE_ROWS: tl.constexpr,
    BLOCK_SIZE_COLS: tl.constexpr,
):
    # --- 基础设置 (与V2相同) ---
    pid_row = tl.program_id(axis=0)
    sorted_indices = pid_row * BLOCK_SIZE_ROWS + tl.arange(0, BLOCK_SIZE_ROWS)
    row_mask = sorted_indices < N_ELEMENTS

    # --- 核心逻辑变更 1: 计算带偏移的目标索引 ---
    target_indices = sorted_indices - num_negative_one_in_indices
    target_valid_mask = (target_indices >= 0) & (target_indices < num_out_tokens)

    # --- 任务 1: 创建 row_id_map (使用新逻辑) ---
    original_flat_indices = tl.load(sorted_row_id_ptr + sorted_indices, mask=row_mask)
    original_token_ids = original_flat_indices // num_topK
    original_k_ids = original_flat_indices % num_topK
    map_dest_addrs = original_k_ids * num_tokens + original_token_ids
    
    # 值现在基于新的目标索引和掩码
    value_to_write = tl.where(target_valid_mask, target_indices, -1)
    # 只有有效的原始行才进行写入
    map_write_mask = row_mask & (original_flat_indices >= 0)
    tl.store(row_id_map_ptr + map_dest_addrs, value_to_write, mask=map_write_mask)

    # --- 任务 2: 置换输入数据 (Gather/Scatter) (使用新逻辑) ---
    src_row_indices = original_flat_indices // num_topK
    src_row_base_ptrs = input_ptr + src_row_indices * num_cols

    for col_group_idx in range(tl.cdiv(num_cols, BLOCK_SIZE_COLS)):
        col_offsets = col_group_idx * BLOCK_SIZE_COLS + tl.arange(0, BLOCK_SIZE_COLS)
        col_mask = col_offsets < num_cols
        
        # Gather (源读取) 逻辑不变
        src_ptrs = src_row_base_ptrs[:, None] + col_offsets[None, :]
        load_mask = row_mask[:, None] & col_mask[None, :]
        gathered_data = tl.load(src_ptrs, mask=load_mask, other=0.0, cache_modifier=".ca")
        
        # --- 核心逻辑变更 2: 使用新的目标索引计算写入地址 ---
        # Scatter (目标写入) 地址使用 target_indices
        dst_ptrs = output_ptr + (target_indices[:, None] * num_cols + col_offsets[None, :])
        
        # 写入掩码也必须使用新的有效性掩码
        store_mask = target_valid_mask[:, None] & col_mask[None, :]
        tl.store(dst_ptrs, gathered_data, mask=store_mask)


def moe_permute_topk_op_triton_fused_v3(
    input_tensor: torch.Tensor,
    indices: torch.Tensor,
    num_out_tokens: int,
    workspace: list[torch.Tensor],
    max_expanded_token_num: int,
    # --- 新增参数 ---
    num_negative_one_in_indices: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """
    【V3 - 最终版】调用支持专家容量丢弃逻辑的、融合的 Triton 内核。
    """
    # 步骤 1, 2, 3: 参数提取、工作空间管理、排序 (与V2完全相同)
    num_tokens = input_tensor.size(0)
    num_cols = input_tensor.size(1)
    num_topK = indices.size(1)
    device = input_tensor.device
    num_elements = num_tokens * num_topK

    if not workspace or workspace[0].numel() < num_elements:
        workspace.clear()
        # ... (工作空间管理代码不变)
        int32_options = {'dtype': torch.int32, 'device': device}
        int64_options = {'dtype': torch.int64, 'device': device}
        workspace.extend([
            torch.empty(max_expanded_token_num, **int32_options),
            torch.empty(max_expanded_token_num, **int32_options),
            torch.empty(max_expanded_token_num, **int64_options)
        ])

    flat_indices = indices.view(-1)
    sorted_values_output_view = workspace[0][:num_elements]
    sorted_indices_output_view = workspace[2][:num_elements]
    torch.sort(flat_indices, out=(sorted_values_output_view, sorted_indices_output_view))
    sorted_row_id_result = sorted_indices_output_view

    # 步骤 4: 分配输出 (与V2完全相同)
    if num_out_tokens <= 0:
        num_out_tokens = num_elements - num_negative_one_in_indices
    permuted_output = torch.empty(num_out_tokens, num_cols, dtype=input_tensor.dtype, device=device)
    # row_id_map 的大小是 [num_topK, num_tokens]，所以总元素是 num_elements
    row_id_map = torch.empty(num_topK, num_tokens, dtype=torch.int32, device=device)

    # 步骤 5: 启动【新的 V3 优化内核】
    grid = lambda META: (triton.cdiv(num_elements, META['BLOCK_SIZE_ROWS']),)
    
    # 调用新的内核，并传入新参数
    permute_fused_gather_and_map_kernel_v3[grid](
        input_tensor,
        sorted_row_id_result,
        permuted_output,
        row_id_map,
        num_cols,
        num_topK,
        num_tokens,
        num_out_tokens,
        # --- 传入新参数 ---
        num_negative_one_in_indices,
        N_ELEMENTS=num_elements,
    )
    
    return permuted_output, row_id_map, sorted_row_id_result, workspace

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_ROWS': 128, 'BLOCK_SIZE_COLS': 64}),
        triton.Config({'BLOCK_SIZE_ROWS': 128, 'BLOCK_SIZE_COLS': 128}),
        triton.Config({'BLOCK_SIZE_ROWS': 256, 'BLOCK_SIZE_COLS': 256}),
        triton.Config({'BLOCK_SIZE_ROWS': 256, 'BLOCK_SIZE_COLS': 64 }),
        triton.Config({'BLOCK_SIZE_ROWS': 512, 'BLOCK_SIZE_COLS': 32}),
        triton.Config({'BLOCK_SIZE_ROWS': 256, 'BLOCK_SIZE_COLS': 128}),
        triton.Config({'BLOCK_SIZE_ROWS': 256, 'BLOCK_SIZE_COLS': 16}),
        triton.Config({'BLOCK_SIZE_ROWS': 512, 'BLOCK_SIZE_COLS': 16}),
        triton.Config({'BLOCK_SIZE_ROWS': 128, 'BLOCK_SIZE_COLS': 256}),
        triton.Config({'BLOCK_SIZE_ROWS': 64,  'BLOCK_SIZE_COLS': 256}),
        triton.Config({'BLOCK_SIZE_ROWS': 256, 'BLOCK_SIZE_COLS': 64}),
        triton.Config({'BLOCK_SIZE_ROWS': 256, 'BLOCK_SIZE_COLS': 64}),
    ],
    key=['num_cols', 'num_topK', 'num_tokens', 'num_out_tokens', 'N_ELEMENTS'],
)
@triton.jit
def permute_fused_gather_and_map_kernel_v2(
    # 指针
    input_ptr, sorted_row_id_ptr, output_ptr, row_id_map_ptr,
    # 维度信息
    num_cols, num_topK, num_tokens, num_out_tokens,
    # Triton 元参数
    N_ELEMENTS: tl.constexpr,
    # OPTIMIZATION 1: 这些参数现在由 autotuner 提供
    BLOCK_SIZE_ROWS: tl.constexpr,
    BLOCK_SIZE_COLS: tl.constexpr,
):

    pid_row = tl.program_id(axis=0)
    new_indices = pid_row * BLOCK_SIZE_ROWS + tl.arange(0, BLOCK_SIZE_ROWS)
    row_mask = new_indices < N_ELEMENTS

    original_flat_indices = tl.load(sorted_row_id_ptr + new_indices, mask=row_mask)
    original_token_ids = original_flat_indices // num_topK
    original_k_ids = original_flat_indices % num_topK
    map_dest_addrs = original_k_ids * num_tokens + original_token_ids
    value_to_write = tl.where(new_indices < num_out_tokens, new_indices, -1)
    tl.store(row_id_map_ptr + map_dest_addrs, value_to_write, mask=row_mask)
    src_row_indices = original_flat_indices // num_topK
    src_row_base_ptrs = input_ptr + src_row_indices * num_cols

    for col_group_idx in range(tl.cdiv(num_cols, BLOCK_SIZE_COLS)):
        col_offsets = col_group_idx * BLOCK_SIZE_COLS + tl.arange(0, BLOCK_SIZE_COLS)
        col_mask = col_offsets < num_cols
        src_ptrs = src_row_base_ptrs[:, None] + col_offsets[None, :]
        load_mask = row_mask[:, None] & col_mask[None, :]
        gathered_data = tl.load(src_ptrs, mask=load_mask, other=0.0, cache_modifier=".ca")
        dst_ptrs = output_ptr + (new_indices[:, None] * num_cols + col_offsets[None, :])
        store_mask = (new_indices < num_out_tokens)[:, None] & col_mask[None, :]
        tl.store(dst_ptrs, gathered_data, mask=store_mask)
        
@triton.jit
def permute_fused_gather_and_map_kernel(
    input_ptr,
    sorted_row_id_ptr,
    output_ptr,
    row_id_map_ptr,
    num_cols,      # 隐藏维度 H
    num_topK,      # 每个 Token 选择的专家数 K
    num_tokens,    # 原始 Token 数量 (num_rows)
    num_out_tokens,# 实际输出的 Token 数量 (可能小于 T*K)

    # ===== Triton 元参数 =====
    N_ELEMENTS: tl.constexpr,      # 总的扁平化元素数量 (num_tokens * num_topK)
    BLOCK_SIZE_ROWS: tl.constexpr, # 每个程序实例处理的行数
    BLOCK_SIZE_COLS: tl.constexpr, # 每次处理的列数
):

    pid_row = tl.program_id(axis=0)
    new_indices = pid_row * BLOCK_SIZE_ROWS + tl.arange(0, BLOCK_SIZE_ROWS)
    row_mask = new_indices < N_ELEMENTS
    original_flat_indices = tl.load(sorted_row_id_ptr + new_indices, mask=row_mask)
    original_token_ids = original_flat_indices // num_topK
    original_k_ids = original_flat_indices % num_topK

    map_dest_addrs = original_k_ids * num_tokens + original_token_ids
    value_to_write = tl.where(new_indices < num_out_tokens, new_indices, -1)
    
    tl.store(row_id_map_ptr + map_dest_addrs, value_to_write, mask=row_mask)
    src_row_indices = original_flat_indices // num_topK

    for col_group_idx in range(tl.cdiv(num_cols, BLOCK_SIZE_COLS)):
        col_offsets = col_group_idx * BLOCK_SIZE_COLS + tl.arange(0, BLOCK_SIZE_COLS)
        col_mask = col_offsets < num_cols

        # 从 input_ptr 中“收集”数据
        # 这是一个 Gather 操作，因为 src_row_indices 是不连续的
        src_ptrs = input_ptr + (src_row_indices[:, None] * num_cols + col_offsets[None, :])
        
        # 我们只为有效的新行加载数据
        # 组合掩码：行在范围内 且 列在范围内
        load_mask = row_mask[:, None] & col_mask[None, :]
        gathered_data = tl.load(src_ptrs, mask=load_mask, other=0.0)

        # 将数据“顺序”写入 output_ptr
        # 这是一个标准的连续写操作
        dst_ptrs = output_ptr + (new_indices[:, None] * num_cols + col_offsets[None, :])
        
        # 我们只向有效的输出行写入数据 (小于 num_out_tokens)
        store_mask = (new_indices < num_out_tokens)[:, None] & col_mask[None, :]
        tl.store(dst_ptrs, gathered_data, mask=store_mask)
    # =========================================================================
        
@triton.autotune(
    configs=[
        # 统一使用 triton.Config
        triton.Config({'BLOCK_SIZE_COLS': 16}, num_warps=2),
        triton.Config({'BLOCK_SIZE_COLS': 32}, num_warps=2),
        triton.Config({'BLOCK_SIZE_COLS': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_COLS': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_COLS': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE_COLS': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE_COLS': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE_COLS': 2048}, num_warps=8),
    ],
    key=['num_cols'],
)     


@triton.jit
def moe_recover_topk_kernel_triton(
    # 指针
    permuted_grad_ptr,      # 输入：permute 后的梯度 (对应 CUDA 的 input)
    unpermuted_grad_ptr,    # 输出：原始梯度 (对应 CUDA 的 unpermuted_output)
    row_id_map_ptr,         # 输入：行映射 (对应 CUDA 的 row_id_map)
    # 维度参数 (必须是 constexpr 以便编译时优化)
    original_num_rows: tl.constexpr,
    num_cols: tl.constexpr,
    num_topK: tl.constexpr,
    # 调优参数
    BLOCK_SIZE_COLS: tl.constexpr,
):
    """
    Triton 内核，严格对齐 moe_recover_topK_kernel 的 Gather-Sum 逻辑 (hasProb=false)。
    """
    # 1. 并行模型：每个 program 对应一个原始 token (等同于 CUDA 的 blockIdx.x)
    source_token_id = tl.program_id(axis=0)
    
    # 目标数据类型
    TARGET_DTYPE = unpermuted_grad_ptr.dtype.element_ty
    
    # 2. 按列分块循环 (等同于 CUDA 的 for 循环 `i += blockDim.x * kElementsPerAccess`)
    for col_start in range(0, num_cols, BLOCK_SIZE_COLS):
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE_COLS)
        col_mask = col_offsets < num_cols
        
        # 3. 初始化累加器 (等同于 CUDA 的 FragmentCompute frag_sum)
        #    【精度对齐】使用 tl.float32 进行累加，以匹配 CUDA 中的 TCompute
        accum_block = tl.zeros((BLOCK_SIZE_COLS,), dtype=tl.float32)
        
        # 4. 循环 K 个专家，进行 Gather-Sum (等同于 CUDA 的 for k 循环)
        for k in range(num_topK):
            # 4.1. 获取 permute 后的行号
            #      【逻辑对齐】CUDA 的 map 布局是 [num_topK, num_rows]，所以偏移是 k * num_rows
            map_offset = k * original_num_rows + source_token_id
            source_row = tl.load(row_id_map_ptr + map_offset)
            
            # 4.2. 创建掩码，处理无效专家 (source_row == -1)
            is_valid_expert = source_row != -1
            
            # 最终的加载掩码
            load_mask = col_mask & is_valid_expert
            
            # 4.3. Gather: 从 permuted_grad_ptr 加载数据
            #      【逻辑对齐】如果专家无效，加载 0
            grad_fragment = tl.load(
                permuted_grad_ptr + source_row * num_cols + col_offsets,
                mask=load_mask,
                other=0.0,
            )
            
            # 4.4. Sum: 累加到 float32 累加器中
            #      【精度对齐】将加载的片段转换为 float32 再进行累加
            accum_block += grad_fragment.to(tl.float32)
            
        # 5. 将累加完成的块写回全局内存 (等同于 CUDA 的最后一次 store)
        #    【逻辑对齐】每个 program 只写自己的输出行，因此不需要原子操作
        dest_ptr = unpermuted_grad_ptr + source_token_id * num_cols + col_offsets
        tl.store(
            dest_ptr,
            accum_block.to(TARGET_DTYPE), # 写回前转换回原始类型
            mask=col_mask
        ) 
@triton.jit
def permute_bwd_atomic_add_kernel(
    grad_output_ptr,
    grad_input_ptr,         
    sorted_row_id_ptr,
    num_permuted_rows,
    num_cols,
    top_k,
    BLOCK_SIZE_COLS: tl.constexpr,
):
    permuted_row_idx = tl.program_id(0)
    original_flat_index = tl.load(sorted_row_id_ptr + permuted_row_idx)
    original_row_idx = original_flat_index // top_k

    grad_output_row_ptr = grad_output_ptr + permuted_row_idx * num_cols
    grad_input_row_ptr = grad_input_ptr + original_row_idx * num_cols

    for col_block_start in range(0, num_cols, BLOCK_SIZE_COLS):
        col_offsets = col_block_start + tl.arange(0, BLOCK_SIZE_COLS)
        col_mask = col_offsets < num_cols
        
        grad_to_add = tl.load(
            grad_output_row_ptr + col_offsets,
            mask=col_mask, other=0.0
        )
        
       
        tl.atomic_add(
            grad_input_row_ptr + col_offsets,
            grad_to_add,
            mask=col_mask
        )




def moe_permute_topk_bwd_op_triton(
    permuted_act_grad: torch.Tensor,
    sorted_row_id: torch.Tensor,
    original_shape: tuple,
    num_topK: int
) -> torch.Tensor:
   
    # 1. 获取原始数据类型和设备
    original_dtype = permuted_act_grad.dtype
    device = permuted_act_grad.device
    act_grad=torch.zeros(original_shape, dtype=original_dtype, device=device)
    # 4. 获取维度信息并确保输入连续
    num_permuted_rows, num_cols = permuted_act_grad.shape
    if not permuted_act_grad.is_contiguous():
        permuted_act_grad = permuted_act_grad.contiguous()

    # 5. 启动内核。内核现在总是在一个支持原子加的缓冲区上操作。
    grid = (num_permuted_rows,)
    permute_bwd_atomic_add_kernel[grid](
        permuted_act_grad,
        act_grad, # 传入的可能是 float32 缓冲区
        sorted_row_id,
        num_permuted_rows=num_permuted_rows,
        num_cols=num_cols,
        top_k=num_topK,
    )
    return act_grad

# v2 内核：使用1D网格，优化循环结构
@triton.autotune(
    configs=[
        # 探索不同的块大小、warps和stages组合
        triton.Config({'BLOCK_SIZE_COLS': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE_COLS': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE_COLS': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE_COLS': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE_COLS': 512}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE_COLS': 1024}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE_COLS': 2048}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE_COLS': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE_COLS': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE_COLS': 8}, num_warps=4, num_stages=3),
    ],
    key=['num_cols', 'num_topK'],
)
@triton.jit
def permute_bwd_v2_kernel(
    go_ptr, inv_idx_ptr, out_ptr,
    num_rows, num_cols,
    num_topK: tl.constexpr,
    BLOCK_SIZE_COLS: tl.constexpr,
):
    # 1D 网格，每个程序处理一个输出行 (token)
    pid_m = tl.program_id(axis=0)

    # 循环处理当前行的所有列块
    for col_block_start in range(0, tl.cdiv(num_cols, BLOCK_SIZE_COLS)):
        col_offsets = col_block_start * BLOCK_SIZE_COLS + tl.arange(0, BLOCK_SIZE_COLS)
        col_mask = col_offsets < num_cols

        # 累加器初始化在列循环的开始
        acc = tl.zeros((BLOCK_SIZE_COLS,), dtype=out_ptr.dtype.element_ty)
        
        # 内部循环：累加来自 topK 个专家的梯度
        # tl.static_range 确保这个循环在编译时展开，没有运行时开销
        for k in tl.static_range(num_topK):
            # 加载需要 gather 的行索引
            perm_row = tl.load(inv_idx_ptr + pid_m * num_topK + k)
            is_valid = perm_row >= 0

            # 执行分散读取 (Gather)
            # 这里的 is_valid 会同时作用于 col_mask
            vals = tl.load(go_ptr + perm_row * num_cols + col_offsets, 
                           mask=col_mask & is_valid, 
                           other=0.0)
            acc += vals
        
        # 写回当前块的结果
        tl.store(out_ptr + pid_m * num_cols + col_offsets, acc, mask=col_mask)

@triton.jit
def build_inv_idx_kernel(
    sorted_row_id_ptr, # *i64, [num_permuted_rows]
    inv_idx_ptr,       # *i32, [num_rows, num_topK]
    num_permuted_rows,
    num_topK: tl.constexpr,
):
    """
    专用于构建 inv_idx 的 Triton 内核。
    每个程序处理 sorted_row_id 中的一个元素。
    """
    # 1D 网格，每个程序处理一个 permuted_row
    pid = tl.program_id(axis=0)
    
    # 边界检查
    if pid >= num_permuted_rows:
        return

    # 1. Fusion: 直接从 sorted_row_id 解码目标 token_id 和 k_id
    # 这替代了 torch.div 和 torch.fmod
    original_flat_index = tl.load(sorted_row_id_ptr + pid)
    target_token_id = original_flat_index // num_topK
    k_id = original_flat_index % num_topK

    # 2. Fusion: 计算要写入的值
    # perm_rows 的值就是当前处理的 permuted_row 的索引，即 pid
    value_to_write = pid.to(tl.int32)

    # 3. Fusion: 计算目标地址并写入
    # 这替代了重量级的 index_put_
    out_ptr = inv_idx_ptr + target_token_id * num_topK + k_id
    tl.store(out_ptr, value_to_write)
def moe_permute_topk_bwd_op_v2(permuted_act_grad, sorted_row_id, original_shape, num_topK):
    num_permuted_rows, num_cols = permuted_act_grad.shape
    num_rows = original_shape[0]
    device = permuted_act_grad.device
    
    # --- 优化区域开始 ---
    
    # 1. 只需一次 torch.full 来初始化 inv_idx
    # 这是必要的，因为我们的内核只会写入有效位置，其他位置需要保持 -1
    inv_idx = torch.full((num_rows, num_topK), -1, dtype=torch.int32, device=device)

    # 2. 启动我们的融合内核来填充 inv_idx
    # 网格大小为 num_permuted_rows，每个程序处理一个元素
    grid = (num_permuted_rows,)
    build_inv_idx_kernel[grid](
        sorted_row_id.to(torch.int64), # 确保输入是 int64
        inv_idx,
        num_permuted_rows,
        num_topK=num_topK,
    )
    
    act_grad = torch.empty(original_shape, dtype=permuted_act_grad.dtype, device=device)

    # 使用更简单的 1D 网格启动
    grid = (num_rows,)
    permute_bwd_v2_kernel[grid](
        permuted_act_grad, inv_idx, act_grad,
        num_rows, num_cols,
        num_topK=num_topK, # 传递为 constexpr
    )
    return act_grad
  


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_COLS': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE_COLS': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE_COLS': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE_COLS': 64},  num_warps=8, num_stages=3),
    ],
    key=['num_cols', 'num_topK'], 
)
@triton.jit
def permute_bwd_no_atomic_kernel(
    go_ptr,                 # *T   [num_permuted_rows, num_cols]  permuted_act_grad
    inv_idx_ptr,            # *i32 [num_rows, num_topK]           inv_idx[token, k] = permuted_row_idx 或 -1
    out_ptr,                # *T   [num_rows, num_cols]           act_grad
    num_rows,               # int
    num_cols,               # int
    num_topK: tl.constexpr, # constexpr，便于静态展开
    BLOCK_SIZE_COLS: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)  # token (row)
    pid_n = tl.program_id(axis=1)  # 列块
    if pid_m >= num_rows:
        return

    col_offsets = pid_n * BLOCK_SIZE_COLS + tl.arange(0, BLOCK_SIZE_COLS)
    col_mask = col_offsets < num_cols
    T_ELEM = go_ptr.dtype.element_ty
    acc = tl.zeros((BLOCK_SIZE_COLS,), dtype=T_ELEM)
    base_inv = inv_idx_ptr + pid_m * num_topK
    for k in tl.static_range(num_topK):
        perm_row = tl.load(base_inv + k)  # i32，可能为 -1
        is_valid = perm_row >= 0

        go_row_ptr = go_ptr + perm_row * num_cols
        frag = tl.load(go_row_ptr + col_offsets, mask=col_mask & is_valid, other=0)
        acc += frag
    out_row_ptr = out_ptr + pid_m * num_cols
    tl.store(out_row_ptr + col_offsets, acc, mask=col_mask)

def moe_permute_topk_bwd_op_triton_no_atomic(
    permuted_act_grad: torch.Tensor,   # [num_permuted_rows, num_cols], dtype = T
    sorted_row_id: torch.Tensor,       # [num_permuted_rows], 整型，值为 original_flat_index = token * num_topK + k
    original_shape: tuple,             # (num_rows, num_cols)
    num_topK: int,
    block_size_cols: int = 128,
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:
    """
    返回 act_grad，形状 original_shape，dtype 与 permuted_act_grad 相同。
    不使用原子加；对每个 token 按固定顺序累加 k=0..K-1。
    """
    assert permuted_act_grad.ndim == 2
    num_permuted_rows, num_cols = permuted_act_grad.shape
    num_rows = original_shape[0]
    device = permuted_act_grad.device
    T = permuted_act_grad.dtype

    # 确保连续
    if not permuted_act_grad.is_contiguous():
        permuted_act_grad = permuted_act_grad.contiguous()
    if not sorted_row_id.is_contiguous():
        sorted_row_id = sorted_row_id.contiguous()

    # 1) 构建 inv_idx[token, k] = permuted_row_idx（未出现的槽位为 -1）
    #    sorted_row_id[p] = token * num_topK + k
    flat = sorted_row_id.to(torch.int64)
    tokens = torch.div(flat, num_topK, rounding_mode='floor')
    ks = flat % num_topK
    perm_rows = torch.arange(num_permuted_rows, device=device, dtype=torch.int64)

    inv_idx = torch.full((num_rows, num_topK), -1, dtype=torch.int32, device=device)
    # 将 (tokens, ks) 的位置写成 perm_rows（自动广播形状一致）
    # 注意：若存在重复键，最后一次写入生效；假定输入合法且无重复
    inv_idx.index_put_((tokens.to(torch.long), ks.to(torch.long)), perm_rows.to(torch.int32), accumulate=False)

    # 2) 分配输出（TCompute = T）
    act_grad = torch.empty(original_shape, dtype=T, device=device)

    # 3) 启动 2D Kernel: (token, 列块)
    grid = lambda META: (
            num_rows,
            triton.cdiv(num_cols, META['BLOCK_SIZE_COLS']),
        )    
    permute_bwd_no_atomic_kernel[grid](
        permuted_act_grad,  # *T
        inv_idx,            # *i32 [num_rows, num_topK]
        act_grad,           # *T [num_rows, num_cols]
        num_rows,
        num_cols,
        num_topK,           # constexpr 以静态展开 k 循环
    )
    return act_grad

@triton.autotune(
    configs=[
        # H100 上较常用的两档列块；也给到不同并行与流水
        triton.Config({'BLOCK_SIZE_COLS': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_COLS': 256}, num_warps=8, num_stages=3),
        # 如需更细搜索可加 64 或更多组合
        # triton.Config({'BLOCK_SIZE_COLS': 64},  num_warps=4, num_stages=2),
        # triton.Config({'BLOCK_SIZE_COLS': 128}, num_warps=8, num_stages=2),
    ],
    key=['num_cols', 'num_topK'],
)
@triton.jit
def permute_bwd_no_atomic_kernel_autotuned(
    go_ptr,                 # *T   [num_permuted_rows, num_cols]  permuted_act_grad
    inv_idx_ptr,            # *i32 [num_rows, num_topK]           inv_idx[token, k] = permuted_row_idx 或 -1
    out_ptr,                # *T   [num_rows, num_cols]           act_grad
    num_rows,               # int
    num_cols,               # int
    num_topK: tl.constexpr, # constexpr，便于静态展开
    BLOCK_SIZE_COLS: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)  # token (row)
    pid_n = tl.program_id(axis=1)  # 列块

    if pid_m >= num_rows:
        return

    # 针对当前配置的列块总数；若外部用更小参考块发射了更多列块，这里直接早退
    num_pid_n_local = (num_cols + BLOCK_SIZE_COLS - 1) // BLOCK_SIZE_COLS
    if pid_n >= num_pid_n_local:
        return

    col_offsets = pid_n * BLOCK_SIZE_COLS + tl.arange(0, BLOCK_SIZE_COLS)
    col_mask = col_offsets < num_cols

    # TCompute = T：累加器用与输入相同的 dtype
    T_ELEM = go_ptr.dtype.element_ty
    acc = tl.zeros((BLOCK_SIZE_COLS,), dtype=T_ELEM)

    # 预取该 token 的基址
    base_inv = inv_idx_ptr + pid_m * num_topK

    # 固定顺序：k=0..K-1（严格保持加法顺序，避免非确定性）
    for k in tl.static_range(num_topK):
        perm_row = tl.load(base_inv + k)  # i32，可能为 -1
        is_valid = perm_row >= 0

        go_row_ptr = go_ptr + perm_row * num_cols
        frag = tl.load(go_row_ptr + col_offsets, mask=col_mask & is_valid, other=0)

        acc += frag  # 在 T 上累加

    # 写回该 token 的这一列块
    out_row_ptr = out_ptr + pid_m * num_cols
    tl.store(out_row_ptr + col_offsets, acc, mask=col_mask)


def moe_permute_topk_bwd_op_triton_no_atomic_autotuned(
    permuted_act_grad: torch.Tensor,   # [num_permuted_rows, num_cols], dtype = T
    sorted_row_id: torch.Tensor,       # [num_permuted_rows], 值为 token*num_topK + k
    original_shape: tuple,             # (num_rows, num_cols)
    num_topK: int,
    reference_block: int = 128,        # 参考块大小，用于计算统一网格；需<= autotune configs里的最小 BLOCK_SIZE_COLS
    warmup: int = 1,                   # 预热次数，避免把编译/搜索算进计时
) -> torch.Tensor:
    """
    Autotune 版本（H100 友好）。数值顺序保持不变（k 递增，TCompute = T）。
    注意：不要在调用时传 BLOCK_SIZE_COLS/num_warps/num_stages。
    """
    assert permuted_act_grad.ndim == 2
    num_permuted_rows, num_cols = permuted_act_grad.shape
    num_rows = original_shape[0]
    device = permuted_act_grad.device
    T = permuted_act_grad.dtype

    # 连续性
    if not permuted_act_grad.is_contiguous():
        permuted_act_grad = permuted_act_grad.contiguous()
    if not sorted_row_id.is_contiguous():
        sorted_row_id = sorted_row_id.contiguous()

    # 1) 构建 inv_idx[token, k] = permuted_row_idx（未出现设为 -1）
    flat = sorted_row_id.to(torch.int64)
    tokens = torch.div(flat, num_topK, rounding_mode='floor')
    ks = flat % num_topK
    perm_rows = torch.arange(num_permuted_rows, device=device, dtype=torch.int64)

    inv_idx = torch.full((num_rows, num_topK), -1, dtype=torch.int32, device=device)
    inv_idx.index_put_(
        (tokens.to(torch.long), ks.to(torch.long)),
        perm_rows.to(torch.int32),
        accumulate=False
    )

    # 2) 输出（TCompute = T）
    act_grad = torch.empty(original_shape, dtype=T, device=device)

    # 3) 统一 2D 网格：第二维用参考块大小计算，兼容所有 autotune 配置
    assert reference_block > 0
    grid_n_ref = triton.cdiv(num_cols, reference_block)
    grid = (num_rows, grid_n_ref)

    # 正式执行
    permute_bwd_no_atomic_kernel_autotuned[grid](
        permuted_act_grad,
        inv_idx,
        act_grad,
        num_rows,
        num_cols,
        num_topK,
    )
    return act_grad


@triton.jit
def row_map_kernel(
    sorted_row_id_ptr, row_id_map_ptr,
    num_rows, num_topK, num_out_tokens,
    N_ELEMENTS, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_ELEMENTS
    source_row_block = tl.load(sorted_row_id_ptr + offsets, mask=mask)
    source_token_id_block = source_row_block // num_topK
    source_topK_id_block = source_row_block % num_topK
    is_kept_mask = offsets < num_out_tokens
    value_to_write = tl.where(is_kept_mask, offsets, -1)
    output_offsets = source_topK_id_block * num_rows + source_token_id_block
    tl.store(row_id_map_ptr + output_offsets, value_to_write, mask=mask)
# 内核 2: SCATTER (复制) - 用于 permute_fwd
# 对应 C++ 的 moe_permute_topK_kernel (hasProb=false)
@triton.jit
def scatter_fwd_kernel(
    input_ptr, output_ptr, row_id_map_ptr,
    num_rows, num_cols, top_k,
    BLOCK_SIZE_COLS: tl.constexpr,
):
    pid_token = tl.program_id(axis=0)
    for col_start in range(0, num_cols, BLOCK_SIZE_COLS):
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE_COLS)
        col_mask = col_offsets < num_cols
        data = tl.load(input_ptr + pid_token * num_cols + col_offsets, mask=col_mask)
        for k in range(top_k):
            dest_row = tl.load(row_id_map_ptr + k * num_rows + pid_token)
            if dest_row != -1:
                tl.store(output_ptr + dest_row * num_cols + col_offsets, data, mask=col_mask)




@triton.jit
def gather_kernel_optimized(
    input_ptr, output_ptr, row_id_map_ptr, prob_ptr,
    num_rows: tl.constexpr, 
    num_cols: tl.constexpr,
    num_topK: tl.constexpr, 
    HAS_PROB: tl.constexpr,
    BLOCK_SIZE_COLS: tl.constexpr,
):
    source_token_id = tl.program_id(axis=0)
    TARGET_DTYPE = output_ptr.dtype.element_ty
    
    # --- 【步骤 1】向量化加载 K 维度的元数据 ---
    k_offsets = tl.arange(0, num_topK)
    map_offsets = k_offsets * num_rows + source_token_id
    source_rows = tl.load(row_id_map_ptr + map_offsets)
    
    probs = tl.zeros((num_topK,), dtype=tl.float32)
    if HAS_PROB:
        prob_offsets = source_token_id * num_topK + k_offsets
        probs = tl.load(prob_ptr + prob_offsets)

    # --- 【步骤 2】按列分块，计算并直接存储 ---
    for col_start in range(0, tl.cdiv(num_cols, BLOCK_SIZE_COLS)):
        col_idx = col_start * BLOCK_SIZE_COLS
        col_offsets = col_idx + tl.arange(0, BLOCK_SIZE_COLS)
        col_mask = col_offsets < num_cols

        # 2.1 向量化加载
        source_pointers = input_ptr + \
                          source_rows[:, None] * num_cols + \
                          col_offsets[None, :]
        
        load_mask = (source_rows[:, None] != -1) & col_mask[None, :]
        
        expert_frags = tl.load(source_pointers, mask=load_mask, other=0.0)
        
        # 2.2 向量化计算与归约
        if HAS_PROB:
            weighted_frags = expert_frags.to(tl.float32) * probs[:, None].to(tl.float32)
            accum_block = tl.sum(weighted_frags, axis=0) # shape: [BLOCK_SIZE_COLS]
        else:
            accum_block = tl.sum(expert_frags.to(tl.float32), axis=0) # shape: [BLOCK_SIZE_COLS]
        
        # --- 【步骤 3】直接将计算好的块写入输出指针 ---
        # 核心修正：
        # 1. 计算目标指针：output_ptr 的基地址 + 当前 token 的行偏移 + 当前块的列偏移
        # 2. 将 accum_block (float32) 转换为目标类型
        # 3. 使用 col_mask 进行安全存储
        
        output_pointers = output_ptr + source_token_id * num_cols + col_offsets
        
        tl.store(
            output_pointers, 
            accum_block.to(TARGET_DTYPE), 
            mask=col_mask
        )

@triton.jit
def gather_kernel(
    input_ptr, output_ptr, row_id_map_ptr, prob_ptr,
    num_rows, num_cols,
    num_topK: tl.constexpr, HAS_PROB: tl.constexpr, BLOCK_SIZE_COLS: tl.constexpr,
):
    source_token_id = tl.program_id(axis=0)
    TARGET_DTYPE = input_ptr.dtype.element_ty # 目标输出类型，例如 tl.float16
    
    for col_start in range(0, num_cols, BLOCK_SIZE_COLS):
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE_COLS)
        col_mask = col_offsets < num_cols
        accum = tl.zeros((BLOCK_SIZE_COLS,), dtype=TARGET_DTYPE)  # 使用 float32 作为累加器类型
        
        for k in range(num_topK):
            source_row = tl.load(row_id_map_ptr + k * num_rows + source_token_id)
            is_valid_expert = source_row != -1
            load_mask = col_mask & is_valid_expert
            expert_frag = tl.load(
                input_ptr + (source_row * num_cols + col_offsets),
                mask=load_mask, other=0.0
            ) 
            if HAS_PROB:
                prob_k = tl.load(prob_ptr + source_token_id * num_topK + k).to(TARGET_DTYPE)
                
                weighted= expert_frag * prob_k
                weighted1 = weighted
                accum += weighted1
            else: 
                accum+= expert_frag
        output_base_ptr = output_ptr + source_token_id * num_cols
        accum = accum.to(TARGET_DTYPE)  # 最终结果下转型为目标类型
        tl.store(output_base_ptr + col_offsets, accum, mask=col_mask)



# @triton.autotune(
#     configs=[
#         triton.Config({'BLOCK_SIZE_COLS': 128,'VEC_SIZE':4}, num_warps=4, num_stages=2),
#         triton.Config({'BLOCK_SIZE_COLS': 256,'VEC_SIZE':4}, num_warps=8, num_stages=3),
#         triton.Config({'BLOCK_SIZE_COLS': 64,'VEC_SIZE':4},  num_warps=4, num_stages=2),
#         triton.Config({'BLOCK_SIZE_COLS': 128,'VEC_SIZE':4}, num_warps=8, num_stages=2),
#     ],
#     key=['num_cols', 'num_topK'],  # 按列数与 K 选择最优配置并缓存
# )
# @triton.jit
# def scatter_bwd_kernel_vectorized(
#     # 指针
#     input_bwd_ptr, input_fwd_ptr, act_grad_ptr, prob_ptr, prob_grad_ptr, row_id_map_ptr,
#     # 维度
#     num_rows, num_cols,
#     # 常量
#     num_topK: tl.constexpr,
#     PADDED_TOP_K: tl.constexpr,
#     BLOCK_SIZE_COLS: tl.constexpr,
#     VEC_SIZE: tl.constexpr,
# ):
#     source_token_id = tl.program_id(axis=0)
    
#     col_vec_base_offsets = tl.arange(0, BLOCK_SIZE_COLS // VEC_SIZE)
#     vec_offsets = tl.arange(0, VEC_SIZE)

#     # --- 元数据预加载 (无变化) ---
#     k_offsets = tl.arange(0, PADDED_TOP_K)
#     k_mask = k_offsets < num_topK
#     map_offsets = k_offsets * num_rows + source_token_id
#     dest_rows = tl.load(row_id_map_ptr + map_offsets, mask=k_mask, other=-1)
#     prob_offsets = source_token_id * num_topK + k_offsets
#     probs = tl.load(prob_ptr + prob_offsets, mask=k_mask, other=0.0)
#     is_valid_expert_mask = dest_rows != -1
#     dot_prod_accum = tl.zeros((PADDED_TOP_K,), dtype=tl.float32)

#     # --- 主循环：在向量化的列块上迭代 ---
#     for col_vec_start in range(0, num_cols // VEC_SIZE, BLOCK_SIZE_COLS // VEC_SIZE):
#         col_vec_offsets = col_vec_start + col_vec_base_offsets
#         col_vec_mask = col_vec_offsets < num_cols // VEC_SIZE

#         col_offsets_v_start = col_vec_offsets[:, None] * VEC_SIZE
#         col_offsets = col_offsets_v_start + vec_offsets[None, :]
        
#         # --- 向量化加载 input_bwd ---
#         input_bwd_base_ptr = input_bwd_ptr + source_token_id * num_cols
#         input_bwd_vec_frag = tl.load(input_bwd_base_ptr + col_offsets, mask=col_vec_mask[:, None])
#         input_bwd_frag = tl.reshape(input_bwd_vec_frag, (BLOCK_SIZE_COLS,))

#         # --- 向量化 act_grad 计算和 Scatter ---
#         act_grad_frags = input_bwd_frag[None, :] * probs[:, None].to(input_bwd_ptr.dtype.element_ty)
#         act_grad_vec_frags = tl.reshape(act_grad_frags, (PADDED_TOP_K, BLOCK_SIZE_COLS // VEC_SIZE, VEC_SIZE))
        
#         act_grad_base_ptr = act_grad_ptr + dest_rows[:, None] * num_cols
        
#         # --- 【核心修正 1】: 扩展 act_grad_base_ptr 的维度以进行广播 ---
#         # 从 (PADDED_TOP_K, 1) -> (PADDED_TOP_K, 1, 1)
#         # 以便和 (1, BLOCK_SIZE_COLS/VEC, VEC_SIZE) 的 col_offsets 相加
#         act_grad_store_ptrs = act_grad_base_ptr[:, :, None] + col_offsets[None, :, :]
        
#         act_grad_mask = is_valid_expert_mask[:, None, None] & col_vec_mask[None, :, None]
#         tl.store(act_grad_store_ptrs, act_grad_vec_frags, mask=act_grad_mask) # act_grad_vec_frags 类型已匹配

#         # --- 向量化 input_fwd Gather 和点积计算 ---
#         input_fwd_base_ptr = input_fwd_ptr + dest_rows[:, None] * num_cols
        
#         # --- 【核心修正 2】: 对 input_fwd_base_ptr 应用相同的逻辑 ---
#         input_fwd_gather_ptrs = input_fwd_base_ptr[:, :, None] + col_offsets[None, :, :]
        
#         input_fwd_mask = is_valid_expert_mask[:, None, None] & col_vec_mask[None, :, None]
#         input_fwd_vec_frags = tl.load(input_fwd_gather_ptrs, mask=input_fwd_mask, other=0.0, cache_modifier=".cg")
#         input_fwd_frags = tl.reshape(input_fwd_vec_frags, (PADDED_TOP_K, BLOCK_SIZE_COLS))
        
#         low_prec_products = input_bwd_frag[None, :] * input_fwd_frags
#         partial_dot_prods = tl.sum(low_prec_products.to(tl.float32), axis=1)
#         dot_prod_accum += partial_dot_prods

#     # --- 最终存储 prob_grad (无变化) ---
#     final_prob_offsets = source_token_id * num_topK + tl.arange(0, PADDED_TOP_K)
#     final_store_mask = tl.arange(0, PADDED_TOP_K) < num_topK
#     tl.store(prob_grad_ptr + final_prob_offsets, dot_prod_accum, mask=final_store_mask)



@triton.autotune(
    configs=[
        # 积极探索更深的流水线
        triton.Config({'BLOCK_SIZE_COLS': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE_COLS': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE_COLS': 256}, num_warps=8, num_stages=5), # 更大块+更深流水线
        triton.Config({'BLOCK_SIZE_COLS': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE_COLS': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_COLS': 256}, num_warps=8, num_stages=3),
    ],
    key=['num_cols', 'num_topK'],
)
@triton.jit
def scatter_bwd_kernel_optimized(
    # 指针
    input_bwd_ptr, input_fwd_ptr, act_grad_ptr, prob_ptr, prob_grad_ptr, row_id_map_ptr,
    # 维度
    num_rows, num_cols,
    # 常量
    num_topK: tl.constexpr,
    PADDED_TOP_K: tl.constexpr, # 通常是 next_power_of_2(num_topK)
    BLOCK_SIZE_COLS: tl.constexpr,
):
    # 每个 program 仍然处理一个原始 token
    source_token_id = tl.program_id(axis=0)

    # --- 优化 1: 预加载所有 Top-K 元数据 ---
    # 一次性加载当前 token 对应的所有 topK 个目标行号和概率
    # k_offsets: [0, 1, 2, ..., PADDED_TOP_K-1]
    k_offsets = tl.arange(0, PADDED_TOP_K)
    k_mask = k_offsets < num_topK

    # 加载所有 topK 的目标行号 (dest_rows)
    # dest_rows 将是一个形状为 [PADDED_TOP_K] 的张量
    map_offsets = k_offsets * num_rows + source_token_id
    dest_rows = tl.load(row_id_map_ptr + map_offsets, mask=k_mask, other=-1)
    
    # 加载所有 topK 的概率 (probs)
    prob_offsets = source_token_id * num_topK + k_offsets
    probs = tl.load(prob_ptr + prob_offsets, mask=k_mask, other=0.0)

    # is_valid_expert_mask: [True, True, False, ...] (形状为 [PADDED_TOP_K])
    is_valid_expert_mask = dest_rows != -1

    # 初始化点积累加器 (形状为 [PADDED_TOP_K])
    dot_prod_accum = tl.zeros((PADDED_TOP_K,), dtype=tl.float32)

    # --- 主循环：在列上迭代 ---
    for col_start in range(0, num_cols, BLOCK_SIZE_COLS):
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE_COLS)
        col_mask = col_offsets < num_cols

        # 加载 input_bwd 的一个片段，这个片段将被广播给所有 topK 个 expert
        # input_bwd_frag 形状: [BLOCK_SIZE_COLS]
        input_bwd_frag = tl.load(
            input_bwd_ptr + source_token_id * num_cols + col_offsets,
            mask=col_mask, other=0.0
        )

        # --- 优化 2: 向量化 act_grad 的计算和 Scatter (存储) ---
        # act_grad_frags 形状: [PADDED_TOP_K, BLOCK_SIZE_COLS]
        act_grad_frags = input_bwd_frag[None, :] * probs[:, None].to(input_bwd_frag.dtype)
        
        # 构造一个 2D 的指针块来进行 scatter
        # act_grad_base_ptrs 形状: [PADDED_TOP_K, BLOCK_SIZE_COLS]
        act_grad_base_ptrs = act_grad_ptr + dest_rows[:, None] * num_cols + col_offsets[None, :]
        
        # 构造一个 2D 的掩码
        act_grad_mask = is_valid_expert_mask[:, None] & col_mask[None, :]
        
        tl.store(act_grad_base_ptrs, act_grad_frags.to(act_grad_ptr.dtype.element_ty), mask=act_grad_mask)

        # --- 优化 3: 向量化 input_fwd 的 Gather (加载) 和点积计算 ---
        # 构造一个 2D 的指针块来进行 gather
        # input_fwd_base_ptrs 形状: [PADDED_TOP_K, BLOCK_SIZE_COLS]
        input_fwd_base_ptrs = input_fwd_ptr + dest_rows[:, None] * num_cols + col_offsets[None, :]
        
        # 构造一个 2D 的掩码
        input_fwd_mask = is_valid_expert_mask[:, None] & col_mask[None, :]

        # --- 优化 4: 使用缓存优化 Gather ---
        # .cg (cache globally) 是不规则内存访问 (gather) 的理想选择
        input_fwd_frags = tl.load(
            input_fwd_base_ptrs,
            mask=input_fwd_mask, other=0.0,
            cache_modifier=".cg"
        )
        
        # 向量化计算点积
        # low_prec_products 形状: [PADDED_TOP_K, BLOCK_SIZE_COLS]
        low_prec_products = input_bwd_frag[None, :] * input_fwd_frags
        
        # 在列维度上进行 reduce sum，得到每个 expert 的部分点积
        # partial_dot_prods 形状: [PADDED_TOP_K]
        partial_dot_prods = tl.sum(low_prec_products.to(tl.float32), axis=1)
        
        # 累加到总点积中 (这里不需要掩码，因为无效的 expert 加载的是0)
        dot_prod_accum += partial_dot_prods

    final_prob_offsets = source_token_id * num_topK + tl.arange(0, PADDED_TOP_K)
    final_store_mask = tl.arange(0, PADDED_TOP_K) < num_topK
    tl.store(prob_grad_ptr + final_prob_offsets, dot_prod_accum, mask=final_store_mask)

@triton.autotune(
    configs=[
        # 覆盖常见列块与并行度（H100 友好）
        triton.Config({'BLOCK_SIZE_COLS': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_COLS': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE_COLS': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_COLS': 128}, num_warps=8, num_stages=2),
    ],
    key=['num_cols', 'num_topK'],  # 按列数与 K 选择最优配置并缓存
)
@triton.jit
def scatter_bwd_kernel(
    input_bwd_ptr, input_fwd_ptr, act_grad_ptr, prob_ptr, prob_grad_ptr, row_id_map_ptr,
    num_rows, num_cols,
    num_topK: tl.constexpr,
    PADDED_TOP_K: tl.constexpr,
    BLOCK_SIZE_COLS: tl.constexpr,
):
    source_token_id = tl.program_id(axis=0)
    
    dot_prod_accum = tl.zeros((PADDED_TOP_K,), dtype=tl.float32)

    for col_start in range(0, num_cols, BLOCK_SIZE_COLS):
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE_COLS)
        col_mask = col_offsets < num_cols

        input_bwd_frag = tl.load(
            input_bwd_ptr + source_token_id * num_cols + col_offsets,
            mask=col_mask, other=0.0
        )

        for k in range(num_topK):
            dest_row = tl.load(row_id_map_ptr + k * num_rows + source_token_id)
            is_valid_expert = dest_row != -1
            
            # --- act_grad 的计算部分 ---
            prob_k = tl.load(prob_ptr + source_token_id * num_topK + k)
            act_grad_frag = input_bwd_frag * prob_k.to(input_bwd_frag.dtype)
            
            act_grad_base_ptr = act_grad_ptr + dest_row * num_cols
            
            # ========================= 核心修正 =========================
            # 将原子加替换为直接存储 (Store)
            # 这与 C++ 的直接赋值 *(float4*)... = ...; 逻辑一致
            tl.store(
                act_grad_base_ptr + col_offsets,
                act_grad_frag.to(act_grad_ptr.dtype.element_ty), # 确保写入类型与缓冲区一致
                mask=col_mask & is_valid_expert
            )
            # ============================================================
            
            # --- prob_grad 的计算部分 (保持之前的修正) ---
            input_fwd_base_ptr = input_fwd_ptr + dest_row * num_cols
            input_fwd_frag = tl.load(
                input_fwd_base_ptr + col_offsets,
                mask=col_mask & is_valid_expert, other=0.0
            )
            
            # 修正后的高精度点积计算
            low_prec_product = input_bwd_frag * input_fwd_frag
            partial_dot_prod = tl.sum(low_prec_product.to(tl.float32))

            if is_valid_expert:
                dot_prod_accum = dot_prod_accum + tl.where(tl.arange(0, PADDED_TOP_K) == k, partial_dot_prod, 0.0)

    # --- 最后的存储逻辑 (保持不变) ---
    prob_offsets = source_token_id * num_topK + tl.arange(0, PADDED_TOP_K)
    store_mask = tl.arange(0, PADDED_TOP_K) < num_topK
    tl.store(prob_grad_ptr + prob_offsets, dot_prod_accum, mask=store_mask)


def moe_permute_topK_kernel_launcher_triton(
    FWD: bool,
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    sorted_row_id: torch.Tensor,
    row_id_map: torch.Tensor,
    prob: torch.Tensor,
    num_rows: int,
    num_topK: int,
    num_cols: int,
    num_out_tokens: int,
    prob_grad: torch.Tensor = None,
    input_fwd: torch.Tensor = None,
):
    grid = (num_rows,)
    BLOCK_SIZE_COLS = 64

    if FWD:
        if prob_grad is None:
            # 路径 1: permute_fwd (Scatter)
            # print("正在执行 C++ 逻辑: permute_fwd (Scatter)")
            n_elements = sorted_row_id.numel()
            map_grid = (triton.cdiv(n_elements, 1024),)
            row_map_kernel[map_grid](
                sorted_row_id, row_id_map, num_rows, num_topK, num_out_tokens, n_elements, BLOCK_SIZE=1024
            )
            scatter_fwd_kernel[grid](
                input_tensor, output_tensor, row_id_map, num_rows, num_cols, num_topK, BLOCK_SIZE_COLS=BLOCK_SIZE_COLS
            )
        else:
            # 执行 unpermute_bwd 
            padded_top_K = triton.next_power_of_2(num_topK)
            scatter_bwd_kernel_optimized[grid](
                input_tensor, input_fwd, output_tensor, prob, prob_grad, row_id_map,
                num_rows, num_cols, num_topK=num_topK, PADDED_TOP_K=padded_top_K,
            )
    else: # not FWD
        if prob is None:
            # 路径 4: unpermute_bwd (Gather, permute_fwd的梯度)
            # print("正在执行 C++ 逻辑: unpermute_bwd (Gather)")
            prob_placeholder = torch.empty(num_rows * num_topK, device=input_tensor.device, dtype=torch.float32)
            gather_kernel[grid](
                input_tensor, output_tensor, row_id_map, prob_ptr=prob_placeholder,
                num_rows=num_rows, num_cols=num_cols, num_topK=num_topK, HAS_PROB=False, BLOCK_SIZE_COLS=BLOCK_SIZE_COLS
            )
        else:
            # 路径 2: unpermute_fwd (Weighted Gather)
            # print("正在执行 C++ 逻辑: unpermute_fwd (Weighted Gather)")
            gather_kernel[grid](
                input_tensor, output_tensor, row_id_map, prob_ptr=prob,
                num_rows=num_rows, num_cols=num_cols, num_topK=num_topK, HAS_PROB=True, BLOCK_SIZE_COLS=BLOCK_SIZE_COLS
            )


def moe_permute_topk_op_triton_fused(
    input_tensor: torch.Tensor,
    indices: torch.Tensor,
    num_out_tokens: int,
    workspace: list[torch.Tensor],
    max_expanded_token_num: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """
    【最终修正版】调用单次全融合的 Triton 内核，严格复现 C++ moe_permute_topK_op 的行为。
    """
    # 1. 参数提取 (与原始版本完全相同)
    num_tokens = input_tensor.size(0)
    num_cols = input_tensor.size(1)
    num_topK = indices.size(1)
    device = input_tensor.device
    num_elements = num_tokens * num_topK

    # 2. 工作空间管理 (与原始版本完全相同)
    if not workspace or workspace[0].numel() < num_elements:
        if not workspace:
            print("Initializing workspace...")
        else:
            print(f"Resizing workspace from {workspace[0].numel()} to {max_expanded_token_num}...")
        workspace.clear()
        int32_options = {'dtype': torch.int32, 'device': device}
        int64_options = {'dtype': torch.int64, 'device': device}
        workspace.extend([
            torch.empty(max_expanded_token_num, **int32_options),
            torch.empty(max_expanded_token_num, **int32_options), # 这个可以优化掉，但为了对齐先留着
            torch.empty(max_expanded_token_num, **int64_options)
        ])

    # 3. 核心操作：排序 (与原始版本完全相同)
    flat_indices = indices.flatten()
    sorted_values_output_view = workspace[0][:num_elements]
    sorted_indices_output_view = workspace[2][:num_elements]
    sorted_values, sorted_indices = torch.sort(flat_indices)
    sorted_values_output_view.copy_(sorted_values)
    sorted_indices_output_view.copy_(sorted_indices)
    sorted_row_id_result = sorted_indices_output_view

    # 4. 定义元参数和分配输出 (与原始版本类似)
    dtype_to_block_size = {
        torch.float32: 4, torch.float16: 8, torch.bfloat16: 8,
        torch.float8_e4m3fn: 16, torch.float8_e5m2: 16,
    }
    BLOCK_SIZE_COLS = dtype_to_block_size.get(input_tensor.dtype, 8)
    BLOCK_SIZE_ROWS = 256 # 这是一个很好的通用值

    if num_out_tokens <= 0:
        num_out_tokens = num_elements
    
    permuted_output = torch.empty(num_out_tokens, num_cols, dtype=input_tensor.dtype, device=device)
    row_id_map = torch.empty(num_elements, dtype=torch.int32, device=device)

    # 5. 启动【单个融合内核】
    grid = (triton.cdiv(num_elements, BLOCK_SIZE_ROWS),)
    permute_fused_gather_and_map_kernel[grid](
        input_tensor,
        sorted_row_id_result,
        permuted_output,
        row_id_map,
        num_cols=num_cols,
        num_topK=num_topK,
        num_tokens=num_tokens,         # 传递 num_tokens
        num_out_tokens=num_out_tokens, # 传递 num_out_tokens
        N_ELEMENTS=num_elements,
        BLOCK_SIZE_ROWS=BLOCK_SIZE_ROWS,
        BLOCK_SIZE_COLS=BLOCK_SIZE_COLS,
    )
    
    # 6. 返回结果 (与原始版本完全相同)
    return permuted_output, row_id_map, sorted_row_id_result, workspace



#优化
def moe_permute_topk_op_triton_fused_v2(
    input_tensor: torch.Tensor,
    indices: torch.Tensor,
    num_out_tokens: int,
    workspace: list[torch.Tensor],
    max_expanded_token_num: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """
    【V2 - 优化版】调用自动调优的、单次全融合的 Triton 内核。
    """
    # 步骤 1, 2, 3: 参数提取、工作空间管理、排序 (与之前完全相同)
    num_tokens = input_tensor.size(0)
    num_cols = input_tensor.size(1)
    num_topK = indices.size(1)
    device = input_tensor.device
    num_elements = num_tokens * num_topK

    if not workspace or workspace[0].numel() < num_elements:
        workspace.clear()
        int32_options = {'dtype': torch.int32, 'device': device}
        int64_options = {'dtype': torch.int64, 'device': device}
        workspace.extend([
            torch.empty(max_expanded_token_num, **int32_options),
            torch.empty(max_expanded_token_num, **int32_options),
            torch.empty(max_expanded_token_num, **int64_options)
        ])

    # flat_indices = indices.flatten()
    flat_indices=indices.view(-1)
    sorted_values_output_view = workspace[0][:num_elements]
    sorted_indices_output_view = workspace[2][:num_elements]
    # sorted_values, sorted_indices = torch.sort(flat_indices)
    # sorted_values_output_view.copy_(sorted_values)
    # sorted_indices_output_view.copy_(sorted_indices)
    torch.sort(flat_indices,out=(sorted_values_output_view, sorted_indices_output_view))
    sorted_row_id_result = sorted_indices_output_view

    # 步骤 4: 分配输出 (与之前完全相同)
    if num_out_tokens <= 0:
        num_out_tokens = num_elements
    permuted_output = torch.empty(num_out_tokens, num_cols, dtype=input_tensor.dtype, device=device)
    row_id_map = torch.empty(num_elements, dtype=torch.int32, device=device)

    # 步骤 5: 启动【新的 V2 优化内核】
    grid = lambda META: (triton.cdiv(num_elements, META['BLOCK_SIZE_ROWS']),)
    
    # 调用新的内核
    permute_fused_gather_and_map_kernel_v2[grid](
        input_tensor,
        sorted_row_id_result,
        permuted_output,
        row_id_map,
        num_cols,
        num_topK,
        num_tokens,
        num_out_tokens,
        N_ELEMENTS=num_elements,
    )
    
    return permuted_output, row_id_map, sorted_row_id_result, workspace



def moe_permute_topk_op_triton(
    input_tensor: torch.Tensor,
    indices: torch.Tensor,
    num_out_tokens: int,
    workspace: list[torch.Tensor],
    max_expanded_token_num: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
   

    # 1. 参数提取
    num_tokens = input_tensor.size(0)
    num_cols = input_tensor.size(1)
    num_topK = indices.size(1)
    device = input_tensor.device
    num_elements = num_tokens * num_topK

    # 2. 工作空间管理 (完全模仿C++逻辑，并增加动态调整)
    # ======================= 核心修正：动态工作空间 =======================
    # 检查工作空间是否需要初始化或扩容
    if not workspace or workspace[0].numel() < num_elements:
        if not workspace:
            print("Initializing workspace...")
        else:
            print(f"Resizing workspace from {workspace[0].numel()} to {max_expanded_token_num}...")
        
        # 清空旧的工作空间（如果有的话）
        workspace.clear()
        
        int32_options = {'dtype': torch.int32, 'device': device, 'requires_grad': False}
        int64_options = {'dtype': torch.int64, 'device': device, 'requires_grad': False}
        
        sorted_indices_buffer = torch.empty(max_expanded_token_num, **int32_options)
        row_id_buffer = torch.arange(0, max_expanded_token_num, **int32_options) # 这一行其实可以优化掉
        sorted_row_id_buffer = torch.empty(max_expanded_token_num, **int64_options)
        
        workspace.extend([sorted_indices_buffer, row_id_buffer, sorted_row_id_buffer])
    # ======================================================================

    # 3. 核心操作：排序
    flat_indices = indices.flatten()
    
    # 从工作区获取视图，现在可以确保其大小足够
    sorted_values_output_view = workspace[0][:num_elements]
    sorted_indices_output_view = workspace[2][:num_elements]
    
    # 执行排序，不使用 out= 参数以避免 UserWarning
    sorted_values, sorted_indices = torch.sort(flat_indices)
    
    # 将结果复制到工作区视图中
    sorted_values_output_view.copy_(sorted_values)
    sorted_indices_output_view.copy_(sorted_indices)
    
    sorted_row_id_result = sorted_indices_output_view

    # ... (4, 5, 6, 7 部分与上一条回复中的修正保持一致) ...
    dtype_to_block_size = {
        torch.float32: 4, torch.float16: 8, torch.bfloat16: 8,
        torch.float8_e4m3fn: 16, torch.float8_e5m2: 16,
    }
    block_size_cols = dtype_to_block_size.get(input_tensor.dtype, 8)
    
    if num_out_tokens <= 0:
        num_out_tokens = num_elements
    
    permuted_output = torch.empty(num_out_tokens, num_cols, dtype=input_tensor.dtype, device=device)
    row_id_map = torch.empty(num_elements, dtype=torch.int32, device=device)

    map_grid = (triton.cdiv(num_elements, 1024),)
    row_map_kernel[map_grid](
        sorted_row_id_ptr=sorted_row_id_result,
        row_id_map_ptr=row_id_map,
        num_rows=num_tokens,
        num_topK=num_topK,
        num_out_tokens=num_out_tokens,
        N_ELEMENTS=num_elements,
        BLOCK_SIZE=1024,
    )
    
    scatter_grid = (num_tokens,)
    scatter_fwd_kernel[scatter_grid](
        input_ptr=input_tensor,
        output_ptr=permuted_output,
        row_id_map_ptr=row_id_map,
        num_rows=num_tokens,
        num_cols=num_cols,
        top_k=num_topK,
        BLOCK_SIZE_COLS=block_size_cols,
    )
    
    return permuted_output, row_id_map, sorted_row_id_result, workspace
# =====================================================================
# 新增: C++ moe_recover_topK_op 的 Triton OP 实现   op2
# =====================================================================
def moe_recover_topk_op_triton(
    input_tensor: torch.Tensor,
    row_id_map: torch.Tensor,
    prob: torch.Tensor, # 可以是 None
    num_tokens: int,
    num_topK: int
) -> torch.Tensor:
    # 1. 从输入张量获取维度和类型信息 (与C++逻辑一致)
    num_cols = input_tensor.size(1)
    dtype = input_tensor.dtype
    device = input_tensor.device

    # 2. 分配输出缓冲区 (与C++逻辑一致)
    unpermuted_output = torch.empty(
        (num_tokens, num_cols), dtype=dtype, device=device, requires_grad=False
    )
    # print("prob", prob)
    moe_permute_topK_kernel_launcher_triton(
        FWD=False,  # <-- 关键！设置为False来执行recover/gather逻辑
        input_tensor=input_tensor,
        output_tensor=unpermuted_output,
        sorted_row_id=None, # recover/gather操作不需要此参数
        row_id_map=row_id_map,
        prob=prob,
        num_rows=num_tokens,
        num_topK=num_topK,
        num_cols=num_cols,
        num_out_tokens=0, # recover/gather操作不需要此参数
        prob_grad=None,   # 非梯度计算
        input_fwd=None,   # 非梯度计算
    )

    # 4. 返回填充好的输出张量 (与C++逻辑一致)
    return unpermuted_output



# --- PyTorch 参考实现 (保持不变) ---
def permute_fwd_pytorch(X, sorted_row_id, num_rows, num_topK, num_out_tokens):
    row_id_map = torch.full((num_rows * num_topK,), -1, dtype=torch.int32, device='cuda')
    for i in range(sorted_row_id.numel()):
        val = sorted_row_id[i]
        token_id, k_id = val // num_topK, val % num_topK
        if i < num_out_tokens:
            row_id_map[k_id * num_rows + token_id] = i
    Xp = torch.zeros(num_out_tokens, X.shape[1], device='cuda', dtype=X.dtype)
    for i in range(num_rows):
        for k in range(num_topK):
            dest_row = row_id_map[k * num_rows + i]
            if dest_row != -1:
                Xp[dest_row] = X[i]
    return Xp, row_id_map

def unpermute_fwd_pytorch(Yp, row_id_map, probs, num_rows, num_topK):
    Y = torch.zeros(num_rows, Yp.shape[1], device='cuda', dtype=Yp.dtype)
    for i in range(num_rows):
        for k in range(num_topK):
            source_row = row_id_map[k * num_rows + i]
            if source_row != -1:
                Y[i] += Yp[source_row] * probs[i * num_topK + k]
    return Y

def unpermute_bwd_pytorch(grad_Xp, row_id_map, num_rows, num_topK):
    grad_X = torch.zeros(num_rows, grad_Xp.shape[1], device='cuda', dtype=grad_Xp.dtype)
    for i in range(num_rows):
        for k in range(num_topK):
            source_row = row_id_map[k * num_rows + i]
            if source_row != -1:
                grad_X[i] += grad_Xp[source_row]
    return grad_X

def permute_bwd_pytorch(grad_Y, Yp, row_id_map, probs, num_rows, num_topK):
    grad_Yp = torch.zeros_like(Yp)
    grad_prob = torch.zeros(num_rows * num_topK, device='cuda', dtype=probs.dtype)
    for i in range(num_rows):
        for k in range(num_topK):
            dest_row = row_id_map[k * num_rows + i]
            if dest_row != -1:
                grad_Yp[dest_row] += grad_Y[i] * probs[i * num_topK + k]
                grad_prob[i * num_topK + k] = torch.dot(grad_Y[i].float(), Yp[dest_row].float())
    return grad_Yp, grad_prob


# in moe_triton_kernels.py

def moe_recover_topk_bwd_op_triton(
    input_bwd: torch.Tensor,
    input_fwd: torch.Tensor,
    row_id_map: torch.Tensor,
    prob: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:

    num_tokens = prob.size(0)
    num_topK = prob.size(1)
    num_cols = input_bwd.size(1)


    act_grad = torch.zeros_like(input_fwd, dtype=torch.float32)
    # =========================================================

    prob_grad = torch.empty_like(prob, dtype=torch.float32)

    moe_permute_topK_kernel_launcher_triton(
        FWD=True,
        input_tensor=input_bwd,
        output_tensor=act_grad, 
        sorted_row_id=None,
        row_id_map=row_id_map,
        prob=prob,
        num_rows=num_tokens,
        num_topK=num_topK,
        num_cols=num_cols,
        num_out_tokens=0,
        prob_grad=prob_grad,
        input_fwd=input_fwd
    )

    # 4. 返回填充好的梯度张量元组
    # 注意：这里的 act_grad 仍然是 float32 类型
    return act_grad, prob_grad

# def simulate_expert_capacity(
#     indices: torch.Tensor,
#     probs: torch.Tensor,
#     expert_capacity: int
# ) -> torch.Tensor:
#     """
#     Simulates expert capacity filtering.
#     It takes ideal indices and probs, and returns a new indices tensor
#     where some choices are replaced with -1 to simulate being dropped.
#     """
#     num_tokens, num_topK = indices.shape
#     num_experts = indices.max().item() + 1
    
#     # 创建一个包含所有选择信息的列表: (expert_id, prob, token_id, k_id)
#     all_choices = []
#     for token_id in range(num_tokens):
#         for k_id in range(num_topK):
#             expert_id = indices[token_id, k_id].item()
#             prob = probs[token_id, k_id].item()
#             all_choices.append((expert_id, prob, token_id, k_id))
            
#     # 按专家ID分组
#     expert_assignments = [[] for _ in range(num_experts)]
#     for choice in all_choices:
#         expert_id = choice[0]
#         expert_assignments[expert_id].append(choice)
        
#     # 对每个专家的候选项按概率排序，并丢弃超出容量的
#     modified_indices = indices.clone()
#     for expert_id in range(num_experts):
#         assignments = expert_assignments[expert_id]
#         # 按概率降序排序
#         assignments.sort(key=lambda x: x[1], reverse=True)
        
#         # 如果候选项超过容量，则将多余的标记为-1
#         if len(assignments) > expert_capacity:
#             for choice_to_drop in assignments[expert_capacity:]:
#                 _, _, token_id, k_id = choice_to_drop
#                 modified_indices[token_id, k_id] = -1
                
#     return modified_indices

# # ==============================================================================
# #  Step 3: Main Test Logic
# # ==============================================================================

# if __name__ == "__main__":
#     # --- 3.1: 测试参数设置 ---
#     num_token = 8
#     num_cols = 4
#     num_expert = 4
#     num_topK = 2
#     expert_capacity = 3 # 每个专家最多处理3个令牌，总容量12，总请求16，必有丢弃
#     device = 'cuda'
#     torch.manual_seed(42) # 固定随机种子以保证结果可复现

#     # --- 3.2: 生成原始数据 (用户提供的方法) ---
#     input_tensor = (torch.arange(num_token, device=device).float() + 1).view(-1, 1).repeat(1, num_cols)
#     if num_token > 0:
#         indices = torch.stack([torch.randperm(num_expert)[:num_topK] for _ in range(num_token)])
#     else:
#         indices = torch.empty((num_token, num_topK))
#     indices = indices.to(torch.int32).cuda()
    
#     probs = torch.rand(num_token, num_topK).cuda()
#     # 注意：这里的概率归一化不是必须的，我们只关心相对大小
    
#     print("--- 原始生成数据 ---")
#     print("Input Tensor (first 4 rows):\n", input_tensor[:4])
#     print("Original Indices:\n", indices)

#     # --- 3.3: 模拟专家容量过滤，生成包含-1的indices ---
#     modified_indices = simulate_expert_capacity(indices, probs, expert_capacity)
#     num_negative_one_in_indices = (modified_indices == -1).sum().item()
#     num_out_tokens = modified_indices.numel() - num_negative_one_in_indices
    
#     print("\n--- 模拟容量限制后 ---")
#     print("Modified Indices (with -1 for dropped tokens):\n", modified_indices)
#     print(f"Total tokens to permute (num_out_tokens): {num_out_tokens}")
#     print(f"Dropped tokens (num_negative_one_in_indices): {num_negative_one_in_indices}")

#     # --- 3.4: 在PyTorch中计算Ground Truth ---
#     # 1. 模拟Triton的排序步骤
#     flat_indices = modified_indices.flatten()
#     sorted_experts, sorted_row_id = torch.sort(flat_indices)
    
#     # 2. 计算Ground Truth Permuted Output
#     valid_original_flat_indices = sorted_row_id[num_negative_one_in_indices:]
#     valid_original_token_ids = valid_original_flat_indices // num_topK
#     ground_truth_permuted_output = input_tensor[valid_original_token_ids]
    
#     # 3. 计算Ground Truth Row ID Map
#     ground_truth_map = torch.full((num_topK, num_token), -1, dtype=torch.int32, device=device)
#     target_indices = torch.arange(num_out_tokens, dtype=torch.int32, device=device) 
#     original_token_ids = valid_original_flat_indices // num_topK
#     original_k_ids = valid_original_flat_indices % num_topK
#     ground_truth_map[original_k_ids, original_token_ids] = target_indices

#     # --- 3.5: 调用V3 Triton内核 ---
#     workspace = []
#     permuted_output, row_id_map, _, _ = moe_permute_topk_op_triton_fused_v3(
#         input_tensor,
#         modified_indices,
#         num_out_tokens,
#         workspace,
#         max_expanded_token_num=num_token*num_topK,
#         num_negative_one_in_indices=num_negative_one_in_indices
#     )
    
#     # --- 3.6: 验证结果 ---
#     print("\n--- 验证 ---")
#     try:
#         assert torch.allclose(permuted_output, ground_truth_permuted_output)
#         print("✅ Permuted Output (数据置换) 结果正确！")
#     except AssertionError:
#         print("❌ Permuted Output (数据置换) 结果错误！")
#         print("Expected:\n", ground_truth_permuted_output)
#         print("Got:\n", permuted_output)

#     try:
#         assert torch.equal(row_id_map, ground_truth_map)
#         print("✅ Row ID Map (索引图) 结果正确！")
#     except AssertionError:
#         print("❌ Row ID Map (索引图) 结果错误！")
#         print("Expected:\n", ground_truth_map)
#         print("Got:\n", row_id_map)
