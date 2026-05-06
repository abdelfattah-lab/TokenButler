
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_N': 64}, num_warps=4),
        triton.Config({'BLOCK_N': 32}, num_warps=4),
    ],
    key=['D'],
)
@triton.jit
def score_int8_kernel(
    Q_ptr,      # [Batch, Heads, 1, D]  (int8)
    K_ptr,      # [Batch, Heads, Limit, D] (int8)
    Out_ptr,    # [Batch, Heads, Limit] (float16/bfloat16)
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_ob, stride_oh, stride_on,
    limit,      # Dimension Limit
    D,          # Dimension D (dDash), must be power of 2 (16)
    BLOCK_N: tl.constexpr,
):
    # Map grid to logical indices
    # Grid is (Batch * Heads, CEIL_DIV(Limit, BLOCK_N))
    pid_batch_head = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Reconstruct batch and head indices
    # We flatten Batch and Heads into one dimension for the grid
    # Strides will handle the addressing
    
    # Q offsets:
    # Q acts as a broadcasted vector for this (Batch, Head) pair
    # Q ptr is already offset by (batch_idx * stride_qb + head_idx * stride_qh) in the wrapper usually, 
    # but let's do full calculation here for clarity if passed raw pointers.
    # Actually, to simplify, let's assume pointers are passed pointing to the start of the tensor
    
    # Note: To avoid division in kernel for recovering batch/head, we can just treat independent problems.
    # But we need stride info.
    # Let's assume the grid 0 dimension is mapping to (Batch * Heads).
    # We can use the strides directly from the Q tensor to find the Q vector.
    
    # Current Problem (Batch, Head) Base Pointers
    # Q[pid_batch_head, 0, :]
    off_q_base = pid_batch_head * stride_qh # Assumes packed layout [B, H, 1, D] -> stride_qh covers B*H logic if flattened? 
    # Wait, if Q is [B, H, 1, D], stride_qh is H*1*D? No.
    # stride_qb = H*1*D, stride_qh = 1*D.
    # We need to decompose pid_batch_head into b and h?
    # Or just rely on the fact that Q and K and Out are virtually flattened on the first two dims.
    # Yes, if we pass inputs as [B*H, 1, D] and [B*H, Limit, D], then:
    # stride_q_problem = stride_qh (if real shape was B,H,...)
    
    # Let's assume the inputs are viewed as [Total_Heads, ...] inside the wrapper. 
    # That simplifies this kernel to just:
    # Q: [Total_Heads, 1, D]
    # K: [Total_Heads, Limit, D]
    # Out: [Total_Heads, Limit]
    
    off_problem = pid_batch_head
    
    # Q Vector Pointers
    # Q is [Total_Heads, 1, D]. Stride 0 is "stride per problem". Stride 2 is 1 (element).
    q_ptr = Q_ptr + off_problem * stride_qb
    
    # K Matrix Pointers
    # K is [Total_Heads, Limit, D]
    k_ptr_base = K_ptr + off_problem * stride_kb
    
    # Out Vector Pointers
    out_ptr_base = Out_ptr + off_problem * stride_ob
    
    # Block N loop (Parallelized by grid(1))
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < limit
    
    # Load Q (1, D)
    # D is small (16), so we load it as a single block
    # offs_d = tl.arange(0, 16) # Hardcoded D=16 based on problem desc
    offs_d = tl.arange(0, 16) # We can make this dynamic or constexpr later
    
    # Load Q row
    q_vals = tl.load(q_ptr + offs_d * stride_qd) # [16]
    
    # Load K block (BLOCK_N, D)
    # k_ptr points to start of this problem's K matrix
    # Rows are 'n' (0..Limit), Cols are 'd' (0..D)
    # stride_kn is stride between rows (Limit dim)
    # stride_kd is stride between cols (D dim)
    
    # Pointers to K elements: base + n * stride_kn + d * stride_kd
    k_ptrs = k_ptr_base + (offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd)
    k_vals = tl.load(k_ptrs, mask=(mask_n[:, None]), other=0.0) # [BLOCK_N, 16] (int8)
    
    # Compute Dot Product
    # Triton handles int8 dot product automatically if we cast to float for accumulation?
    # Or use tl.dot?
    # tl.dot requires blocks.
    # Since D=16 is small, we can just do element-wise mul and sum reduction along D-dimension.
    
    # Cast to float (or bf16) for computation
    q_f = q_vals.to(tl.float32)
    k_f = k_vals.to(tl.float32)
    
    # Element-wise multiply: [16] * [BLOCK_N, 16] -> [BLOCK_N, 16] (broadcasting Q)
    prod = q_f[None, :] * k_f
    
    # Sum along D dimension
    scores = tl.sum(prod, axis=1) # [BLOCK_N]
    
    # Store
    out_ptrs = out_ptr_base + offs_n * stride_on
    tl.store(out_ptrs, scores.to(Out_ptr.dtype.element_ty), mask=mask_n)


def tokenbutler_score_int8_triton(q, k, dDash=16):
    """
    Compute dot product scores between Q and K using Triton Int8 kernel.
    
    Args:
        q: [Batch, Heads, 1, D] (int8)
        k: [Batch, Heads, Limit, D] (int8)
        dDash: Dimension size (default 16)
        
    Returns:
        scores: [Batch, Heads, Limit] (bfloat16)
    """
    # Check shapes
    if q.ndim == 3:
        # Assumed [Total_Heads, 1, D]
        # Treat as B=Total, H=1
        Total, _, D_q = q.shape
        Total_k, Limit, D_k = k.shape
        assert Total == Total_k
        assert D_q == dDash
        assert D_k == dDash
        
        q_flat = q
        k_flat = k
        B = Total
        H = 1
        D = D_q
        
    else:
        assert q.ndim == 4
        assert k.ndim == 4
        assert q.shape[-1] == dDash
        assert k.shape[-1] == dDash
        assert q.dtype == torch.int8
        assert k.dtype == torch.int8
        
        B, H, _, D = q.shape
        _, _, Limit, _ = k.shape
        
        q_flat = q.view(B*H, 1, D)
        k_flat = k.view(B*H, Limit, D)
        
    scores = torch.empty((B*H, Limit), device=q.device, dtype=torch.bfloat16)
    out_flat = scores
    
    grid = lambda META: (B * H, triton.cdiv(Limit, META['BLOCK_N']))
    
    score_int8_kernel[grid](
        q_flat, k_flat, out_flat,
        q_flat.stride(0), 1, 1, q_flat.stride(2), # stride_qb (problem), stride_qh (dummy), stride_qm, stride_qd
        k_flat.stride(0), 1, k_flat.stride(1), k_flat.stride(2), # stride_kb, stride_kh, stride_kn, stride_kd
        out_flat.stride(0), 1, out_flat.stride(1), # stride_ob, stride_oh, stride_on
        limit=Limit,
        D=D,
    )
    
    return scores
