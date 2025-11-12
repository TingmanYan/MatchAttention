import torch
import torch.nn as nn
from typing import List, Tuple


@torch.library.custom_op("match_attention::fused_forward_ops", mutates_args={"output", "attn_out"})
def fused_forward_ops(
    max_offset: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    attn_out: torch.Tensor,
    H: int,
    W: int,
    win_r: List[int],
    attn_num: int,
    attn_type: str,
    scale: float
) -> None:
    """
    Opaque custom op for fused forward pass that prevents torch.compile tracing.
    
    This wrapper ensures that torch.compile treats this as an opaque operation
    and doesn't try to trace into the CUDA kernel internals.
    """
    # Call the original CUDA extension
    try:
        import match_attention
        match_attention.fused_forward(
            max_offset, q, k, v, output, attn_out,
            H, W, win_r, attn_num, attn_type, scale
        )
    except ImportError:
        # Fallback to torch.ops if direct import fails
        torch.ops.match_attention.fused_forward(
            max_offset, q, k, v, output, attn_out,
            H, W, win_r, attn_num, attn_type, scale
        )


@fused_forward_ops.register_fake
def _(max_offset, q, k, v, output, attn_out, H, W, win_r, attn_num, attn_type, scale):
    """
    Fake implementation for torch.compile that defines tensor shapes and dtypes
    without actually executing the kernel.
    """
    # Validate input shapes
    B, N, C = q.shape
    h = max_offset.size(2)
    
    # Ensure output tensors have correct shapes
    torch._check(output.shape == (B, N, C), lambda: f"output shape mismatch: expected {(B, N, C)}, got {output.shape}")
    torch._check(attn_out.shape == (B, N, h, attn_num), lambda: f"attn_out shape mismatch: expected {(B, N, h, attn_num)}, got {attn_out.shape}")
    
    # Ensure output tensors have correct dtypes and devices
    torch._check(output.dtype == q.dtype, lambda: f"output dtype mismatch: expected {q.dtype}, got {output.dtype}")
    torch._check(attn_out.dtype == q.dtype, lambda: f"attn_out dtype mismatch: expected {q.dtype}, got {attn_out.dtype}")
    torch._check(output.device == q.device, lambda: f"output device mismatch: expected {q.device}, got {output.device}")
    torch._check(attn_out.device == q.device, lambda: f"attn_out device mismatch: expected {q.device}, got {attn_out.device}")
    
    return None


class MF_FusedForwardOps(nn.Module):
    """
    Opaque MatchAttention fused forward, optimized for torch.compile
    
    This version uses torch.library.custom_op to create opaque custom operators,
    preventing torch.compile from tracing into CUDA kernel internals.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        max_offset: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        H: int,
        W: int,
        win_r: List[int],
        attn_num: int,
        attn_type: str = 'l1_norm',
        scale: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fused forward
        
        Args:
            max_offset: Offset tensor with shape [B, N, h, 2]
            q: Query tensor with shape [B, N, C]
            k: Key tensor with shape [B, N, C]
            v: Value tensor with shape [B, N, C]
            H: Feature map height
            W: Feature map width
            win_r: Window radius [r_h, r_w]
            attn_num: Number of attention heads
            attn_type: Attention type ('l1_norm' or 'l2_norm')
            scale: Scale factor
            
        Returns:
            output: Output features with shape [B, N, C]
            attn_out: Attention weights with shape [B, N, h, attn_num]
        """
        B, N, C = q.shape
        h = max_offset.size(2)
        
        # Create output tensors
        output = torch.zeros_like(v)
        attn_out = q.new_zeros([B, N, h, attn_num])
        
        # Call opaque custom operator
        fused_forward_ops(
            max_offset, q, k, v, output, attn_out,
            H, W, win_r, attn_num, attn_type, scale
        )
        
        return output, attn_out