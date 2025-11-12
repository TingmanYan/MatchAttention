import torch

def compute_bilinear_weights(grid):
    """
    Compute bilinear weights for BilinearSoftmax
    Args:
        grid: [..., 2], (x, y)
    Returns:
        weights: [..., 4], [nw, ne, sw, se] 
    """
    x = grid[..., 0]
    y = grid[..., 1]
    
    x0 = torch.floor(x)
    y0 = torch.floor(y)
    
    dx = x - x0
    dy = y - y0
    
    nw = (1 - dx) * (1 - dy)
    ne = dx * (1 - dy)      
    sw = (1 - dx) * dy      
    se = dx * dy            
    
    weights = torch.stack([nw, ne, sw, se], dim=-1)
    
    return weights

def compute_match_attention(q, k, m_id, win_r, H, W):
    """
    Args:
        q: [B, N, h, C]   # Query tensor
        k: [B, N, h, C]   # Key tensor
        m_id: [B, N, h, 2] # Sampling centers, last dim is (x, y)
        r: int             # Sampling window radius
        H: int             # Height
        W: int             # Width
    
    Returns:
        output: [B, N, h, M] where M = (2*win_r[0]+2)*(2*win_r[1]+2)
    """
    B, N, h, C = q.shape
    M = (2*win_r[0] + 2)*(2*win_r[1] + 2)
    
    dx = torch.arange(-win_r[0], win_r[0] + 2, device=q.device, dtype=torch.long)
    dy = torch.arange(-win_r[1], win_r[1] + 2, device=q.device, dtype=torch.long)
    dy, dx = torch.meshgrid(dy, dx, indexing='ij')
    offsets = torch.stack((dx, dy), dim=-1).reshape(M, 2)  # [M, 2]
    
    centers = m_id.unsqueeze(3)          # [B, N, h, 1, 2]
    offsets = offsets.view(1, 1, 1, M, 2) # [1, 1, 1, M, 2]
    coords = centers + offsets            # [B, N, h, M, 2]
    
    x_coords = coords[..., 0]  # [B, N, h, M]
    y_coords = coords[..., 1]  # [B, N, h, M]
    
    # Clamp coordinates to valid range
    x_coords = x_coords.clamp(0, W-1)
    y_coords = y_coords.clamp(0, H-1)
    
    indices = y_coords * W + x_coords  # [B, N, h, M]
    
    # [B, N, h, C] -> [B, N, h, M, C]
    k_expanded = k.unsqueeze(3).expand(-1, -1, -1, M, -1)
    
    # [B, N, h, M] -> [B, N, h, M, C]
    indices_gather = indices.unsqueeze(-1).expand(-1, -1, -1, -1, C)
    
    # [B, N, h, M, C]
    k_sampled = torch.gather(k_expanded, dim=1, index=indices_gather)
    
    # [B, N, h, M, C] -> [B, N, h, M]
    # negative L1 norm
    output = -torch.abs(q.unsqueeze(3) - k_sampled).sum(dim=-1)
    
    return output, indices_gather

def attn_scatter(attn, win_r):
    """
    Scatter the attn to four sub-windows
    
    Args:
        attn: [B, N, h, M], M = (2*win_r[0]+2) * (2*win_r[1]+2)
        win_r: window radius
        
    Returns:
        attn_sub: [B, N, h, 4, M_sub] attn for four sub-windows
    """
    B, N, h, M = attn.shape
    M_sub = (2*win_r[0] + 1)*(2*win_r[1] + 1)
    
    # [B, N, h, H_win, W_win]
    attn_2d = attn.view(B, N, h, 2*win_r[0] + 2, 2*win_r[1] + 2)
    
    # nw [0, 0] offset
    win_nw = attn_2d[..., :2*win_r[0]+1, :2*win_r[1]+1]
    # ne [1, 0] offset
    win_ne = attn_2d[..., :2*win_r[0]+1, 1:2*win_r[1]+2]
    # sw [0, 1] offset
    win_sw = attn_2d[..., 1:2*win_r[0]+2, :2*win_r[1]+1]
    # se [1, 1] offset
    win_se = attn_2d[..., 1:2*win_r[0]+2, 1:2*win_r[1]+2]
    
    win_nw = win_nw.reshape(B, N, h, M_sub)
    win_ne = win_ne.reshape(B, N, h, M_sub)
    win_sw = win_sw.reshape(B, N, h, M_sub)
    win_se = win_se.reshape(B, N, h, M_sub)
    
    attn_sub = torch.stack([win_nw, win_ne, win_sw, win_se], dim=3)
    
    return attn_sub

def attn_gather(attn_sub, win_r):
    """
    Gather the four attn_sub to attn
    
    Args:
        attn_sub: [B, N, h, 4, M_sub] 
        win_r: window radius
        
    Returns:
        merged_attn: [B, N, h, M]
    """
    B, N, h, _, M_sub = attn_sub.shape
    
    merged = torch.zeros(B, N, h, 2*win_r[0] + 2, 2*win_r[1] + 2, device=attn_sub.device, dtype=attn_sub.dtype)
    
    # nw [0, 0] offset
    win_nw = attn_sub[:, :, :, 0, :].view(B, N, h, 2*win_r[0]+1, 2*win_r[1]+1)
    merged[..., :2*win_r[0]+1, :2*win_r[1]+1] += win_nw
    
    # ne [1, 0] offset
    win_ne = attn_sub[:, :, :, 1, :].view(B, N, h, 2*win_r[0]+1, 2*win_r[1]+1)
    merged[..., :2*win_r[0]+1, 1:2*win_r[1]+2] += win_ne
    
    # sw [0, 1] offset
    win_sw = attn_sub[:, :, :, 2, :].view(B, N, h, 2*win_r[0]+1, 2*win_r[1]+1)
    merged[..., 1:2*win_r[0]+2, :2*win_r[1]+1] += win_sw
    
    # se [1, 1] offset
    win_se = attn_sub[:, :, :, 3, :].view(B, N, h, 2*win_r[0]+1, 2*win_r[1]+1)
    merged[..., 1:2*win_r[0]+2, 1:2*win_r[1]+2] += win_se
    
    merged_attn = merged.view(B, N, h, -1)
    
    return merged_attn

def compute_bilinear_softmax(attn, bilinear_weight, win_r):
    """
    Blinear Softmax: Attention sampled on a contiguous position
    
    Args:
        attn: [B, N, h, M] attention on discreate position
        win_r: window radius
        
    Returns:
        output: [B, N, h, M] effective attention on contiguous position
    """
    attn_sub = attn_scatter(attn, win_r) # [B, N, h, 4, M_sub]
    
    attn_weighted = bilinear_weight.unsqueeze(-1)*attn_sub.softmax(dim=-1)
    
    output = attn_gather(attn_weighted, win_r) # [B, N, h, M]
    
    return output

def attention_aggregate(v, attn, indices_gather, win_r):

    B, N, h, C = v.shape
    M = (2*win_r[0] + 2)*(2*win_r[1] + 2)

    # [B, N, h, C] -> [B, N, h, M, C]
    v_expanded = v.unsqueeze(3).expand(-1, -1, -1, M, -1)
    v_sampled = torch.gather(v_expanded, dim=1, index=indices_gather)

    output = (attn.unsqueeze(-1)*v_sampled).sum(dim=3)

    return output.view(B, N, -1)
