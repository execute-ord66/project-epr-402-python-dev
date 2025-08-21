import numpy as np
from numba import njit, prange

@njit(cache=True, fastmath=True)
def rasterize_f0_lines(bin_pos, active_mask, SS_F, SS_T, n_bins, n_frames, f0_map_hi):
    hi_n_bins = n_bins * SS_F
    for t0 in range(n_frames - 1):
        a0 = 1.0 if active_mask[t0] else 0.0
        a1 = 1.0 if active_mask[t0 + 1] else 0.0
        if (a0 + a1) == 0.0:
            continue
        b0 = bin_pos[t0]
        b1 = bin_pos[t0 + 1]
        hi_t_start = t0 * SS_T
        for hi_t in range(hi_t_start, hi_t_start + SS_T):
            u = (hi_t - hi_t_start) / float(SS_T)
            act = (1.0 - u) * a0 + u * a1
            if act <= 0.5:
                continue
            b = b0 + u * (b1 - b0)
            hi_b = int(round(b * SS_F))
            if 0 <= hi_b < hi_n_bins:
                f0_map_hi[hi_b, hi_t] = 1.0

@njit(cache=True, fastmath=True, parallel=True)
def normalize_peaks_inplace(f0_map):
    F, T = f0_map.shape
    for t in prange(T):
        for p in range(1, F-1):
            val = f0_map[p, t]
            if p < 30:
                continue

            if val > 0.0 and val > f0_map[p-1, t] and val > f0_map[p+1, t]:
                inv = 1.0 / val
                v0 = f0_map[p-1, t] * inv; f0_map[p-1, t] = 1.0 if v0 > 1.0 else v0
                v1 = f0_map[p,   t] * inv; f0_map[p,   t] = 1.0 if v1 > 1.0 else v1
                v2 = f0_map[p+1, t] * inv; f0_map[p+1, t] = 1.0 if v2 > 1.0 else v2

import torch.nn.functional as F
import torch.nn as nn

class GEGLU(nn.Module):
    """
    Implements the Gated Linear Unit with a GELU activation for a CNN.
    
    The input tensor is split into two equal halves along the CHANNEL dimension (dim=1).
    One half is passed through a GELU activation, and they are then multiplied
    element-wise.
    
    Input shape: (N, 2*C, H, W)
    Output shape: (N, C, H, W)
    """
    def forward(self, x):
        # --- THE FIX: Split along the channel dimension (dim=1) ---
        gate, value = x.chunk(2, dim=1)
        
        return F.gelu(gate) * value
    
class SwiGLU(nn.Module):
    """
    Implements the Gated Linear Unit with a GELU activation for a CNN.
    
    The input tensor is split into two equal halves along the CHANNEL dimension (dim=1).
    One half is passed through a GELU activation, and they are then multiplied
    element-wise.
    
    Input shape: (N, 2*C, H, W)
    Output shape: (N, C, H, W)
    """
    def forward(self, x):
        # --- THE FIX: Split along the channel dimension (dim=1) ---
        gate, value = x.chunk(2, dim=1)
        
        return F.silu(gate, inplace=True) * value