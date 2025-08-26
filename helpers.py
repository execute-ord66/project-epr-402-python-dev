# helpers.py
import torch.nn.functional as F
import torch.nn as nn
import torch.distributions as dist
import torch

def mixup_data(x, y, alpha=0.4):
    """
    Applies mixup augmentation to a batch of data directly on the GPU.

    Args:
        x (torch.Tensor): The input batch (e.g., CQTs) on the target device.
        y (torch.Tensor): The target batch (e.g., salience maps) on the target device.
        alpha (float): The alpha parameter for the Beta distribution. If 0, no mixup.

    Returns:
        (torch.Tensor, torch.Tensor): The mixed input and target batches.
    """
    if alpha <= 0:
        return x, y

    # Get the device from the input tensor.
    device = x.device
    batch_size = x.size(0)

    # Create the Beta distribution object.
    beta_distribution = dist.Beta(torch.tensor([alpha], device=device), 
                                  torch.tensor([alpha], device=device))
    
    # Sample a single mixing coefficient directly on the GPU.
    lam = beta_distribution.sample()

    # Generate a random permutation of indices.
    index = torch.randperm(batch_size, device=device)

    # Mix the original batch with the shuffled batch.
    # This is now a pure GPU operation.
    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index, :]
    
    return mixed_x, mixed_y

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