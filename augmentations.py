import torch

class SalienceAugment:
    """
    A GPU-accelerated class to handle SpecAugment for batched HCQTs.
    This class applies frequency and time masking directly on the GPU.
    """
    def __init__(self,
                 freq_mask_param=27,
                 time_mask_param=24,
                 num_freq_masks=2,
                 num_time_masks=2,
                 masking_strategy='all_channels'):
        """
        Initializes the augmentation parameters.

        Args:
            freq_mask_param (int): Max frequency bins to mask (F).
            time_mask_param (int): Max time frames to mask (T).
            num_freq_masks (int): Number of frequency masks to apply.
            num_time_masks (int): Number of time masks to apply.
            masking_strategy (str): 'all_channels' or 'single_channel'.
        """
        self.F = freq_mask_param
        self.T = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.masking_strategy = masking_strategy

    def __call__(self, cqt_batch):
        """
        Applies augmentations to a batch of HCQT tensors on the GPU.

        Args:
            cqt_batch (torch.Tensor): Input tensor of shape (B, C, F_bins, T_frames).

        Returns:
            torch.Tensor: The augmented HCQT batch.
        """
        # Ensure we don't modify the original tensor
        augmented_batch = cqt_batch.clone()
        
        # Get tensor dimensions and the device it's on
        batch_size, num_channels, num_freq_bins, num_time_frames = augmented_batch.shape
        device = augmented_batch.device

        if self.masking_strategy == 'single_channel':
            # Not yet implemented for GPU batch processing, falls back to safer 'all_channels'
            # (Implementing this efficiently requires more complex indexing)
            pass

        # --- Frequency Masking ---
        for _ in range(self.num_freq_masks):
            # Generate random mask widths for each item in the batch
            # Shape: (B,)
            f = torch.randint(0, self.F, (batch_size,), device=device)
            
            # Generate random start points for each item in the batch
            # Shape: (B,)
            f0 = torch.randint(0, num_freq_bins, (batch_size,), device=device)
            
            # Create a mask for the entire batch
            #arange_freq shape: (F_bins,) -> (1, F_bins, 1) for broadcasting
            arange_freq = torch.arange(num_freq_bins, device=device).view(1, num_freq_bins, 1)
            
            # f0.view(-1, 1, 1) -> (B, 1, 1)
            # f.view(-1, 1, 1) -> (B, 1, 1)
            # freq_mask becomes a boolean tensor of shape (B, F_bins, 1)
            freq_mask = (arange_freq >= f0.view(-1, 1, 1)) & \
                        (arange_freq < f0.view(-1, 1, 1) + f.view(-1, 1, 1))
            
            # Expand to match the batch dimensions (B, F_bins, 1) -> (B, 1, F_bins, T_frames)
            # and apply the mask to all channels.
            augmented_batch.masked_fill_(freq_mask.unsqueeze(1).expand_as(augmented_batch), 0)

        # --- Time Masking ---
        for _ in range(self.num_time_masks):
            # Generate random mask widths for each item in the batch
            t = torch.randint(0, self.T, (batch_size,), device=device)
            
            # Generate random start points for each item in the batch
            t0 = torch.randint(0, num_time_frames, (batch_size,), device=device)
            
            # Create a mask for the entire batch
            arange_time = torch.arange(num_time_frames, device=device).view(1, 1, num_time_frames)
            
            # time_mask becomes a boolean tensor of shape (B, 1, T_frames)
            time_mask = (arange_time >= t0.view(-1, 1, 1)) & \
                        (arange_time < t0.view(-1, 1, 1) + t.view(-1, 1, 1))
            
            # Expand and apply the mask
            augmented_batch.masked_fill_(time_mask.unsqueeze(1).expand_as(augmented_batch), 0)
            
        return augmented_batch