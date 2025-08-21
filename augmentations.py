import torch
import random
import numpy as np

class SalienceAugment:
    """
    A class to handle augmentations for HCQT salience maps.
    This class applies SpecAugment (frequency and time masking)
    and additive Gaussian noise with different masking strategies.
    """
    def __init__(self,
                 freq_mask_param=27,
                 time_mask_param=24,
                 num_freq_masks=2,
                 num_time_masks=2,
                 add_gaussian_noise=True,
                 gaussian_noise_std=0.005,
                 masking_strategy='all_channels'): # New parameter
        """
        Initializes the augmentation parameters.

        Args:
            freq_mask_param (int): Max frequency bins to mask.
            time_mask_param (int): Max time frames to mask.
            num_freq_masks (int): Number of frequency masks.
            num_time_masks (int): Number of time masks.
            add_gaussian_noise (bool): Whether to add Gaussian noise.
            gaussian_noise_std (float): Standard deviation of the noise.
            masking_strategy (str): How to apply masks.
                                    'all_channels': Apply the same mask to all channels (old way).
                                    'single_channel': Pick one random channel and mask only it.
        """
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.add_gaussian_noise = add_gaussian_noise
        self.gaussian_noise_std = gaussian_noise_std
        self.masking_strategy = masking_strategy

    def __call__(self, hcqt):
        """
        Applies augmentations to the given HCQT tensor.

        Args:
            hcqt (torch.Tensor): The input HCQT tensor of shape (channels, freqs, time).

        Returns:
            torch.Tensor: The augmented HCQT tensor.
        """
        augmented_hcqt = hcqt.clone()

        if self.add_gaussian_noise and self.gaussian_noise_std > 0:
            noise = torch.randn_like(augmented_hcqt) * self.gaussian_noise_std
            augmented_hcqt += noise
        
        if self.masking_strategy == 'single_channel':
            # Choose one random channel (e.g., fundamental, or 3rd harmonic) to mask
            num_channels = augmented_hcqt.shape[0]
            channel_to_mask = random.randint(0, num_channels - 1)
            
            # Apply frequency masks to the chosen channel
            for _ in range(self.num_freq_masks):
                f = random.randint(0, self.freq_mask_param)
                f0 = random.randint(0, augmented_hcqt.shape[1] - f)
                augmented_hcqt[channel_to_mask, f0:f0 + f, :] = 0

            # Apply time masks to the chosen channel
            for _ in range(self.num_time_masks):
                t = random.randint(0, self.time_mask_param)
                t0 = random.randint(0, augmented_hcqt.shape[2] - t)
                augmented_hcqt[channel_to_mask, :, t0:t0 + t] = 0

        elif self.masking_strategy == 'all_channels':
            # This is the old, more aggressive behavior
            # Apply frequency masks to all channels
            for _ in range(self.num_freq_masks):
                f = random.randint(0, self.freq_mask_param)
                f0 = random.randint(0, augmented_hcqt.shape[1] - f)
                augmented_hcqt[:, f0:f0 + f, :] = 0

            # Apply time masks to all channels
            for _ in range(self.num_time_masks):
                t = random.randint(0, self.time_mask_param)
                t0 = random.randint(0, augmented_hcqt.shape[2] - t)
                augmented_hcqt[:, :, t0:t0 + t] = 0
        
            
        return torch.clamp(augmented_hcqt, 0.0, 1.0)