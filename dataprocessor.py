# dataprocessor.py
import torch
import numpy as np
import librosa
import pandas as pd
import random
import os
import hashlib
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

import torch.nn.functional as F
from torch.utils.data import IterableDataset, Dataset
from nnAudio.features import CQT1992v2
from numba import njit, prange


@njit(cache=True, fastmath=True)
def precompute_rms_env(y: np.ndarray, frame_length=2048, hop_length=512):
    """Vectorized RMS envelope calculation using Numba."""
    num_frames = 1 + (len(y) - frame_length) // hop_length
    rms_values = np.zeros(num_frames, dtype=np.float32)
    for i in prange(num_frames):
        start = i * hop_length
        end = start + frame_length
        frame = y[start:end]
        rms_values[i] = np.sqrt(np.mean(frame**2))
    return rms_values


@njit(cache=True, fastmath=True)
def _njit_rms(y, frame_length=2048, hop_length=512):
    """Numba-jitted core RMS calculation."""
    # Ensure y is a 1D array for this logic
    if y.ndim != 1:
        # Fallback or error for non-1D input, though Librosa's input is 1D
        return np.array([0.0], dtype=np.float32)

    num_frames = 1 + (len(y) - frame_length) // hop_length
    rms_values = np.zeros(num_frames, dtype=np.float32)

    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        frame = y[start:end]
        rms_values[i] = np.sqrt(np.mean(frame**2))

    return rms_values


def get_rms_db(y, sr, frame_length=2048, hop_length=512):
    """
    Calculates the root-mean-square (RMS) energy of an audio signal in decibels.
    Uses a Numba-jitted core for faster computation.

    Args:
        y (np.ndarray): The input audio signal (as a NumPy array).
        sr (int): The sample rate of the audio.
        frame_length (int): The length of each frame for RMS analysis.
        hop_length (int): The number of samples between successive frames.

    Returns:
        float: The mean RMS energy in dB. Returns -100.0 for silence.
    """
    # Use the fast Numba core to calculate RMS values
    rms_values = _njit_rms(y, frame_length=frame_length, hop_length=hop_length)

    if rms_values.size == 0 or np.max(rms_values) == 0:
        return -100.0

    # Convert RMS amplitude to decibels (this part is fast and uses Librosa's utility)
    rms_db = librosa.amplitude_to_db(rms_values, ref=1.0)

    return np.mean(rms_db)


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
        for p in range(1, F - 1):
            val = f0_map[p, t]
            if p < 30:
                continue

            if val > 0.0 and val > f0_map[p - 1, t] and val > f0_map[p + 1, t]:
                inv = 1.0 / val
                v0 = f0_map[p - 1, t] * inv
                f0_map[p - 1, t] = 1.0 if v0 > 1.0 else v0
                v1 = f0_map[p, t] * inv
                f0_map[p, t] = 1.0 if v1 > 1.0 else v1
                v2 = f0_map[p + 1, t] * inv
                f0_map[p + 1, t] = 1.0 if v2 > 1.0 else v2


@njit(cache=True, fastmath=True)
def hz_to_cqt_bin(freq_hz, fmin, bins_per_octave):
    return np.where(freq_hz > 0, bins_per_octave * np.log2(freq_hz / fmin), -1.0)


# =============================================================================
#  DataProcessor (for caching validation/evaluation data)
# =============================================================================
class DataProcessor:
    """
    Processes a group of tracks into a cached CQT/F0 representation for the entire mix.
    Used ONLY for the validation set to ensure reproducibility.
    """

    def __init__(self, config, root_dir, cache_dir, log_callback):
        self.config, self.root_dir, self.cache_dir = config, root_dir, cache_dir
        self.dp = self.config["data_params"]
        self.n_bins = self.dp["n_octaves"] * self.dp["bins_per_octave"]
        self.log = log_callback

    def get_dataset_folder(self, stem_name):
        is_cantoria = stem_name.lower().startswith("cantoria")
        is_dcs = stem_name.lower().startswith("dcs")
        try:
            for folder_name in os.listdir(self.root_dir):
                if not os.path.isdir(os.path.join(self.root_dir, folder_name)):
                    continue
                lower_folder_name = folder_name.lower()
                if is_cantoria and "cantoria" in lower_folder_name:
                    return folder_name
                if is_dcs and (
                    "dcs" in lower_folder_name or "dagstuhl" in lower_folder_name
                ):
                    return folder_name
        except FileNotFoundError:
            pass
        self.log(f"Warning: Could not determine dataset folder for stem '{stem_name}'.")
        return ""

    def process_and_cache_group(self, track_stems):
        """
        Processes and caches an entire group of stems as a single mix. This function
        handles different path structures for different datasets.
        """
        # Create a single, stable, unique string for the group by sorting and joining.
        canonical_name = "_".join(sorted(track_stems))
        # Create a SHA-1 hash of this string to get a safe, fixed-length filename.
        group_hash = hashlib.sha1(canonical_name.encode("utf-8")).hexdigest()

        cache_fname = f"{group_hash}.npz"
        cache_path = os.path.join(self.cache_dir, cache_fname)

        if os.path.exists(cache_path):
            return cache_path

        if self.log:
            self.log(f"Processing & Caching Group: {', '.join(track_stems)}")
            self.log(f"  > Cache file: {cache_path}")  # Log the hash for debugging

        # --- AUDIO MIXING ---
        max_len, all_y = 0, []
        valid_stems_for_f0 = []  # Keep track of stems that were successfully loaded

        for stem in track_stems:
            # Heuristic: ChoralSynth stems contain path separators. Cantoria/DCS stems do not.
            is_choralsynth_style = os.sep in stem

            if is_choralsynth_style:
                # Path for ChoralSynth: ./datasets/ChoralSynth/<stem>.wav
                # The `stem` already contains '<trackname>/voices/<stem_base_name>'
                audio_path = os.path.join(self.root_dir, "ChoralSynth", f"{stem}.wav")
            else:
                # Path for Cantoria/DCS: ./datasets/<DatasetName>/Audio/<stem>.wav
                dataset_folder = self.get_dataset_folder(stem)
                if not dataset_folder:
                    if self.log:
                        self.log(
                            f"ERROR: Could not find dataset folder for stem {stem}"
                        )
                    continue
                audio_path = os.path.join(
                    self.root_dir, dataset_folder, "Audio", f"{stem}.wav"
                )

            if not os.path.exists(audio_path):
                self.log(f"ERROR: File not found at resolved path: {audio_path}")
                continue

            y, _ = librosa.load(audio_path, sr=self.dp["sr"])
            all_y.append(y)
            valid_stems_for_f0.append(stem)  # Add to list for F0 processing
            if len(y) > max_len:
                max_len = len(y)

        if not all_y:
            if self.log:
                self.log(
                    f"Warning: No valid audio files found for group. Skipping cache generation."
                )
            return None

        y_mix = np.sum([np.pad(y, (0, max_len - len(y))) for y in all_y], axis=0)
        if np.max(np.abs(y_mix)) > 0:
            y_mix /= np.max(np.abs(y_mix))

        # --- HCQT CALCULATION for the full mix ---
        cqt_list = [
            librosa.cqt(
                y_mix,
                sr=self.dp["sr"],
                hop_length=self.dp["hop_length"],
                fmin=self.dp["fmin"] * h,
                n_bins=self.n_bins,
                bins_per_octave=self.dp["bins_per_octave"],
            )
            for h in self.dp["harmonics"]
        ]
        min_time = min(c.shape[1] for c in cqt_list)
        hcqt = np.stack([c[:, :min_time] for c in cqt_list])
        log_hcqt = (1.0 / 80.0) * librosa.amplitude_to_db(
            np.abs(hcqt), ref=np.max
        ) + 1.0

        # --- GROUND TRUTH F0 MAP with supersampling & average pooling ---
        n_frames = log_hcqt.shape[2]
        frame_times = librosa.frames_to_time(
            np.arange(n_frames), sr=self.dp["sr"], hop_length=self.dp["hop_length"]
        )

        # Supersample factors (freq & time)
        SS_F = 4  # frequency upsampling
        SS_T = 4  # time upsampling

        hi_n_bins = self.n_bins * SS_F
        hi_n_frames = n_frames * SS_T

        # High-res canvas that accumulates all stems
        f0_map_hi = np.zeros((hi_n_bins, hi_n_frames), dtype=np.float32)

        total_active_frames = 0

        for stem in valid_stems_for_f0:
            is_choralsynth_style = os.sep in stem

            interp_freqs = None
            active_mask = None

            if is_choralsynth_style:
                crepe_path = os.path.join(
                    self.root_dir, "ChoralSynth", f"{stem}.f0.csv"
                )
                if not os.path.exists(crepe_path):
                    continue

                crepe_df = pd.read_csv(crepe_path)
                crepe_df.columns = [c.strip() for c in crepe_df.columns]

                interp_freqs = np.interp(
                    frame_times,
                    crepe_df["time"],
                    crepe_df["frequency"],
                    left=0,
                    right=0,
                )
                interp_confidence = np.interp(
                    frame_times,
                    crepe_df["time"],
                    crepe_df["confidence"],
                    left=0,
                    right=0,
                )
                active_mask = (interp_freqs >= 1.5 * self.dp["fmin"]) & (
                    interp_confidence > 0.55
                )
            else:
                dataset_folder = self.get_dataset_folder(stem)
                if not dataset_folder:
                    continue
                base_path = os.path.join(self.root_dir, dataset_folder)
                crepe_path = os.path.join(base_path, "F0_crepe", f"{stem}.csv")
                pyin_path = os.path.join(base_path, "F0_pyin", f"{stem}.csv")
                if not os.path.exists(crepe_path) or not os.path.exists(pyin_path):
                    continue

                crepe_df = pd.read_csv(crepe_path)
                crepe_df.columns = [c.strip() for c in crepe_df.columns]
                pyin_df = pd.read_csv(
                    pyin_path, header=None, names=["time", "frequency", "confidence"]
                )

                interp_freqs = np.interp(
                    frame_times,
                    crepe_df["time"],
                    crepe_df["frequency"],
                    left=0,
                    right=0,
                )
                pyin_voiced_mask = (pyin_df["frequency"] > 0).astype(float)
                interp_voiced = np.interp(
                    frame_times, pyin_df["time"], pyin_voiced_mask, left=0, right=0
                )
                interp_confidence = np.interp(
                    frame_times,
                    crepe_df["time"],
                    crepe_df["confidence"],
                    left=0,
                    right=0,
                )
                active_mask = (
                    (interp_freqs >= 1.5 * self.dp["fmin"])
                    & (interp_confidence > 0.55)
                    & (interp_voiced > 0.3)
                )

            if interp_freqs is None or active_mask is None:
                continue

            total_active_frames += int(np.sum(active_mask))

            # Convert to continuous CQT bin positions (base resolution)
            bin_pos = hz_to_cqt_bin(
                interp_freqs, self.dp["fmin"], self.dp["bins_per_octave"]
            )
            # Clamp to valid range to avoid NaNs or infs later
            bin_pos = np.clip(bin_pos, 0.0, self.n_bins - 1e-6)

            # For each adjacent frame pair, draw a (possibly short) line segment on the high-res canvas
            # Make sure inputs are light-weight dtypes
            bin_pos32 = bin_pos.astype(np.float32, copy=False)
            active_mask_bool = active_mask.astype(np.bool_, copy=False)

            rasterize_f0_lines(
                bin_pos32,
                active_mask_bool,
                SS_F,
                SS_T,
                self.n_bins,
                n_frames,
                f0_map_hi,  # preallocated once outside the stem loop
            )

        if total_active_frames == 0:
            if self.log:
                self.log(
                    f"  - WARNING: No active frames found for this group. Ground truth will be all zeros."
                )
        else:
            if self.log:
                self.log(
                    f"  - Found {total_active_frames} active pitch frames for this group."
                )

        # --- GAUSSIAN SMOOTHING, DOWNSAMPLING & CLIPPING ---
        f0_map_hi = gaussian_filter(
            f0_map_hi, sigma=(self.dp["gaussian_sigma"] * 2, 0.5), mode="constant"
        )
        row_max = f0_map_hi.max(axis=1, keepdims=True)
        row_max[row_max == 0] = 1.0
        f0_map_hi = f0_map_hi / row_max
        # --- Average-pool (SS_F x SS_T) back down to (n_bins, n_frames)
        if hi_n_bins > 0 and hi_n_frames > 0:
            # reshape to (n_bins, SS_F, n_frames, SS_T) then mean over the SS axes
            try:
                f0_map = f0_map_hi.reshape(self.n_bins, SS_F, n_frames, SS_T).mean(
                    axis=(1, 3)
                )
            except ValueError:
                # Fallback in case shapes are off due to edge rounding (shouldn't happen)
                f0_map = np.zeros((self.n_bins, n_frames), dtype=np.float32)
                if self.log:
                    self.log(
                        "  - WARNING: Supersample reshape failed; produced empty f0_map."
                    )
        else:
            f0_map = np.zeros((self.n_bins, n_frames), dtype=np.float32)

        normalize_peaks_inplace(f0_map)

        # --- Save ---
        np.savez_compressed(cache_path, log_hcqt=log_hcqt, f0_map=f0_map)
        return cache_path


# =============================================================================
#  Torch-Only GPU Helpers
# =============================================================================
def _gaussian_kernel1d(device, sigma: float):
    radius = max(1, int(3.0 * float(sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / float(sigma + 1e-8)) ** 2)
    k = k / (k.sum() + 1e-12)
    return k


def _gaussian_blur_2d(x_b1ft: torch.Tensor, sigma_f: float, sigma_t: float):
    B, C, Freq, Time = x_b1ft.shape
    kf = _gaussian_kernel1d(x_b1ft.device, sigma_f).view(1, 1, -1, 1)
    kt = _gaussian_kernel1d(x_b1ft.device, sigma_t).view(1, 1, 1, -1)
    # Use reflection padding to handle borders more gracefully
    pad_f = kf.shape[2] // 2
    pad_t = kt.shape[3] // 2
    x = F.pad(x_b1ft, (pad_t, pad_t, pad_f, pad_f), mode="reflect")
    x = F.conv2d(x, kf, groups=C)
    x = F.conv2d(x, kt, groups=C)
    return x


def _torch_normalize_peaks(f0_map_b1ft: torch.Tensor):
    """A robust, vectorized, PyTorch-based peak normalization function."""
    if f0_map_b1ft.shape[2] < 3:
        return f0_map_b1ft

    padded_map = F.pad(f0_map_b1ft, (0, 0, 1, 1), mode="constant", value=-1e-6)
    center = padded_map[:, :, 1:-1, :]
    below = padded_map[:, :, :-2, :]
    above = padded_map[:, :, 2:, :]

    peak_mask = (center > below) & (center > above) & (center > 0)

    if f0_map_b1ft.shape[2] > 30:
        freq_rule_mask = (
            torch.arange(f0_map_b1ft.shape[2], device=f0_map_b1ft.device).view(
                1, 1, -1, 1
            )
            >= 30
        )
        peak_mask &= freq_rule_mask

    if not torch.any(peak_mask):
        return f0_map_b1ft

    peak_values_map = torch.where(
        peak_mask, center, torch.tensor(0.0, device=center.device)
    )
    padded_peak_values = F.pad(
        peak_values_map, (0, 0, 1, 1), mode="constant", value=0.0
    )

    divisor_map = (
        padded_peak_values[:, :, 1:-1, :]
        + padded_peak_values[:, :, :-2, :]
        + padded_peak_values[:, :, 2:, :]
    )
    final_divisor = torch.where(
        divisor_map > 1e-8, divisor_map, torch.tensor(1.0, device=divisor_map.device)
    )

    normalized_map = f0_map_b1ft / final_divisor
    return torch.clamp(normalized_map, 0.0, 1.0)


# =============================================================================
#  OnTheFlyProcessor (Single-threaded, uses GPU)
# =============================================================================
class OnTheFlyProcessor:
    def __init__(self, config, device):
        self.config = config
        self.dp = config["data_params"]
        self.tp = config["training_params"]
        self.device = device
        self.n_bins = self.dp["n_octaves"] * self.dp["bins_per_octave"]
        print("Number of bins", self.n_bins)

        if not self.dp["harmonics"] or self.dp["harmonics"][0] != 1:
            raise ValueError(
                "Configuration error: `data_params.harmonics` must be a list that starts with 1."
            )

        if "n_octaves" not in self.dp or self.dp["n_octaves"] <= 1:
            raise ValueError(
                f"Configuration Error in [data_params]: 'n_octaves' is missing or not greater than 1."
                f" Found value: {self.dp.get('n_octaves')}"
            )
        if "bins_per_octave" not in self.dp or self.dp["bins_per_octave"] <= 0:
            raise ValueError(
                "Configuration Error in [data_params]: 'bins_per_octave' is missing or invalid."
            )

        self.cqt_transformers = []
        for h in self.dp["harmonics"]:
            transformer = CQT1992v2(
                sr=self.dp["sr"],
                hop_length=self.dp["hop_length"],
                fmin=self.dp["fmin"] * h,
                n_bins=self.n_bins,
                bins_per_octave=self.dp["bins_per_octave"],
            )
            transformer.to(self.device)
            self.cqt_transformers.append(transformer)

    def batch_generate_features_on_gpu(self, audios_bt: torch.Tensor, f0_lists_b):
        # audios_bt: (B,T) on CPU (pinned) -> move to device once
        B, T = audios_bt.shape
        x = audios_bt.to(self.device, non_blocking=True)

        # CQT once per harmonic on the B batch
        hcqt_list = [t(x) for t in self.cqt_transformers]  # each (B,F,T)
        hcqt = torch.stack(hcqt_list, dim=1)  # (B,H,F,T)

        ref = torch.clamp(hcqt.max(dim=(1, 2, 3), keepdim=True).values, min=1.0)
        hcqt_db = 20.0 * torch.log10(torch.clamp(hcqt, 1e-10) / ref)
        log_hcqt = torch.clamp(hcqt_db / 80.0 + 1.0, 0.0, 1.0)

        # salience per item (small loop on GPU is fine)
        sal_list = []
        for b in range(B):
            _, sal = self.generate_features_on_gpu(
                x[b].detach().cpu().numpy(), f0_lists_b[b]
            )
            sal_list.append(sal)
        sal = torch.stack(sal_list, dim=0)  # (B,F,T)

        return log_hcqt, sal

    def generate_features_on_gpu(self, mixed_audio_np, f0_data_list):
        audio = torch.from_numpy(mixed_audio_np).to(self.device)  # (T,)
        x = audio[None, :]  # (1,T)
        # HCQT via nnAudio (already Torch)
        hcqt_list = [t(x).squeeze(0) for t in self.cqt_transformers]  # each (F,T)
        hcqt_tensor = torch.stack(hcqt_list, dim=0)  # (H,F,T)

        # scale to dB [0,1]
        ref = hcqt_tensor.max()
        ref = torch.clamp(ref, min=1.0)
        hcqt_db = 20.0 * torch.log10(torch.clamp(hcqt_tensor, 1e-10) / ref)
        log_hcqt = torch.clamp(hcqt_db / 80.0 + 1.0, 0.0, 1.0)

        # ---- Torch salience (anti-aliased splat) ----
        n_frames = hcqt_tensor.shape[-1]
        SS_F, SS_T = 4, 4
        hi_n_bins = self.n_bins * SS_F
        hi_n_frames = n_frames * SS_T
        hop_s = self.dp["hop_length"] / self.dp["sr"]
        hi_times = torch.arange(hi_n_frames, device=self.device) * (hop_s / SS_T)

        f0_map_hi = torch.zeros(
            (1, 1, hi_n_bins, hi_n_frames), device=self.device
        )  # (B=1,C=1,F,T)

        for t_np, f_np in f0_data_list:
            if len(t_np) < 2:
                continue
            t = torch.from_numpy(t_np).to(self.device).float()
            f = torch.from_numpy(f_np).to(self.device).float()

            # interp to hi_times
            # torch doesn't have np.interp; do manual linear interp
            # assume t is sorted and within [0, segment_dur]
            idx = torch.searchsorted(t, hi_times.clamp(min=t[0], max=t[-1])) - 1
            idx = idx.clamp(min=0, max=t.numel() - 2)
            t0 = t[idx]
            t1 = t[idx + 1]
            f0 = f[idx]
            f1 = f[idx + 1]
            w = (hi_times - t0) / torch.clamp(t1 - t0, min=1e-6)
            f_interp = (1 - w) * f0 + w * f1

            # mask & continuous bin
            active = f_interp >= (1.5 * self.dp["fmin"])  # align with offline
            if active.any():
                fa = f_interp[active]
                ta = torch.nonzero(active, as_tuple=False).squeeze(1)

                cont_bin = (
                    self.dp["bins_per_octave"]
                    * torch.log2(
                        torch.clamp(
                            fa / (self.dp["fmin"] * self.dp["harmonics"][0]), min=1e-12
                        )
                    )
                    * SS_F
                )

                b0 = torch.floor(cont_bin).long()
                b1 = b0 + 1
                w1 = cont_bin - b0.float()
                w0 = 1.0 - w1

                # bounds
                valid0 = (b0 >= 0) & (b0 < hi_n_bins)
                valid1 = (b1 >= 0) & (b1 < hi_n_bins)

                # index into (B=1,C=1,F_hi,T_hi)
                if valid0.any():
                    f0_map_hi[0, 0, b0[valid0], ta[valid0]] = torch.maximum(
                        f0_map_hi[0, 0, b0[valid0], ta[valid0]], w0[valid0]
                    )
                if valid1.any():
                    f0_map_hi[0, 0, b1[valid1], ta[valid1]] = torch.maximum(
                        f0_map_hi[0, 0, b1[valid1], ta[valid1]], w1[valid1]
                    )

        if torch.any(f0_map_hi):
            # blur (match offline sigma in hi-res freq; mild in time)
            freq_sigma = float(self.dp["gaussian_sigma"] * 2.0)  # to match offline
            time_sigma = 0.5
            f0_map_hi = _gaussian_blur_2d(f0_map_hi, freq_sigma, time_sigma)

            # row-wise max norm
            row_max = f0_map_hi.amax(dim=3, keepdim=True)  # max over time per freq row
            row_max = torch.clamp(row_max, min=1.0)
            f0_map_hi = f0_map_hi / row_max

            # pool to (n_bins, n_frames): mean over (SS_F x SS_T)
            f0_map = f0_map_hi.view(1, 1, self.n_bins, SS_F, n_frames, SS_T).mean(
                dim=(3, 5)
            )
        else:
            f0_map = torch.zeros((1, 1, self.n_bins, n_frames), device=self.device)

        # clip + local-peak normalization (torch)
        f0_map = torch.clamp(f0_map, 0.0, 1.0)
        # quick local-peak normalize:
        center = f0_map[:, :, 1:-1, :]
        below = f0_map[:, :, 0:-2, :]
        above = f0_map[:, :, 2:, :]

        peak = (center > 0) & (center > below) & (center > above)
        if self.n_bins > 30:
            mask_ge30 = (
                torch.arange(self.n_bins, device=self.device).view(1, 1, -1, 1) >= 30
            )
            peak = peak & mask_ge30[:, :, 1:-1, :]

        inv = torch.zeros_like(center)
        inv[peak] = 1.0 / torch.clamp(center[peak], min=1e-6)
        below[:, :, 1:-1, :] = torch.minimum(
            torch.ones_like(below[:, :, 1:-1, :]), below[:, :, 1:-1, :] * inv
        )
        center[:] = torch.minimum(torch.ones_like(center), center * inv)
        above[:, :, 0:-2, :] = torch.minimum(
            torch.ones_like(above[:, :, 0:-2, :]), above[:, :, 0:-2, :] * inv
        )

        sal = f0_map.squeeze(0)  # (1,F,T) -> (F,T)
        return log_hcqt, sal
