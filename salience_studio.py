import os
import sys
import json
import random
import threading
import time
import shutil
from datetime import datetime
from collections import defaultdict
import argparse
from queue import Queue, Empty

import numpy as np
import pandas as pd
import librosa
import librosa.display
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
import mir_eval
import customtkinter as ctk
import hashlib 

# Use the 'Agg' backend for Matplotlib to make it thread-safe for Tkinter
matplotlib.use('Agg')

# -----------------------------------------------------------------------------
# SECTION 1: CORE CONFIGURATION & DEFAULT SETTINGS
# -----------------------------------------------------------------------------

def get_default_config():
    """Returns a dictionary with the default configuration for the entire project."""
    return {
        "run_id": None,
        "data_params": {
            "sr": 22050, "hop_length": 256, "fmin": 32.7, "harmonics": [1, 2, 3, 4, 5],
            "bins_per_octave": 60, "n_octaves": 6, "gaussian_sigma": 1.0,
            "train_dataset": "Cantoria", "eval_dataset": "DCS",
        },
        "model_params": {
            "architecture_name": "SalienceNetV1",
            "input_channels": 5, # Should match len(harmonics)
            "layers": [
                {"type": "conv", "filters": 32, "kernel": 5, "padding": 2},
                {"type": "conv", "filters": 32, "kernel": 5, "padding": 2},
                {"type": "conv", "filters": 32, "kernel": 5, "padding": 2},
                {"type": "conv", "filters": 32, "kernel": (69, 5), "padding": (34, 2)},
                {"type": "conv", "filters": 32, "kernel": (69, 5), "padding": (34, 2)},
                {"type": "conv_out", "filters": 1, "kernel": 1},
            ], "activation": "GELU"
        },
        "training_params": {
            "learning_rate": 1e-3, "batch_size": 32, "num_epochs": 30,
            "optimizer": "AdamW", "patch_width": 50, "patch_overlap": 0.5, "val_split_ratio": 0.15,
        },
        "evaluation_params": {"eval_batch_size": 8, "peak_threshold": 0.3},
        "tuning_params": {
            "population_size": 8,
            "num_generations": 5,
            "epochs_per_eval": 5,
            "mutation_rate": 0.2,
            "crossover_rate": 0.7,
            # Fitness function weights
            "fitness_performance_weight": 0.8, # w1
            "fitness_efficiency_weight": 0.3,  # w2
            "search_space": {
                "layer_filters": {"type": "choice", "choices": [8, 12, 16, 20, 24, 32]},
                "learning_rate": {"type": "log_uniform", "range": [1e-5, 1e-1]},
                "gaussian_sigma": {"type": "uniform", "choices": [1.0, 1.5, 2.0, 2.5, 3.0]},
            }
        }
    }

def get_config_hash(config):
    """Generates a stable SHA-256 hash for a configuration dictionary."""
    # Create a deep copy to avoid modifying the original config
    import copy
    config_copy = copy.deepcopy(config)
    
    # Remove volatile keys that shouldn't affect the hash
    config_copy.pop('run_id', None)
    
    # Convert the dictionary to a sorted JSON string to ensure consistency
    config_string = json.dumps(config_copy, sort_keys=True, indent=None)
    
    # Calculate and return the SHA-256 hash
    return hashlib.sha256(config_string.encode('utf-8')).hexdigest()[:16] # Use first 16 chars for a shorter, manageable folder name

# -----------------------------------------------------------------------------
# SECTION 2: MODELS, LOSS, & METRICS
# -----------------------------------------------------------------------------

class SalienceCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_cfg = config['model_params']
        layers = []
        in_channels = model_cfg['input_channels']
        activation_fn = nn.GELU() if model_cfg.get("activation", "GELU") == "GELU" else nn.ReLU()
        for layer_cfg in model_cfg['layers']:
            out_channels, kernel, padding = layer_cfg['filters'], layer_cfg['kernel'], layer_cfg.get('padding', 'same')
            layers.append(nn.Conv2d(in_channels, out_channels, kernel, padding=padding))
            if layer_cfg['type'] != "conv_out":
                layers.extend([nn.BatchNorm2d(out_channels), activation_fn])
            in_channels = out_channels
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)
    def forward(self, x): return self.network(x)

def bkld_loss(y_pred, y_true, eps=1e-7):
    y_pred = torch.clamp(y_pred, eps, 1-eps)
    y_true = torch.clamp(y_true, eps, 1-eps)
    bce = - y_true * torch.log(y_pred) - (1 - y_true) * torch.log(1 - y_pred)
    return bce.mean(dim=(1,2)).mean()

class Block(nn.Module):
    """ ConvNeXt Block adapted for 2D audio spectrograms. """
    def __init__(self, dim, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs implemented as linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + x
        return x

class ConvNeXt_Tiny(nn.Module):
    """A tiny ConvNeXt-style model for salience prediction."""
    def __init__(self, config):
        super().__init__()
        # Get hyperparameters from config
        model_cfg = config['model_params']
        input_channels = model_cfg.get('input_channels', 5)
        initial_dim = model_cfg.get('initial_dim', 40)
        depths = model_cfg.get('depths', [1, 1, 1])    # Number of blocks at each stage

        self.downsample_layers = nn.ModuleList()
        # Initial stem layer to project input channels to initial_dim
        stem = nn.Sequential(
            nn.Conv2d(input_channels, initial_dim, kernel_size=1),
            nn.GroupNorm(1, initial_dim) # Equivalent to LayerNorm for this shape
        )
        self.downsample_layers.append(stem)

        # Main stages
        stages = []
        dims = [initial_dim] * len(depths) # For this tiny version, we won't increase dim
        for i in range(len(depths)):
            # No downsampling between stages for this task to preserve frequency resolution
            stage = nn.Sequential(
                *[Block(dim=dims[i]) for j in range(depths[i])]
            )
            stages.append(stage)
        self.stages = nn.ModuleList(stages)

        # Final output layer
        self.output_layer = nn.Sequential(
            nn.Conv2d(dims[-1], 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.downsample_layers[0](x)
        x = self.stages[0](x)
        x = self.stages[1](x)
        x = self.stages[2](x)
        x = self.output_layer(x)
        return x

# --- MODEL REGISTRY ---
# Place this dictionary right after your model class definitions
MODEL_REGISTRY = {
    "SalienceNetV1": SalienceCNN,
    "ConvNeXt_Tiny": ConvNeXt_Tiny
}


def _to_mir_eval_sequence(times, freqs, atol=1e-6):
    """
    Convert event lists into the format expected by mir_eval.multipitch:
    a sorted array of unique times and a list of 1-D arrays of frequencies.
    """
    times, freqs = np.asarray(times, dtype=float), np.asarray(freqs, dtype=float)
    if times.size == 0:
        return np.asarray([]), []

    # Data is already sorted by time, so we just need to group
    out_times = []
    out_freqs = []
    
    # Start with the first event
    cur_t = times[0]
    cur_freqs = [freqs[0]]

    for t, f in zip(times[1:], freqs[1:]):
        if abs(t - cur_t) <= atol: # If it's the same time frame, append frequency
            cur_freqs.append(f)
        else: # If it's a new time frame, save the old one and start a new one
            out_times.append(cur_t)
            out_freqs.append(np.asarray(cur_freqs, dtype=float))
            cur_t = t
            cur_freqs = [f]

    # Append the final group
    out_times.append(cur_t)
    out_freqs.append(np.asarray(cur_freqs, dtype=float))

    return np.asarray(out_times, dtype=float), out_freqs

def calculate_f1_score(ref_times, ref_freqs, est_times, est_freqs):
    ref_times, ref_freqs = np.atleast_1d(ref_times), np.atleast_1d(ref_freqs)
    est_times, est_freqs = np.atleast_1d(est_times), np.atleast_1d(est_freqs)

    # Convert to mir_eval's required format
    ref_time_seq, ref_freqs_seq = _to_mir_eval_sequence(ref_times, ref_freqs)
    est_time_seq, est_freqs_seq = _to_mir_eval_sequence(est_times, est_freqs)

    if est_time_seq.size == 0 or ref_time_seq.size == 0:
        scores = {'Precision': 0.0, 'Recall': 0.0, 'Accuracy': 0.0}
    else:
        scores = mir_eval.multipitch.evaluate(ref_time_seq, ref_freqs_seq, est_time_seq, est_freqs_seq)

    # Manually calculate F1-score for consistency
    p, r = scores['Precision'], scores['Recall']
    if p + r == 0:
        f1 = 0.0
    else:
        f1 = 2 * p * r / (p + r)
    
    scores['F1-score'] = f1
    return scores

# -----------------------------------------------------------------------------
# SECTION 3: DATA HANDLING AND PROCESSING
# -----------------------------------------------------------------------------

class DataProcessor:
    """
    Processes a group of tracks into a cached CQT/F0 representation for the entire mix.
    """
    def __init__(self, config, root_dir, cache_dir, log_callback):
        self.config, self.root_dir, self.cache_dir = config, root_dir, cache_dir
        self.dp = self.config['data_params']
        self.n_bins = self.dp['n_octaves'] * self.dp['bins_per_octave']
        self.log = log_callback

    
    def get_dataset_folder(self, stem_name):
        """Finds the correct dataset subfolder for a given stem."""
        if "choralsynth" in stem_name.lower():
            return "ChoralSynth" # Or whatever the folder is named

        # Keep existing logic for Cantoria/DCS
        is_cantoria = stem_name.lower().startswith("cantoria")
        is_dcs = stem_name.lower().startswith("dcs")
        try:
            for folder_name in os.listdir(self.root_dir):
                if not os.path.isdir(os.path.join(self.root_dir, folder_name)): continue
                lower_folder_name = folder_name.lower()
                if is_cantoria and "cantoria" in lower_folder_name: return folder_name
                if is_dcs and ("dcs" in lower_folder_name or "dagstuhl" in lower_folder_name): return folder_name
        except FileNotFoundError: pass
        self.log(f"Warning: Could not determine dataset folder for stem '{stem_name}'.")
        return ""

    # In the DataProcessor class

    def process_and_cache_group(self, track_stems):
        """
        Processes and caches an entire group of stems as a single mix. This function
        handles different path structures for different datasets.
        """
        # Create a single, stable, unique string for the group by sorting and joining.
        canonical_name = "_".join(sorted(track_stems))
        # Create a SHA-1 hash of this string to get a safe, fixed-length filename.
        group_hash = hashlib.sha1(canonical_name.encode('utf-8')).hexdigest()
        
        cache_fname = f"{group_hash}.npz"
        cache_path = os.path.join(self.cache_dir, cache_fname)

        if os.path.exists(cache_path):
            return cache_path

        self.log(f"Processing & Caching Group: {', '.join(track_stems)}")
        self.log(f"  > Cache file: {cache_path}") # Log the hash for debugging


        # --- AUDIO MIXING ---
        max_len, all_y = 0, []
        valid_stems_for_f0 = [] # Keep track of stems that were successfully loaded

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
                    self.log(f"ERROR: Could not find dataset folder for stem {stem}")
                    continue
                audio_path = os.path.join(self.root_dir, dataset_folder, "Audio", f"{stem}.wav")

            if not os.path.exists(audio_path):
                self.log(f"ERROR: File not found at resolved path: {audio_path}")
                continue
                
            y, _ = librosa.load(audio_path, sr=self.dp['sr'])
            all_y.append(y)
            valid_stems_for_f0.append(stem) # Add to list for F0 processing
            if len(y) > max_len: max_len = len(y)
        
        if not all_y:
            self.log(f"Warning: No valid audio files found for group. Skipping cache generation.")
            return None

        y_mix = np.sum([np.pad(y, (0, max_len - len(y))) for y in all_y], axis=0)
        if np.max(np.abs(y_mix)) > 0:
            y_mix /= np.max(np.abs(y_mix))

        # --- HCQT CALCULATION for the full mix ---
        cqt_list = [librosa.cqt(y_mix, sr=self.dp['sr'], hop_length=self.dp['hop_length'], fmin=self.dp['fmin'] * h, n_bins=self.n_bins, bins_per_octave=self.dp['bins_per_octave']) for h in self.dp['harmonics']]
        min_time = min(c.shape[1] for c in cqt_list)
        hcqt = np.stack([c[:, :min_time] for c in cqt_list])
        log_hcqt = (1.0/80.0) * librosa.amplitude_to_db(np.abs(hcqt), ref=np.max) + 1.0

        # --- GROUND TRUTH F0 MAP (ROBUST VERSION) ---
        n_frames = log_hcqt.shape[2]
        frame_times = librosa.frames_to_time(np.arange(n_frames), sr=self.dp['sr'], hop_length=self.dp['hop_length'])
        cq_freqs = librosa.cqt_frequencies(n_bins=self.n_bins, fmin=self.dp['fmin'], bins_per_octave=self.dp['bins_per_octave'])
        f0_map = np.zeros((self.n_bins, n_frames), dtype=np.float32)
        total_active_frames = 0

        for stem in valid_stems_for_f0:
            is_choralsynth_style = os.sep in stem
            interp_freqs = None # Initialize to avoid reference before assignment
            active_mask = None

            if is_choralsynth_style:
                crepe_path = os.path.join(self.root_dir, "ChoralSynth", f"{stem}.f0.csv")
                if not os.path.exists(crepe_path): continue
                
                crepe_df = pd.read_csv(crepe_path)
                crepe_df.columns = [c.strip() for c in crepe_df.columns]
                
                interp_freqs = np.interp(frame_times, crepe_df['time'], crepe_df['frequency'], left=0, right=0)
                interp_confidence = np.interp(frame_times, crepe_df['time'], crepe_df['confidence'], left=0, right=0)
                active_mask = (interp_freqs >= self.dp['fmin']) & (interp_confidence > 0.5)
            else:
                dataset_folder = self.get_dataset_folder(stem)
                if not dataset_folder: continue
                base_path = os.path.join(self.root_dir, dataset_folder)
                crepe_path = os.path.join(base_path, "F0_crepe", f"{stem}.csv")
                pyin_path = os.path.join(base_path, "F0_pyin", f"{stem}.csv")
                if not os.path.exists(crepe_path) or not os.path.exists(pyin_path): continue
                
                crepe_df = pd.read_csv(crepe_path)
                crepe_df.columns = [c.strip() for c in crepe_df.columns]
                pyin_df = pd.read_csv(pyin_path, header=None, names=['time', 'frequency', 'confidence'])
                
                interp_freqs = np.interp(frame_times, crepe_df['time'], crepe_df['frequency'], left=0, right=0)
                pyin_voiced_mask = (pyin_df['frequency'] > 0).astype(float)
                interp_voiced = np.interp(frame_times, pyin_df['time'], pyin_voiced_mask, left=0, right=0)
                active_mask = (interp_freqs >= self.dp['fmin']) & (interp_voiced > 0.5)

            if active_mask is not None and np.any(active_mask):
                frame_idxs = np.where(active_mask)[0]
                bin_idxs = np.argmin(np.abs(interp_freqs[active_mask, None] - cq_freqs[None, :]), axis=1)
                f0_map[bin_idxs, frame_idxs] = 1.0
                total_active_frames += len(frame_idxs)
        
        if total_active_frames == 0:
            self.log(f"  - WARNING: No active frames found for this group. Ground truth will be all zeros.")
        else:
            self.log(f"  - Found {total_active_frames} active pitch frames for this group.")

        # --- GAUSSIAN SMOOTHING & NORMALIZATION ---
        f0_map = gaussian_filter(f0_map, sigma=(self.dp['gaussian_sigma'], 0), mode='constant')
        if f0_map.max() > 0:
            f0_map /= f0_map.max()
        
        np.savez_compressed(cache_path, log_hcqt=log_hcqt, f0_map=f0_map)
        return cache_path

class PatchDataset(Dataset):
    """
    Creates patches from pre-computed and cached CQT/F0 maps of full track groups.
    """
    def __init__(self, track_groups, root_dir, cache_dir, config, is_train, log_callback):
        self.config, self.is_train = config, is_train
        self.log = log_callback
        self.root_dir = root_dir # Add this line
        self.cache_dir = cache_dir # Add this line

        self.tp, self.dp = self.config['training_params'], self.config['data_params']
        self.patch_width_frames = self.tp['patch_width']
        self.step_size = int(self.patch_width_frames * (1 - self.tp.get('patch_overlap', 0.5))) # Use .get for safety

        processor = DataProcessor(config, root_dir, cache_dir, log_callback)
        
        self.index = []
        self.cache_data = []

        for group in tqdm(track_groups, desc="Checking and loading cache", leave=False):           
            # Use the exact same hashing logic as the DataProcessor to find the file.
            canonical_name = "_".join(sorted(group))
            group_hash = hashlib.sha1(canonical_name.encode('utf-8')).hexdigest()
            cache_fname = f"{group_hash}.npz"
            cache_path = os.path.join(self.cache_dir, cache_fname)

            # Now, ensure the file exists by processing if needed
            # The `process_and_cache_group` function will use the same hash logic internally.
            if not os.path.exists(cache_path):
                processor.process_and_cache_group(group)
            
            # Load the data into memory once
            try:
                data = np.load(cache_path)
                self.cache_data.append({
                    'log_hcqt': data['log_hcqt'],
                    'f0_map': data['f0_map']
                })
                # Build the index to get patches from this data
                n_frames = data['log_hcqt'].shape[2]
                current_data_idx = len(self.cache_data) - 1
                for start in range(0, n_frames - self.patch_width_frames + 1, self.step_size):
                    self.index.append((current_data_idx, start))
            except Exception as e:
                self.log(f"ERROR: Could not load or process cache file {cache_path}. Error: {e}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        # Retrieve the data index and start frame for the patch
        data_idx, start_frame = self.index[idx]
        
        # Get the pre-loaded data
        hcqt_full = self.cache_data[data_idx]['log_hcqt']
        f0_map_full = self.cache_data[data_idx]['f0_map']

        # Slice the patch
        end_frame = start_frame + self.patch_width_frames
        cqt_patch = hcqt_full[:, :, start_frame:end_frame]
        f0_patch = f0_map_full[:, start_frame:end_frame]
        
        return torch.from_numpy(cqt_patch.copy()).float(), torch.from_numpy(f0_patch.copy()).unsqueeze(0).float()

class DatasetManager:
    def __init__(self, root_dir, cache_dir, log_callback=print):
        self.root_dir, self.cache_dir, self.log = root_dir, cache_dir, log_callback
        self.track_groups = self._discover_datasets()
        self.log(f"Discovery complete. Found {len(self.track_groups.get('Cantoria', {}))} Cantoria & {len(self.track_groups.get('DCS', {}))} DCS valid track groups.")

    def _discover_datasets(self):
        self.log("Scanning for datasets using hybrid discovery strategy...")
        all_groups = {"Cantoria": {}, "DCS": {}}
        
        # Find potential dataset folders in the root directory
        try:
            dataset_folders = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        except FileNotFoundError:
            self.log(f"ERROR: Dataset root directory not found at {self.root_dir}")
            return all_groups

        for dataset_folder in dataset_folders:
            base_path = os.path.join(self.root_dir, dataset_folder)
            
            # --- Strategy 1: For Cantoria (using your proven logic) ---
            if "cantoria" in dataset_folder.lower():
                self.log(f"-> Applying Cantoria discovery logic to: '{dataset_folder}'")
                audio_dir = os.path.join(base_path, "Audio")
                crepe_dir = os.path.join(base_path, "F0_crepe")
                pyin_dir = os.path.join(base_path, "F0_pyin")
                
                if not os.path.exists(audio_dir): continue
                
                voice_parts = ['S', 'A', 'T', 'B']
                audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
                
                # Create track base names (e.g., 'Cantoria_0001')
                track_bases = set()
                for f in audio_files:
                    if f.startswith('Cantoria_') and len(f.split('_')) == 3:
                        track_bases.add(f.rsplit('_', 1)[0])

                for base in sorted(list(track_bases)):
                    # Check if all 4 voice parts for all 3 file types exist
                    is_complete_group = True
                    for part in voice_parts:
                        stem = f"{base}_{part}"
                        if not (os.path.exists(os.path.join(audio_dir, f"{stem}.wav")) and
                                os.path.exists(os.path.join(crepe_dir, f"{stem}.csv")) and
                                os.path.exists(os.path.join(pyin_dir, f"{stem}.csv"))):
                            is_complete_group = False
                            break
                    
                    if is_complete_group:
                        group_id = base # The group ID is the base name, e.g., 'Cantoria_0001'
                        stems = [f"{base}_{p}" for p in voice_parts]
                        all_groups["Cantoria"][group_id] = stems

            # --- Strategy 2: For DCS (the general logic) ---
            elif "dcs" in dataset_folder.lower() or "dagstuhl" in dataset_folder.lower():
                self.log(f"-> Applying DCS discovery logic to: '{dataset_folder}'")
                audio_dir = os.path.join(base_path, "Audio")
                if not os.path.exists(audio_dir): continue

                potential_groups_for_dataset = defaultdict(list)
                for filename in os.listdir(audio_dir):
                    if not filename.endswith('.wav'): continue
                    base_name = filename.rsplit('.', 1)[0]
                    parts = base_name.split('_')
                    mic_type = parts[-1]
                    if mic_type.upper() in {'DYN', 'LRX', 'HSM'}:
                        group_id = "_".join(parts[:-2] + [mic_type])
                        potential_groups_for_dataset[group_id].append(base_name)
                
                for group_id, stems in potential_groups_for_dataset.items():
                    is_valid = all(
                        os.path.exists(os.path.join(base_path, "Audio", f"{s}.wav")) and
                        os.path.exists(os.path.join(base_path, "F0_crepe", f"{s}.csv")) and
                        os.path.exists(os.path.join(base_path, "F0_pyin", f"{s}.csv"))
                        for s in stems
                    )
                    if is_valid:
                        all_groups["DCS"][group_id] = stems

            # --- Strategy 3: For ChoralSynth ---
            # This assumes ChoralSynth dataset is in a folder like "ChoralSynth"
            elif "choralsynth" in dataset_folder.lower():
                self.log(f"-> Applying ChoralSynth discovery logic to: '{dataset_folder}'")
                base_path = os.path.join(self.root_dir, dataset_folder)
                
                # ChoralSynth has subfolders for each track
                track_folders = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

                for track_folder in track_folders:
                    voices_dir = os.path.join(base_path, track_folder, "voices")
                    if not os.path.exists(voices_dir):
                        continue
                    
                    # The track_folder name (e.g., "08_Anima_nostra") is the group_id
                    group_id = track_folder
                    stems_in_group = []
                    
                    # Find all .wav files and check for corresponding .f0.csv
                    for filename in os.listdir(voices_dir):
                        if filename.endswith('.wav'):
                            stem_name = filename[:-4] # e.g., "ALTUS"
                            # The "stem" now needs to include the full path context to be unique
                            full_stem_path_prefix = os.path.join(track_folder, "voices", stem_name)
                            
                            f0_path = os.path.join(base_path, f"{full_stem_path_prefix}.f0.csv")
                            audio_path = os.path.join(base_path, f"{full_stem_path_prefix}.wav")

                            if os.path.exists(f0_path) and os.path.exists(audio_path):
                                stems_in_group.append(full_stem_path_prefix)
                    
                    if stems_in_group:
                        # We need a way to distinguish this dataset's groups
                        dataset_name = "ChoralSynth"
                        if dataset_name not in all_groups:
                            all_groups[dataset_name] = {}
                        all_groups[dataset_name][group_id] = stems_in_group
        return all_groups

    def get_dataloaders(self, config):
        """
        Creates and returns training and validation DataLoaders.
        
        This function performs three main tasks:
        1. Gathers all track groups from the multiple training datasets selected in the config.
        2. Splits this combined pool of track groups into a training set and a validation set
        based on the validation split ratio.
        3. Creates PyTorch PatchDataset and DataLoader objects for both sets.
        """
        # Define an empty dataset class for robust error handling
        class EmptyDataset(Dataset):
            def __len__(self): return 0
            def __getitem__(self, idx): raise IndexError

        # Check if a train/validation split is already defined in the config
        if 'train_groups' in config['data_params'] and 'val_groups' in config['data_params']:
            self.log("Loading persistent train/validation split from config.")
            train_groups = config['data_params']['train_groups']
            val_groups = config['data_params']['val_groups']
            new_config = None # No changes to config, so nothing to return

        else:
            # --- NO SPLIT FOUND, GENERATE A NEW ONE ---
            self.log("No split found in config. Generating a new random train/validation split.")

            # 1. GATHER AND COMBINE TRACK GROUPS
            train_dataset_names = config['data_params'].get('train_datasets', [])
            # ... (rest of the gathering logic is the same as before) ...
            all_track_groups = []
            for name in train_dataset_names:
                if name in self.track_groups:
                    all_track_groups.extend(list(self.track_groups[name].values()))
                else:
                    self.log(f"Warning: Selected training dataset '{name}' not found in manager.")
            
            if not all_track_groups:
                self.log("ERROR: No valid track groups found for the selected training datasets.")
                return DataLoader(EmptyDataset()), DataLoader(EmptyDataset()), None

            random.shuffle(all_track_groups)

            # 2. PERFORM THE SPLIT
            num_total = len(all_track_groups)
            val_split_ratio = config['training_params'].get('val_split_ratio', 0.1)
            num_val = int(num_total * val_split_ratio)
            # ... (rest of the splitting logic is the same) ...
            if num_total > 1 and num_val == 0: num_val = 1
            if num_total > 1 and num_val == num_total: num_val = num_total - 1

            val_groups = all_track_groups[:num_val]
            train_groups = all_track_groups[num_val:]
            if not train_groups and val_groups: train_groups = val_groups

            # --- IMPORTANT NEW STEP: UPDATE THE CONFIG OBJECT ---
            # We add the generated split to the config dictionary so it can be saved.
            config['data_params']['train_groups'] = train_groups
            config['data_params']['val_groups'] = val_groups
            new_config = config # This updated config will be returned to be saved.

        # --- 3. CREATE DATASET AND DATALOADER OBJECTS (this part is mostly the same) ---
        train_dataset_names_str = ', '.join(config['data_params'].get('train_datasets', []))
        self.log(f"Using datasets: {train_dataset_names_str}")
        self.log(f"Preparing {len(train_groups)} training groups and {len(val_groups)} validation groups.")

        if not train_groups:
            self.log("ERROR: Training group is empty. Cannot create DataLoader.")
            return DataLoader(EmptyDataset()), DataLoader(EmptyDataset()), new_config

        train_dataset = PatchDataset(train_groups, self.root_dir, self.cache_dir, config, is_train=True, log_callback=self.log)
        val_dataset = PatchDataset(val_groups, self.root_dir, self.cache_dir, config, is_train=False, log_callback=self.log)

        if len(train_dataset) == 0:
            self.log(f"ERROR: Training dataset is empty after processing patches. Audio files might be too short.")
            return DataLoader(EmptyDataset()), DataLoader(EmptyDataset()), new_config

        train_loader = DataLoader(train_dataset, batch_size=config['training_params']['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=config['evaluation_params']['eval_batch_size'], shuffle=False, num_workers=0, pin_memory=True)
        
        return train_loader, val_loader, new_config

# -----------------------------------------------------------------------------
# SECTION 4: TRAINING & EVALUATION ENGINE
# -----------------------------------------------------------------------------

class Trainer:
    """Handles the model training and validation loop with checkpoint resuming."""
    def __init__(self, config, device, data_manager, log_callback, progress_callback, epoch_end_callback):
        self.config, self.device, self.data_manager = config, device, data_manager
        self.log, self.progress, self.epoch_end_callback = log_callback, progress_callback, epoch_end_callback
        
        # The checkpoint directory is based on the hash (run_id)
        self.checkpoint_dir = os.path.join("checkpoints", config['run_id'])
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train(self):
        self.log(f"Starting training for run: {self.config['run_id']}")
        
        # --- DYNAMIC MODEL INSTANTIATION ---
        model_name = self.config['model_params']['architecture_name']
        if model_name not in MODEL_REGISTRY:
            self.log(f"ERROR: Model '{model_name}' not found in registry. Aborting.")
            return None, None
        
        model_class = MODEL_REGISTRY[model_name]
        model = model_class(self.config).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config['training_params']['learning_rate'])
        criterion = bkld_loss

        self.log(f"Model: '{model_name}' created with {sum(p.numel() for p in model.parameters()):,} parameters.")

        # --- PERSISTENT DATA SPLIT LOGIC ---
        config_path = os.path.join(self.checkpoint_dir, 'config.json')
        if os.path.exists(config_path):
            self.log("Found existing config.json, loading for persistent data split.")
            with open(config_path, 'r') as f:
                self.config = json.load(f)

        train_loader, val_loader, updated_config = self.data_manager.get_dataloaders(self.config)

        if updated_config is not None:
            self.log("A new train/validation split was generated. Saving to config.json.")
            self.config = updated_config
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
        
        # --- DYNAMIC STEP CALCULATION ---
        tp = self.config['training_params']
        num_epochs = tp.get('num_epochs', 30)
        batch_size = tp.get('batch_size', 16)
        
        if train_loader is None or len(train_loader.dataset) == 0:
            self.log("ERROR: Cannot calculate training steps, training dataset is empty.")
            return None, None
            
        num_patches = len(train_loader.dataset)
        steps_per_epoch = (num_patches + batch_size - 1) // batch_size
        total_train_steps = steps_per_epoch * num_epochs
        steps_per_checkpoint = tp.get("steps_per_checkpoint", max(1, steps_per_epoch // 2))
        
        self.log(f"Training configured for {num_epochs} epochs.")
        self.log(f"Dataset has {num_patches} patches, with batch size {batch_size}.")
        self.log(f"This translates to {steps_per_epoch} steps/epoch, for a total of {total_train_steps} training steps.")
        self.log(f"Checkpoints and validation will occur every {steps_per_checkpoint} steps.")
        
        # --- SINGLE, UNIFIED CHECKPOINT AND METRICS LOADING ---
        global_step = 0
        best_val_loss = float('inf')
        metrics = {'train_loss': [], 'val_loss': [], 'steps': []}
        
        metrics_path = os.path.join(self.checkpoint_dir, 'metrics.json')
        latest_model_path = os.path.join(self.checkpoint_dir, "latest_model.pth")

        if os.path.exists(latest_model_path):
            self.log("Found existing model checkpoint. Resuming training.")
            try:
                checkpoint = torch.load(latest_model_path, map_location=self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                global_step = checkpoint.get('global_step', 0)
                best_val_loss = checkpoint.get('best_val_loss', float('inf'))
                
                if os.path.exists(metrics_path):
                    with open(metrics_path, 'r') as f:
                        metrics = json.load(f)
                
                self.log(f"Resuming from step {global_step}. Best validation loss so far: {best_val_loss:.4f}")
                self.epoch_end_callback(metrics.copy())
            except Exception as e:
                self.log(f"Error loading checkpoint, starting fresh. Error: {e}")
                global_step = 0
                best_val_loss = float('inf')
                metrics = {'train_loss': [], 'val_loss': [], 'steps': []}
        else:
            self.log("No checkpoint found. Starting fresh training.")

        # --- MAIN STEP-BASED TRAINING LOOP ---
        if global_step >= total_train_steps:
            self.log("Model has already been trained for the specified number of steps or more.")
            self.epoch_end_callback(metrics)
            return self.checkpoint_dir, metrics

        model.train()
        train_iterator = iter(train_loader)
        pbar = tqdm(initial=global_step, total=total_train_steps, desc="Training Steps")
        
        while global_step < total_train_steps:
            try:
                cqt_batch, f0_batch = next(train_iterator)
            except StopIteration:
                self.log(f"--- Epoch finished at step {global_step}. Shuffling data for next epoch. ---")
                train_iterator = iter(train_loader)
                cqt_batch, f0_batch = next(train_iterator)
                
            cqt_batch, f0_batch = cqt_batch.to(self.device), f0_batch.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(cqt_batch)
            loss = criterion(outputs, f0_batch)
            loss.backward()
            optimizer.step()

            if (global_step + 1) % steps_per_checkpoint == 0 or (global_step + 1) == total_train_steps:
                avg_val_loss = self._run_validation(model, val_loader, criterion)
                
                metrics['val_loss'].append(avg_val_loss)
                metrics['steps'].append(global_step + 1)
                
                self.log(f"Step {global_step+1}/{total_train_steps} | Val Loss: {avg_val_loss:.4f}")
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(model.state_dict(), os.path.join(self.checkpoint_dir, "best_model.pth"))
                    self.log(f"  > New best val_loss. Best model saved.")

                torch.save({
                    'global_step': global_step + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                }, latest_model_path)

                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=4)
                
                self.epoch_end_callback(metrics.copy())
                model.train()

            global_step += 1
            pbar.update(1)
            self.progress(global_step, total_train_steps)

        pbar.close()
        self.log("Training finished.")
        return self.checkpoint_dir, metrics
    
    def _run_validation(self, model, val_loader, criterion):
        """Helper to run the validation loop and return the average loss."""
        model.eval()
        total_val_loss = 0
        if val_loader is None or len(val_loader) == 0:
            return float('inf')

        with torch.no_grad():
            for cqt_batch, f0_batch in val_loader:
                cqt_batch, f0_batch = cqt_batch.to(self.device), f0_batch.to(self.device)
                outputs = model(cqt_batch)
                loss = criterion(outputs, f0_batch)
                total_val_loss += loss.item()
        return total_val_loss / len(val_loader)

class Evaluator:
    """Same as the provided complete version."""
    def __init__(self, checkpoint_path, device, log_callback, progress_callback):
        self.device = device
        self.log = log_callback
        self.progress = progress_callback
        
        config_path = os.path.join(checkpoint_path, "config.json")
        model_path = os.path.join(checkpoint_path, "best_model.pth")
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        # --- DYNAMIC MODEL INSTANTIATION ---
        model_name = self.config['model_params']['architecture_name']
        if model_name not in MODEL_REGISTRY:
            self.log(f"ERROR: Model '{model_name}' from checkpoint not found in registry. Cannot evaluate.")
            # Handle this gracefully, maybe by raising an exception or setting a flag
            self.model = None
            return

        model_class = MODEL_REGISTRY[model_name]
        self.model = model_class(self.config).to(self.device)
        # --- END OF CHANGE ---

        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.log(f"Evaluator ready. Loaded model '{model_name}' from {checkpoint_path}")

    def _run_inference(self, audio_mix):
        dp, tp, n_bins = self.config['data_params'], self.config['training_params'], self.config['data_params']['n_octaves'] * self.config['data_params']['bins_per_octave']
        cqt_list = [librosa.cqt(audio_mix, sr=dp['sr'], hop_length=dp['hop_length'], fmin=dp['fmin'] * h, n_bins=n_bins, bins_per_octave=dp['bins_per_octave']) for h in dp['harmonics']]
        min_time = min(c.shape[1] for c in cqt_list); hcqt = np.stack([c[:, :min_time] for c in cqt_list])
        log_hcqt = (1.0/80.0) * librosa.amplitude_to_db(np.abs(hcqt), ref=np.max) + 1.0
        patch_width, n_f = tp['patch_width'], log_hcqt.shape[2]; step = int(patch_width * (1-tp['patch_overlap']))
        if n_f < patch_width: return np.zeros((n_bins, n_f))
        patches = np.stack([log_hcqt[:, :, st:st+patch_width] for st in range(0, n_f - patch_width + 1, step)])
        dataset = TensorDataset(torch.from_numpy(patches).float()); loader = DataLoader(dataset, batch_size=self.config['evaluation_params']['eval_batch_size'], shuffle=False)
        total_frames = (len(patches) - 1) * step + patch_width; out_map, ov_count = np.zeros((n_bins, total_frames)), np.zeros((n_bins, total_frames))
        with torch.no_grad():
            for i, (batch,) in enumerate(tqdm(loader, desc="Inference", leave=False)):
                preds = self.model(batch.to(self.device)).squeeze(1).cpu().numpy()
                for j, p in enumerate(preds): start_frame = (i * loader.batch_size + j) * step; out_map[:, start_frame:start_frame+patch_width] += p; ov_count[:, start_frame:start_frame+patch_width] += 1
        ov_count[ov_count == 0] = 1e-6; salience_map = out_map / ov_count
        if salience_map.max() > 0: salience_map /= salience_map.max()
        return salience_map
    
    def _get_ground_truth(self, track_stems, root_dir, n_frames):
        """Reuses DataProcessor logic to get ground truth F0 map."""
        dp = self.config['data_params']
        frame_times = librosa.frames_to_time(np.arange(n_frames), sr=dp['sr'], hop_length=dp['hop_length'])
        
        ref_times_list, ref_freqs_list = [], []

        for stem in track_stems:
            is_choralsynth_style = os.sep in stem
            interp_freqs = None # Initialize to avoid reference before assignment
            active = None

            if is_choralsynth_style:
                # ChoralSynth F0 file logic
                crepe_path = os.path.join(root_dir, "ChoralSynth", f"{stem}.f0.csv")
                if not os.path.exists(crepe_path): continue

                crepe_df = pd.read_csv(crepe_path)
                crepe_df.columns = [c.strip() for c in crepe_df.columns]
                
                interp_freqs = np.interp(frame_times, crepe_df['time'], crepe_df['frequency'], left=0, right=0)
                interp_confidence = np.interp(frame_times, crepe_df['time'], crepe_df['confidence'], left=0, right=0)
                active = (interp_freqs >= dp['fmin']) & (interp_confidence > 0.5)

            else:
                # Cantoria/DCS F0 file logic
                processor = DataProcessor(self.config, root_dir, "./cache", self.log)
                dataset_folder = processor.get_dataset_folder(stem)
                if not dataset_folder: continue

                base_path = os.path.join(root_dir, dataset_folder)
                crepe_path = os.path.join(base_path, "F0_crepe", f"{stem}.csv")
                pyin_path = os.path.join(base_path, "F0_pyin", f"{stem}.csv")
                if not os.path.exists(crepe_path) or not os.path.exists(pyin_path): continue

                crepe_df = pd.read_csv(crepe_path)
                crepe_df.columns = [c.strip() for c in crepe_df.columns]
                pyin_df = pd.read_csv(pyin_path, header=None, names=['time', 'frequency', 'confidence'])
                
                interp_freqs = np.interp(frame_times, crepe_df['time'], crepe_df['frequency'], left=0, right=0)
                pyin_voiced_mask = (pyin_df['frequency'] > 0).astype(float)
                interp_voiced = np.interp(frame_times, pyin_df['time'], pyin_voiced_mask, left=0, right=0)
                active = (interp_freqs >= dp['fmin']) & (interp_voiced > 0.5)

            if active is not None and np.any(active):
                ref_times_list.extend(frame_times[active])
                ref_freqs_list.extend(interp_freqs[active])
        
        # Convert to numpy arrays
        ref_times = np.array(ref_times_list)
        ref_freqs = np.array(ref_freqs_list)

        if ref_times.size == 0:
            return ref_times, ref_freqs # Return empty arrays if no pitches found

        # Get the indices that would sort the time array
        sort_indices = np.argsort(ref_times)
        
        # Apply these indices to both arrays to sort them together
        sorted_ref_times = ref_times[sort_indices]
        sorted_ref_freqs = ref_freqs[sort_indices]

        return sorted_ref_times, sorted_ref_freqs
    
    def _extract_pitches(self, salience_map, threshold):
        dp, n_bins = self.config['data_params'], self.config['data_params']['n_octaves'] * self.config['data_params']['bins_per_octave']
        times = librosa.times_like(salience_map, sr=dp['sr'], hop_length=dp['hop_length'])
        freqs = librosa.cqt_frequencies(n_bins=n_bins, fmin=dp['fmin'], bins_per_octave=dp['bins_per_octave'])
        est_times, est_freqs = [], []
        for t_idx in range(salience_map.shape[1]):
            peaks, _ = find_peaks(salience_map[:, t_idx], height=threshold)
            if peaks.size > 0: est_times.extend([times[t_idx]] * len(peaks)); est_freqs.extend(freqs[peaks])
        return np.array(est_times), np.array(est_freqs)
    def evaluate_track(self, track_stems, root_dir, threshold):
        self.log(f"Evaluating: {', '.join(track_stems)} @ thresh {threshold:.2f}")
        processor = DataProcessor(self.config, root_dir, "./cache", self.log)
        max_len, all_y = 0, []
        for stem in track_stems:
            dataset_folder = processor.get_dataset_folder(stem)
            audio_path = os.path.join(root_dir, dataset_folder, "Audio", f"{stem}.wav")
            y, _ = librosa.load(audio_path, sr=self.config['data_params']['sr']); all_y.append(y)
            if len(y) > max_len: max_len = len(y)
        audio_mix = np.sum([np.pad(y, (0, max_len - len(y))) for y in all_y], axis=0)
        salience_map = self._run_inference(audio_mix)
        ref_times, ref_freqs = self._get_ground_truth(track_stems, root_dir, salience_map.shape[1])
        est_times, est_freqs = self._extract_pitches(salience_map, threshold)
        scores = calculate_f1_score(ref_times, ref_freqs, est_times, est_freqs)
        self.log(f"Scores: F1={scores['F1-score']:.3f}, P={scores['Precision']:.3f}, R={scores['Recall']:.3f}")
        return scores, (ref_times, ref_freqs), (est_times, est_freqs)
    
    def tune_threshold(self, all_eval_track_groups, root_dir):
        """
        Tunes the peak-picking threshold using an efficient ternary search to find
        the value that maximizes the AVERAGE F1-score across the evaluation set.
        """
        num_tracks = len(all_eval_track_groups)
        self.log(f"Tuning threshold across {num_tracks} tracks using Ternary Search...")
        
        processor = DataProcessor(self.config, root_dir, "./cache", self.log)
        
        # --- Step 1: Pre-compute salience maps (this part remains the same) ---
        salience_maps = {}
        ground_truths = {}
        
        track_values = list(all_eval_track_groups.values())
        for i, track_stems in enumerate(tqdm(track_values, desc="Pre-computing Salience Maps", leave=False)):
            group_id = "_".join(sorted(track_stems))
            
            max_len, all_y = 0, []
            for stem in track_stems:
                dataset_folder = processor.get_dataset_folder(stem)
                audio_path = os.path.join(root_dir, dataset_folder, "Audio", f"{stem}.wav")
                y, _ = librosa.load(audio_path, sr=self.config['data_params']['sr'])
                all_y.append(y)
                if len(y) > max_len: max_len = len(y)
            audio_mix = np.sum([np.pad(y, (0, max_len - len(y))) for y in all_y], axis=0)

            salience_map = self._run_inference(audio_mix)
            salience_maps[group_id] = salience_map
            
            ref_times, ref_freqs = self._get_ground_truth(track_stems, root_dir, salience_map.shape[1])
            ground_truths[group_id] = (ref_times, ref_freqs)
            
            self.progress(i + 1, num_tracks)

        self.log("All salience maps pre-computed. Starting efficient search...")

        # --- Step 2: Helper function to evaluate a single threshold ---
        # This avoids code duplication inside the search loop.
        memo = {} # Memoization to cache results for thresholds we've already seen
        def get_avg_f1(thresh):
            if thresh in memo:
                return memo[thresh]
                
            f1_scores = []
            for track_stems in all_eval_track_groups.values():
                lookup_id = "_".join(sorted(track_stems))
                salience_map = salience_maps[lookup_id]
                ref_times, ref_freqs = ground_truths[lookup_id]
                
                est_times, est_freqs = self._extract_pitches(salience_map, thresh)
                scores = calculate_f1_score(ref_times, ref_freqs, est_times, est_freqs)
                f1_scores.append(scores['F1-score'])
            
            avg_f1 = np.mean(f1_scores)
            self.log(f"  - Testing Threshold {thresh:.4f}: Average F1 = {avg_f1:.4f}")
            memo[thresh] = avg_f1
            return avg_f1

        # --- Step 3: Ternary Search Implementation ---
        low = 0.1
        high = 0.8
        # We can use a fixed number of iterations for simplicity and guaranteed termination
        # 10 iterations will narrow the range [0.1, 0.8] down to a width of ~0.01
        iterations = 10 

        for i in range(iterations):
            # If the range is already very small, we can stop early
            if (high - low) < 0.01:
                break
                
            # Update progress bar for search iterations
            self.progress(i + 1, iterations)

            # Calculate two midpoints
            m1 = low + (high - low) / 3
            m2 = high - (high - low) / 3
            
            f1_m1 = get_avg_f1(m1)
            f1_m2 = get_avg_f1(m2)

            if f1_m1 < f1_m2:
                low = m1  # The peak is in the right two-thirds
            else:
                high = m2 # The peak is in the left two-thirds
                
        # The optimal threshold is now within the small [low, high] range.
        # We can take the midpoint as our final answer.
        best_thresh = (low + high) / 2
        best_avg_f1 = get_avg_f1(best_thresh) # One final calculation for the report

        self.log(f"--- Optimal threshold found via Ternary Search ---")
        self.log(f"  > Optimal Threshold: {best_thresh:.4f}")
        self.log(f"  > Best Average F1-score: {best_avg_f1:.4f}")
        
        return best_thresh, best_avg_f1

class HyperparameterTuner:
    def __init__(self, base_config, device, data_manager, log_callback, progress_callback):
        self.base_config = base_config
        self.device = device
        self.data_manager = data_manager
        self.log = log_callback
        self.progress = progress_callback
        self.tuning_params = base_config['tuning_params']
        self.search_space = self.tuning_params['search_space']
        self.history = {'best_fitness': [], 'avg_fitness': []}

    def _create_individual(self):
        ind = {}
        for key, space in self.search_space.items():
            if space['type'] == 'log_uniform':
                ind[key] = np.exp(random.uniform(np.log(space['range'][0]), np.log(space['range'][1])))
            elif space['type'] == 'uniform':
                ind[key] = random.uniform(space['range'][0], space['range'][1])
            elif space['type'] == 'choice':
                ind[key] = random.choice(space['choices'])
        return ind

    def _evaluate_fitness(self, individual_params):
        # Create a temporary config for this evaluation run
        eval_config = self.base_config.copy()
        eval_config['training_params']['learning_rate'] = individual_params['learning_rate']
        eval_config['data_params']['gaussian_sigma'] = individual_params['gaussian_sigma']
        eval_config['training_params']['num_epochs'] = self.tuning_params['epochs_per_eval']
        
        run_id = f"tune_{datetime.now().strftime('%H%M%S_%f')}"
        eval_config['run_id'] = run_id
        
        # Short training run
        trainer = Trainer(eval_config, self.device, self.data_manager, lambda msg: None, lambda c, t: None)
        checkpoint_dir, _ = trainer.train()

        # Evaluation on one validation track
        evaluator = Evaluator(checkpoint_dir, self.device, lambda msg: None, lambda c, t: None)
        val_dataset_name = eval_config['data_params']['eval_dataset']
        val_track_stems = random.choice(list(self.data_manager.track_groups[val_dataset_name].values()))
        
        # Use a fixed threshold for fair comparison during tuning
        scores, _, _ = evaluator.evaluate_track(val_track_stems, self.data_manager.root_dir, 0.3)
        
        shutil.rmtree(checkpoint_dir) # Clean up temporary checkpoint
        return scores['F1-score']

    def _selection(self, population, fitnesses):
        # Tournament selection
        selected = []
        for _ in range(len(population)):
            i1, i2 = random.sample(range(len(population)), 2)
            winner = i1 if fitnesses[i1] > fitnesses[i2] else i2
            selected.append(population[winner])
        return selected

    def _crossover(self, p1, p2):
        if random.random() > self.tuning_params['crossover_rate']: return p1.copy(), p2.copy()
        c1, c2 = p1.copy(), p2.copy()
        for key in self.search_space:
            if random.random() < 0.5: c1[key], c2[key] = c2[key], c1[key]
        return c1, c2

    def _mutation(self, individual):
        if random.random() > self.tuning_params['mutation_rate']: return individual
        mutated_ind = individual.copy()
        key_to_mutate = random.choice(list(self.search_space.keys()))
        space = self.search_space[key_to_mutate]
        if space['type'] == 'log_uniform':
            mutated_ind[key_to_mutate] = np.exp(random.uniform(np.log(space['range'][0]), np.log(space['range'][1])))
        elif space['type'] == 'uniform':
            mutated_ind[key_to_mutate] = random.uniform(space['range'][0], space['range'][1])
        elif space['type'] == 'choice':
            mutated_ind[key_to_mutate] = random.choice(space['choices'])
        return mutated_ind

    def run_tuning(self):
        population = [self._create_individual() for _ in range(self.tuning_params['population_size'])]
        
        for gen in range(self.tuning_params['num_generations']):
            self.log(f"--- Generation {gen + 1}/{self.tuning_params['num_generations']} ---")
            
            fitnesses = []
            for i, ind in enumerate(population):
                self.log(f"  Evaluating individual {i+1}/{len(population)}: { {k: f'{v:.2e}' if 'rate' in k else f'{v:.2f}' for k,v in ind.items()} }")
                fitness = self._evaluate_fitness(ind)
                fitnesses.append(fitness)
                self.log(f"    > Fitness (F1-score): {fitness:.4f}")
            
            best_fitness_gen = max(fitnesses)
            avg_fitness_gen = sum(fitnesses) / len(fitnesses)
            self.history['best_fitness'].append(best_fitness_gen)
            self.history['avg_fitness'].append(avg_fitness_gen)
            self.progress(gen + 1, self.tuning_params['num_generations'])

            self.log(f"Generation {gen+1} Summary | Best F1: {best_fitness_gen:.4f}, Avg F1: {avg_fitness_gen:.4f}")

            parents = self._selection(population, fitnesses)
            next_population = []
            for i in range(0, len(parents), 2):
                p1, p2 = parents[i], parents[i+1]
                c1, c2 = self._crossover(p1, p2)
                next_population.extend([self._mutation(c1), self._mutation(c2)])
            
            population = next_population
        
        best_gen_idx = np.argmax(self.history['best_fitness'])
        final_best_fitness = self.history['best_fitness'][best_gen_idx]
        self.log(f"Tuning finished. Best F1-score of {final_best_fitness:.4f} achieved in generation {best_gen_idx+1}.")
        return self.history

# -----------------------------------------------------------------------------
# SECTION 5: GUI APPLICATION
# -----------------------------------------------------------------------------

class SalienceStudioApp(ctk.CTk):
    def __init__(self, root_dir):
        super().__init__()
        self.title("Deep Salience Studio"); self.geometry("1200x850"); ctk.set_appearance_mode("dark")
        self.root_dir, self.cache_dir = root_dir, "./cache"
        self.config = get_default_config()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ui_queue = Queue()

        self.grid_columnconfigure(0, weight=1); self.grid_rowconfigure(0, weight=4); self.grid_rowconfigure(1, weight=1)
        self.console_frame = ctk.CTkFrame(self); self.console_frame.grid(row=1, column=0, padx=10, pady=(0,10), sticky="nsew")
        self.console_frame.grid_rowconfigure(1, weight=1); self.console_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(self.console_frame, text="Log Console").grid(row=0, column=0, sticky="w", padx=5)
        self.console = ctk.CTkTextbox(self.console_frame, wrap="word"); self.console.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        self.data_manager = DatasetManager(self.root_dir, self.cache_dir, self.log)
        self.tab_view = ctk.CTkTabview(self, anchor="w"); self.tab_view.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.tab_view.add("Train"); self.tab_view.add("Evaluate"); self.tab_view.add("Hyper-parameter Tuning")

        self._create_train_tab(); self._create_evaluate_tab(); self._create_tuning_tab()
        self.check_ui_queue()
        self.log(f"Welcome to Deep Salience Studio! Using device: {self.device}")

    def _create_train_tab(self):
        tab = self.tab_view.tab("Train")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_columnconfigure(1, weight=1)
        tab.grid_rowconfigure(0, weight=2)
        tab.grid_rowconfigure(1, weight=1)
        
        settings_frame = ctk.CTkScrollableFrame(tab, label_text="Training Configuration")
        settings_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.train_widgets = {}

        # --- Model Parameters ---
        ctk.CTkLabel(settings_frame, text="Model Architecture", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=(10,0))

        # --- Data Parameters ---
        ctk.CTkLabel(settings_frame, text="Training Datasets", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=(10,0))
        self.train_dataset_checkboxes = {}
        # Get all discovered dataset names from the manager
        dataset_names = list(self.data_manager.track_groups.keys())
        for name in dataset_names:
            var = ctk.StringVar(value="on") # Default to training on this dataset
            chk = ctk.CTkCheckBox(settings_frame, text=name, variable=var, onvalue="on", offvalue="off")
            chk.pack(anchor="w", padx=20)
            self.train_dataset_checkboxes[name] = var

        ctk.CTkLabel(settings_frame, text="Evaluation Dataset", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=(10,0))
        self.train_widgets['eval_dataset'] = ctk.CTkOptionMenu(settings_frame, values=dataset_names)
        # Automatically select a different dataset for evaluation
        if len(dataset_names) > 1:
            self.train_widgets['eval_dataset'].set(dataset_names[1])
        else:
            self.train_widgets['eval_dataset'].set(dataset_names[0])
        self.train_widgets['eval_dataset'].pack(fill="x", padx=10, pady=(0,10))

        # --- Training Parameters ---
        ctk.CTkLabel(settings_frame, text="Training Hyperparameters", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=(10,0))
        
        ctk.CTkLabel(settings_frame, text="Learning Rate").pack(anchor="w", padx=10)
        self.train_widgets['learning_rate'] = ctk.CTkEntry(settings_frame, placeholder_text=f"{self.config['training_params']['learning_rate']:.1e}")
        self.train_widgets['learning_rate'].pack(fill="x", padx=10)

        ctk.CTkLabel(settings_frame, text="Epochs").pack(anchor="w", padx=10, pady=(10,0))
        self.train_widgets['num_epochs'] = ctk.CTkEntry(settings_frame, placeholder_text=str(self.config['training_params']['num_epochs']))
        self.train_widgets['num_epochs'].pack(fill="x", padx=10)
        
        ctk.CTkLabel(settings_frame, text="Batch Size").pack(anchor="w", padx=10, pady=(10,0))
        self.train_widgets['batch_size'] = ctk.CTkEntry(settings_frame, placeholder_text=str(self.config['training_params']['batch_size']))
        self.train_widgets['batch_size'].pack(fill="x", padx=10)

        
        # Get model names from the registry
        model_names = list(MODEL_REGISTRY.keys())
        self.train_widgets['architecture_name'] = ctk.CTkOptionMenu(settings_frame, values=model_names)
        self.train_widgets['architecture_name'].set(self.config["model_params"]["architecture_name"])
        self.train_widgets['architecture_name'].pack(fill="x", padx=10, pady=(0,10))

        # --- Control Buttons and Progress Bar ---
        self.train_button = ctk.CTkButton(tab, text="Start Training", command=self.start_training_thread)
        self.train_button.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        
        self.progress_bar = ctk.CTkProgressBar(tab)
        self.progress_bar.set(0)
        self.progress_bar.grid(row=2, column=0, padx=10, pady=10, sticky="ew")

        # --- Visualization Panel ---
        self.train_viz_frame = ctk.CTkFrame(tab)
        self.train_viz_frame.grid(row=0, column=1, rowspan=3, padx=10, pady=10, sticky="nsew")
        ctk.CTkLabel(self.train_viz_frame, text="Training & Validation Loss").pack()
        self.train_canvas_widget = None

    def _create_evaluate_tab(self):
        tab = self.tab_view.tab("Evaluate")
        tab.grid_columnconfigure(0, weight=2)
        tab.grid_columnconfigure(1, weight=1)
        tab.grid_rowconfigure(0, weight=2)
        tab.grid_rowconfigure(1, weight=1)

        settings_frame = ctk.CTkFrame(tab)
        settings_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # --- CREATE WIDGETS FIRST ---
        ctk.CTkLabel(settings_frame, text="Checkpoint").pack(anchor="w", padx=10, pady=(10,0))
        self.checkpoint_menu = ctk.CTkOptionMenu(settings_frame, values=["None"], command=self.on_checkpoint_selected)
        self.checkpoint_menu.pack(fill="x", padx=10)

        ctk.CTkLabel(settings_frame, text="Evaluation Track").pack(anchor="w", padx=10, pady=(10,0))
        self.eval_track_menu = ctk.CTkOptionMenu(settings_frame, values=["None"])
        self.eval_track_menu.pack(fill="x", padx=10)

        ctk.CTkLabel(settings_frame, text="Threshold").pack(anchor="w", padx=10, pady=(10,0))
        self.threshold_entry = ctk.CTkEntry(settings_frame, placeholder_text="0.3")
        self.threshold_entry.pack(fill="x", padx=10)

        self.eval_button = ctk.CTkButton(settings_frame, text="Evaluate Track", command=self.start_evaluation_thread, state="disabled")
        self.eval_button.pack(fill="x", padx=10, pady=20)

        self.tune_single_track_button = ctk.CTkButton(settings_frame, text="Tune Threshold for This Track", command=self.start_single_track_tuning_thread, state="disabled")
        self.tune_single_track_button.pack(fill="x", padx=10, pady=10)

        self.tune_button = ctk.CTkButton(settings_frame, text="Auto-Tune Threshold", command=self.start_tuning_thread, state="disabled")
        self.tune_button.pack(fill="x", padx=10, pady=(0,20))

        self.eval_progress_bar = ctk.CTkProgressBar(settings_frame)
        self.eval_progress_bar.set(0)
        self.eval_progress_bar.pack(fill="x", padx=10, pady=5)
        
        self.eval_viz_frame = ctk.CTkFrame(tab)
        self.eval_viz_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        ctk.CTkLabel(self.eval_viz_frame, text="Prediction vs. Reference Pitch").pack()
        self.eval_canvas_widget = None

        self.refresh_checkpoints()

    def _create_tuning_tab(self):
        tab = self.tab_view.tab("Hyper-parameter Tuning")
        tab.grid_columnconfigure(0, weight=2)
        tab.grid_columnconfigure(1, weight=1)
        tab.grid_rowconfigure(0, weight=2)
        tab.grid_rowconfigure(1, weight=1)

        settings_frame = ctk.CTkScrollableFrame(tab, label_text="Tuning Configuration")
        settings_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.tuning_widgets = {}

        ctk.CTkLabel(settings_frame, text="Genetic Algorithm Parameters", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=(10,0))

        ctk.CTkLabel(settings_frame, text="Population Size").pack(anchor="w", padx=10)
        self.tuning_widgets['population_size'] = ctk.CTkEntry(settings_frame, placeholder_text=str(self.config['tuning_params']['population_size']))
        self.tuning_widgets['population_size'].pack(fill="x", padx=10)

        ctk.CTkLabel(settings_frame, text="Number of Generations").pack(anchor="w", padx=10, pady=(10,0))
        self.tuning_widgets['num_generations'] = ctk.CTkEntry(settings_frame, placeholder_text=str(self.config['tuning_params']['num_generations']))
        self.tuning_widgets['num_generations'].pack(fill="x", padx=10)
        
        ctk.CTkLabel(settings_frame, text="Epochs per Evaluation").pack(anchor="w", padx=10, pady=(10,0))
        self.tuning_widgets['epochs_per_eval'] = ctk.CTkEntry(settings_frame, placeholder_text=str(self.config['tuning_params']['epochs_per_eval']))
        self.tuning_widgets['epochs_per_eval'].pack(fill="x", padx=10)
        
        ctk.CTkLabel(settings_frame, text="Note: Search space (LR, etc.) is currently hard-coded in config.py.").pack(anchor="w", padx=10, pady=(10,0))

        # --- Control Buttons and Progress Bar ---
        self.tune_run_button = ctk.CTkButton(tab, text="Start Hyper-parameter Tuning", command=self.start_hp_tuning_thread)
        self.tune_run_button.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

        self.tune_progress_bar = ctk.CTkProgressBar(tab)
        self.tune_progress_bar.set(0)
        self.tune_progress_bar.grid(row=2, column=0, padx=10, pady=10, sticky="ew")
        
        # --- Visualization Panel ---
        self.tune_viz_frame = ctk.CTkFrame(tab)
        self.tune_viz_frame.grid(row=0, column=1, rowspan=3, padx=10, pady=10, sticky="nsew")
        ctk.CTkLabel(self.tune_viz_frame, text="Tuning Fitness History").pack()
        self.tune_canvas_widget = None

    def start_training_thread(self):
        self.update_config_from_ui()
        
        # --- USE HASH FOR RUN ID ---
        config_hash = get_config_hash(self.config)
        self.config['run_id'] = config_hash # The run_id is now the hash
        
        self.log(f"Config hash (run ID): {config_hash}")
        
        # Pass the new epoch_end_callback to the Trainer
        trainer = Trainer(
            self.config, self.device, self.data_manager, 
            self.log_threadsafe, self.progress_threadsafe, 
            self.epoch_end_threadsafe
        )
        
        threading.Thread(target=self.run_training_and_get_results, args=(trainer,), daemon=True).start()
        self.train_button.configure(state="disabled", text="Training...")

    def run_training_and_get_results(self, trainer):
        checkpoint_dir, final_metrics = trainer.train()
        self.ui_queue.put(("training_complete", (checkpoint_dir, final_metrics)))
    
    def start_evaluation_thread(self):
        try: threshold = float(self.threshold_entry.get())
        except: threshold = 0.3
        checkpoint_path = os.path.join("checkpoints", self.checkpoint_menu.get())
        evaluator = Evaluator(checkpoint_path, self.device, self.log_threadsafe, self.eval_progress_threadsafe)
        eval_dataset_name = evaluator.config['data_params']['eval_dataset']
        track_stems = self.data_manager.track_groups[eval_dataset_name].get(self.eval_track_menu.get())
        threading.Thread(target=self.run_evaluation, args=(evaluator, track_stems, threshold), daemon=True).start()
        self.eval_button.configure(state="disabled"); self.tune_button.configure(state="disabled")

    def run_evaluation(self, evaluator, track_stems, threshold):
        self.ui_queue.put(("evaluation_complete", evaluator.evaluate_track(track_stems, self.root_dir, threshold)))

    def start_tuning_thread(self):
        checkpoint_path = os.path.join("checkpoints", self.checkpoint_menu.get())
        evaluator = Evaluator(checkpoint_path, self.device, self.log_threadsafe, self.eval_progress_threadsafe)
        
        # Get the config from the loaded checkpoint to know which dataset to use
        eval_dataset_name = evaluator.config['data_params']['eval_dataset']
        # --- PASS THE ENTIRE DICTIONARY OF TRACKS ---
        all_tracks_for_eval = self.data_manager.track_groups[eval_dataset_name]
        print("all tracks", all_tracks_for_eval)

        if not all_tracks_for_eval:
            self.log(f"ERROR: No tracks found in the evaluation dataset '{eval_dataset_name}'. Cannot tune threshold.")
            return

        threading.Thread(target=self.run_tuning, args=(evaluator, all_tracks_for_eval), daemon=True).start()
        self.eval_button.configure(state="disabled")
        self.tune_button.configure(state="disabled")

    def run_tuning(self, evaluator, all_tracks_for_eval):
        # We need to pass the dictionary values (the lists of stems) to the method
        best_thresh, best_f1 = evaluator.tune_threshold(all_tracks_for_eval, self.root_dir)
        self.ui_queue.put(("tuning_complete", (best_thresh, best_f1)))

    def start_single_track_tuning_thread(self):
        """Starts threshold tuning for only the currently selected track."""
        checkpoint_path = os.path.join("checkpoints", self.checkpoint_menu.get())
        evaluator = Evaluator(checkpoint_path, self.device, self.log_threadsafe, self.eval_progress_threadsafe)
        
        # Get the config from the loaded checkpoint to know which dataset to use
        eval_dataset_name = evaluator.config['data_params']['eval_dataset']
        
        # --- GET ONLY THE SELECTED TRACK ---
        selected_track_id = self.eval_track_menu.get()
        if selected_track_id == "None":
            self.log("ERROR: No track selected to tune for. Please select a track.")
            return

        # Create a dictionary containing only the selected track's stems
        # This format is required by the tune_threshold method
        track_stems = self.data_manager.track_groups[eval_dataset_name].get(selected_track_id)
        single_track_dict = {selected_track_id: track_stems}
        
        self.log(f"Starting threshold tuning for single track: {selected_track_id}")

        # The run_tuning method is generic enough to work with our single-item dictionary
        threading.Thread(target=self.run_tuning, args=(evaluator, single_track_dict), daemon=True).start()
        
        # Disable all buttons during operation
        self.eval_button.configure(state="disabled")
        self.tune_single_track_button.configure(state="disabled")
        self.tune_button.configure(state="disabled")


    def start_hp_tuning_thread(self):
        self.update_tuning_config_from_ui()
        tuner = HyperparameterTuner(self.config, self.device, self.data_manager, self.log_threadsafe, self.tune_progress_threadsafe)
        threading.Thread(target=self.run_hp_tuning, args=(tuner,), daemon=True).start()
        self.tune_run_button.configure(state="disabled")

    def run_hp_tuning(self, tuner):
        self.ui_queue.put(("hp_tuning_complete", tuner.run_tuning()))

    def check_ui_queue(self):
        try:
            message, data = self.ui_queue.get_nowait()
        except Empty:
            pass
        else:
            if message == "epoch_complete": self.handle_epoch_completion(data)
            elif message == "training_complete": self.handle_training_completion(data)
            elif message == "evaluation_complete": self.handle_evaluation_completion(data)
            elif message == "tuning_complete": self.handle_tuning_completion(data)
            elif message == "hp_tuning_complete": self.handle_hp_tuning_completion(data)
        self.after(100, self.check_ui_queue)

    def handle_epoch_completion(self, metrics):
        """Called after each epoch to update the plot."""
        self.plot_training_loss(metrics)

    def handle_training_completion(self, data):
        """Called only when all epochs are finished."""
        checkpoint_dir, metrics = data
        self.log(f"Training run completed. Final results in {checkpoint_dir}")
        self.train_button.configure(state="normal", text="Start Training")
        self.progress_bar.set(1.0) # Ensure it's full at the end
        self.plot_training_loss(metrics) # Final plot update
        self.refresh_checkpoints()

    def handle_training_completion(self, data):
        checkpoint_dir, metrics = data; self.log(f"Training run completed. Results in {checkpoint_dir}")
        self.train_button.configure(state="normal", text="Start Training"); self.progress_bar.set(0)
        self.plot_training_loss(metrics); self.refresh_checkpoints()

    def handle_evaluation_completion(self, data):
        scores, ref_data, est_data = data; self.plot_evaluation_result(ref_data, est_data)
        self.eval_button.configure(state="normal"); self.tune_button.configure(state="normal"); self.tune_single_track_button.configure(state="normal"); self.eval_progress_bar.set(0)

    def handle_tuning_completion(self, data):
        best_thresh, best_f1 = data; self.threshold_entry.delete(0, 'end'); self.threshold_entry.insert(0, f"{best_thresh:.2f}")
        self.eval_button.configure(state="normal"); self.tune_button.configure(state="normal"); self.tune_single_track_button.configure(state="normal"); self.eval_progress_bar.set(0)

    def handle_hp_tuning_completion(self, data):
        history = data; self.plot_tuning_history(history)
        self.tune_run_button.configure(state="normal"); self.tune_progress_bar.set(0)

    # --- Plotting methods ---
    def plot_training_loss(self, metrics):
        if self.train_canvas_widget: self.train_canvas_widget.get_tk_widget().destroy()
        fig, ax = plt.subplots(figsize=(6, 4), facecolor="#2B2B2B"); ax.plot(metrics['train_loss'], label='Train Loss', color="#1F6AA5"); ax.plot(metrics['val_loss'], label='Val Loss', color="#FFA500")
        ax.set_title("Training History", color="white"); ax.set_xlabel("Epoch", color="white"); ax.set_ylabel("Loss", color="white"); ax.legend(); ax.grid(True, linestyle=':', color='gray'); ax.tick_params(axis='x', colors='white'); ax.tick_params(axis='y', colors='white'); fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.train_viz_frame); self.train_canvas_widget = canvas; canvas.draw(); canvas.get_tk_widget().pack(side=ctk.TOP, fill=ctk.BOTH, expand=1)

    def plot_evaluation_result(self, ref_data, est_data):
        if self.eval_canvas_widget: self.eval_canvas_widget.get_tk_widget().destroy()
        ref_times, ref_freqs = ref_data; est_times, est_freqs = est_data
        fig, ax = plt.subplots(figsize=(6, 4), facecolor="#2B2B2B")
        if ref_times.size > 0: ax.scatter(ref_times, ref_freqs, c='black', marker='.', s=50, label='Reference', zorder=2)
        if est_times.size > 0: ax.scatter(est_times, est_freqs, c='#e60000', marker='.', s=25, alpha=0.9, label='Prediction', zorder=3)
        ax.set_yscale('log'); ax.set_yticks([128, 256, 512, 1024], labels=['128', '256', '512', '1024']); ax.set_ylabel('Frequency (Hz)', color='white'); ax.set_xlabel('Time (s)', color='white')
        ax.set_title("Pitch Estimation Result", color="white"); ax.legend(); ax.grid(True, linestyle=':', color='gray'); ax.tick_params(axis='x', colors='white'); ax.tick_params(axis='y', colors='white')
        if ref_times.size > 0: ax.set_xlim(ref_times.min() - 1, ref_times.max() + 1); ax.set_ylim(bottom=60)
        fig.tight_layout(); canvas = FigureCanvasTkAgg(fig, master=self.eval_viz_frame); self.eval_canvas_widget = canvas; canvas.draw(); canvas.get_tk_widget().pack(side=ctk.TOP, fill=ctk.BOTH, expand=1)

    def plot_tuning_history(self, history):
        if self.tune_canvas_widget: self.tune_canvas_widget.get_tk_widget().destroy()
        fig, ax = plt.subplots(figsize=(6, 4), facecolor="#2B2B2B"); ax.plot(history['best_fitness'], label='Best Fitness', color="#1F6AA5", marker='o'); ax.plot(history['avg_fitness'], label='Average Fitness', color="#FFA500", linestyle='--')
        ax.set_title("Tuning Fitness History", color="white"); ax.set_xlabel("Generation", color="white"); ax.set_ylabel("F1-Score", color="white"); ax.legend(); ax.grid(True, linestyle=':', color='gray'); ax.tick_params(axis='x', colors='white'); ax.tick_params(axis='y', colors='white'); fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.tune_viz_frame); self.tune_canvas_widget = canvas; canvas.draw(); canvas.get_tk_widget().pack(side=ctk.TOP, fill=ctk.BOTH, expand=1)

    # --- UI update and callback methods ---
    def refresh_checkpoints(self):
        if not os.path.exists("checkpoints"): self.checkpoint_menu.configure(values=["None"]); return
        checkpoints = sorted([d for d in os.listdir("checkpoints") if os.path.isdir(os.path.join("checkpoints", d))], reverse=True)
        if not checkpoints: checkpoints = ["None"]
        self.checkpoint_menu.configure(values=checkpoints); self.on_checkpoint_selected(checkpoints[0])

    def on_checkpoint_selected(self, name):
        if name == "None": self.tune_single_track_button.configure(state="disabled"); self.eval_button.configure(state="disabled"); self.tune_button.configure(state="disabled"); self.eval_track_menu.configure(values=["None"]); return
        self.eval_button.configure(state="normal"); self.tune_button.configure(state="normal")
        self.tune_single_track_button.configure(state="normal")
        try:
            with open(os.path.join("checkpoints", name, "config.json"), 'r') as f: chk_config = json.load(f)
            eval_dataset_name = chk_config['data_params']['eval_dataset']
            track_ids = list(self.data_manager.track_groups[eval_dataset_name].keys())
            if not track_ids: track_ids = ["None"]
            self.eval_track_menu.configure(values=track_ids); self.eval_track_menu.set(random.choice(track_ids))
        except Exception as e: self.log(f"Error loading checkpoint config: {e}")

    def update_config_from_ui(self):
        """Update the main config dictionary from the UI widgets before a run."""
        # --- READ MODEL ARCHITECTURE ---
        self.config['model_params']['architecture_name'] = self.train_widgets['architecture_name'].get()

        # --- READ DATASET CONFIG (CORRECTED LOGIC) ---
        
        # This is the new logic to read from the checkboxes
        selected_train_datasets = [name for name, var in self.train_dataset_checkboxes.items() if var.get() == "on"]
        self.config['data_params']['train_datasets'] = selected_train_datasets # Note: plural 'datasets'
        
        # The old line that caused the error is no longer needed.
        # self.config['data_params']['train_dataset'] = self.train_widgets['train_dataset'].get() # REMOVED

        # The eval_dataset logic is correct as this widget still exists
        self.config['data_params']['eval_dataset'] = self.train_widgets['eval_dataset'].get()

        # --- Training Params ---
        try: 
            self.config['training_params']['learning_rate'] = float(self.train_widgets['learning_rate'].get())
        except (ValueError, TypeError): pass # Keep default if entry is empty/invalid
        try: 
            self.config['training_params']['num_epochs'] = int(self.train_widgets['num_epochs'].get())
        except (ValueError, TypeError): pass
        try: 
            self.config['training_params']['batch_size'] = int(self.train_widgets['batch_size'].get())
        except (ValueError, TypeError): pass
        
        # You might want to add entries for the new step-based training parameters here too
        # Example:
        # try:
        #     self.config['training_params']['steps_per_checkpoint'] = int(self.train_widgets['steps_per_checkpoint'].get())
        # except (ValueError, TypeError): pass
        
        self.log("Configuration updated from UI settings.")
    
    def update_tuning_config_from_ui(self):
        """Update the tuning_params in the config from the UI widgets."""
        try: self.config['tuning_params']['population_size'] = int(self.tuning_widgets['population_size'].get())
        except (ValueError, TypeError): pass
        try: self.config['tuning_params']['num_generations'] = int(self.tuning_widgets['num_generations'].get())
        except (ValueError, TypeError): pass
        try: self.config['tuning_params']['epochs_per_eval'] = int(self.tuning_widgets['epochs_per_eval'].get())
        except (ValueError, TypeError): pass
        
        self.log("Tuning configuration updated from UI settings.")

    def epoch_end_threadsafe(self, metrics): self.ui_queue.put(("epoch_complete", metrics))
    def log_threadsafe(self, message): self.after(0, self.log, message)
    def progress_threadsafe(self, current, total): self.after(0, lambda: self.progress_bar.set(current/total))
    def eval_progress_threadsafe(self, current, total): self.after(0, lambda: self.eval_progress_bar.set(current/total))
    def tune_progress_threadsafe(self, current, total): self.after(0, lambda: self.tune_progress_bar.set(current/total))
    def log(self, message): timestamp = datetime.now().strftime("%H:%M:%S"); self.console.insert("end", f"[{timestamp}] {message}\n"); self.console.see("end")

# -----------------------------------------------------------------------------
# SECTION 6: MAIN EXECUTION
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Deep Salience Studio")
    parser.add_argument("--dataset_root", type=str, default="./datasets", help="Path to the root directory containing dataset folders (e.g., CantoriaDataset_v1.0.0).")
    args = parser.parse_args()
    if not os.path.exists(args.dataset_root):
        print(f"Error: Dataset root path does not exist: {args.dataset_root}"); sys.exit(1)
    
    app = SalienceStudioApp(root_dir=args.dataset_root)
    app.mainloop()