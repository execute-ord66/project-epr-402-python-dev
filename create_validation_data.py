import os
import sys
import json
import argparse
import hashlib
from queue import Queue

import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

# -----------------------------------------------------------------------------
# SECTION 1: DATA HANDLING AND PROCESSING CLASSES
# (Copied from the main project to make this script standalone)
# -----------------------------------------------------------------------------

class DataProcessor:
    """
    Processes a group of tracks into a cached CQT/F0 representation.
    This class is essential for finding and processing audio/F0 files from
    different datasets (Cantoria, DCS, ChoralSynth) based on their unique
    path structures.
    """
    def __init__(self, config, root_dir, cache_dir, log_callback):
        self.config, self.root_dir, self.cache_dir = config, root_dir, cache_dir
        self.dp = self.config['data_params']
        self.n_bins = self.dp['n_octaves'] * self.dp['bins_per_octave']
        self.log = log_callback

    def get_dataset_folder(self, stem_name):
        """Finds the correct dataset subfolder for a given stem."""
        if "choralsynth" in stem_name.lower() or os.sep in stem_name:
            return "ChoralSynth"

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

    def process_and_cache_group(self, track_stems):
        """
        Processes and caches an entire group of stems as a single mix.
        This is a fallback in case the cache wasn't generated during training.
        """
        canonical_name = "_".join(sorted(track_stems))
        group_hash = hashlib.sha1(canonical_name.encode('utf-8')).hexdigest()
        cache_fname = f"{group_hash}.npz"
        cache_path = os.path.join(self.cache_dir, cache_fname)

        if os.path.exists(cache_path):
            return cache_path

        self.log(f"Cache not found for group. Processing & Caching: {', '.join(track_stems)}")

        # --- AUDIO MIXING ---
        max_len, all_y = 0, []
        valid_stems_for_f0 = []

        for stem in track_stems:
            is_choralsynth_style = os.sep in stem
            dataset_folder = self.get_dataset_folder(stem)
            if not dataset_folder:
                self.log(f"ERROR: Could not find dataset folder for stem {stem}")
                continue

            if is_choralsynth_style:
                audio_path = os.path.join(self.root_dir, dataset_folder, f"{stem}.wav")
            else:
                audio_path = os.path.join(self.root_dir, dataset_folder, "Audio", f"{stem}.wav")

            if not os.path.exists(audio_path):
                self.log(f"ERROR: File not found at resolved path: {audio_path}")
                continue
                
            y, _ = librosa.load(audio_path, sr=self.dp['sr'])
            all_y.append(y)
            valid_stems_for_f0.append(stem)
            if len(y) > max_len: max_len = len(y)
        
        if not all_y:
            self.log(f"Warning: No valid audio files found for group. Skipping cache generation.")
            return None

        y_mix = np.sum([np.pad(y, (0, max_len - len(y))) for y in all_y], axis=0)
        if np.max(np.abs(y_mix)) > 0:
            y_mix /= np.max(np.abs(y_mix))

        # --- HCQT CALCULATION ---
        cqt_list = [librosa.cqt(y_mix, sr=self.dp['sr'], hop_length=self.dp['hop_length'], fmin=self.dp['fmin'] * h, n_bins=self.n_bins, bins_per_octave=self.dp['bins_per_octave']) for h in self.dp['harmonics']]
        min_time = min(c.shape[1] for c in cqt_list)
        hcqt = np.stack([c[:, :min_time] for c in cqt_list])
        log_hcqt = (1.0/80.0) * librosa.amplitude_to_db(np.abs(hcqt), ref=np.max) + 1.0

        # --- GROUND TRUTH F0 MAP (simplified for this script's purpose) ---
        n_frames = log_hcqt.shape[2]
        f0_map = np.zeros((self.n_bins, n_frames), dtype=np.float32) # Dummy F0 map
        
        np.savez_compressed(cache_path, log_hcqt=log_hcqt, f0_map=f0_map)
        return cache_path

class PatchDataset(Dataset):
    """
    Creates patches from pre-computed and cached CQT/F0 maps. This class
    will read the validation track groups from the config and create the
    input patches that need to be saved as .npy files.
    """
    def __init__(self, track_groups, root_dir, cache_dir, config, log_callback):
        self.config = config
        self.log = log_callback
        self.root_dir = root_dir
        self.cache_dir = cache_dir

        self.tp, self.dp = self.config['training_params'], self.config['data_params']
        self.patch_width_frames = self.tp['patch_width']
        self.step_size = int(self.patch_width_frames * (1 - self.tp.get('patch_overlap', 0.5)))

        processor = DataProcessor(config, root_dir, cache_dir, log_callback)
        
        self.index = []
        self.cache_data = []

        self.log(f"Loading cached data for {len(track_groups)} validation groups...")
        for group in tqdm(track_groups, desc="Loading Cache", leave=False):           
            canonical_name = "_".join(sorted(group))
            group_hash = hashlib.sha1(canonical_name.encode('utf-8')).hexdigest()
            cache_fname = f"{group_hash}.npz"
            cache_path = os.path.join(self.cache_dir, cache_fname)

            if not os.path.exists(cache_path):
                self.log(f"Cache file not found for group '{canonical_name}'. Attempting to generate it now...")
                processor.process_and_cache_group(group)
            
            try:
                data = np.load(cache_path)
                self.cache_data.append({
                    'log_hcqt': data['log_hcqt'],
                    'f0_map': data['f0_map']
                })
                n_frames = data['log_hcqt'].shape[2]
                current_data_idx = len(self.cache_data) - 1
                for start in range(0, n_frames - self.patch_width_frames + 1, self.step_size):
                    self.index.append((current_data_idx, start))
            except Exception as e:
                self.log(f"ERROR: Could not load or process cache file {cache_path}. Error: {e}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        data_idx, start_frame = self.index[idx]
        hcqt_full = self.cache_data[data_idx]['log_hcqt']
        f0_map_full = self.cache_data[data_idx]['f0_map']

        end_frame = start_frame + self.patch_width_frames
        cqt_patch = hcqt_full[:, :, start_frame:end_frame]
        f0_patch = f0_map_full[:, start_frame:end_frame]
        
        return torch.from_numpy(cqt_patch.copy()).float(), torch.from_numpy(f0_patch.copy()).unsqueeze(0).float()

# -----------------------------------------------------------------------------
# SECTION 2: MAIN SCRIPT LOGIC
# -----------------------------------------------------------------------------

def main():
    """
    Main function to generate .npy calibration files from a model's config.
    """
    parser = argparse.ArgumentParser(
        description="Generate .npy calibration files for NCNN from a model's validation set.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        required=True,
        help="Path to the model's checkpoint directory (e.g., './checkpoints/some_hash_1234')."
    )
    parser.add_argument(
        '--dataset-root',
        type=str,
        default="./datasets",
        help="Path to the root directory containing dataset folders (e.g., Cantoria, DCS)."
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default="./calibration_data",
        help="Directory where the generated .npy files and list will be saved."
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        default="./cache",
        help="Directory where pre-processed data is cached."
    )
    args = parser.parse_args()

    # --- 1. Load Configuration ---
    config_path = os.path.join(args.checkpoint_dir, 'config.json')
    if not os.path.exists(config_path):
        print(f"FATAL: config.json not found in checkpoint directory: {args.checkpoint_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)

    # --- 2. Get Validation Set Information ---
    try:
        val_groups = config['data_params']['val_groups']
        if not val_groups:
            raise KeyError
    except KeyError:
        print(f"FATAL: 'val_groups' not found or is empty in the config's 'data_params'.", file=sys.stderr)
        print("This key is essential for identifying the validation files.", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(val_groups)} validation groups defined in config.")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output will be saved to: {args.output_dir}")

    # --- 3. Create the Validation Patch Dataset ---
    print("\nInitializing validation dataset... This may take a moment as it loads cached data.")
    val_dataset = PatchDataset(
        track_groups=val_groups,
        root_dir=args.dataset_root,
        cache_dir=args.cache_dir,
        config=config,
        log_callback=print
    )

    if len(val_dataset) == 0:
        print("\nFATAL: The validation dataset is empty. No .npy files can be generated.", file=sys.stderr)
        print("This could be due to missing audio files, cache issues, or tracks being too short.", file=sys.stderr)
        sys.exit(1)

    print(f"Successfully created validation dataset with {len(val_dataset)} patches.")

    # --- 4. Generate .npy files and the calibration list ---
    list_filename = "calibration_list.txt"
    list_filepath = os.path.join(args.output_dir, list_filename)

    print(f"\nGenerating {len(val_dataset)} .npy files and '{list_filename}'...")
    with open(list_filepath, 'w') as list_file:
        for i in tqdm(range(len(val_dataset)), desc="Exporting Patches"):
            # Get the pre-processed input patch tensor
            cqt_patch_tensor, _ = val_dataset[i]
            
            # Convert to a NumPy array
            numpy_patch = cqt_patch_tensor.cpu().numpy()

            # Define file path and save
            npy_filename = f"val_patch_{i:06d}.npy"
            npy_filepath = os.path.join(args.output_dir, npy_filename)
            np.save(npy_filepath, numpy_patch)

            # Write the relative filename to the list file
            list_file.write(f"{npy_filename}\n")

    print("\n--- Process Complete! ---")
    print(f"Successfully generated {len(val_dataset)} .npy files in '{args.output_dir}'.")
    print(f"A file list for calibration has been saved to: '{list_filepath}'")
    
    # Provide next-step instructions
    dp = config['data_params']
    mp = config['model_params']
    input_shape = f"{mp.get('input_channels', len(dp['harmonics']))},{dp['n_octaves'] * dp['bins_per_octave']},{config['training_params']['patch_width']}"
    
    print("\nNext Step (Example usage with ncnn2table):")
    print(f"cd {os.path.abspath(args.output_dir)}")
    print(f"path/to/ncnn2table model-opt.param model-opt.bin {list_filename} model.table shape=[{input_shape}]")


if __name__ == '__main__':
    main()