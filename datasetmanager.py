# datasetmanager.py
from collections import defaultdict
import os
import random
import hashlib

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

# Local imports from your other files
from augmentations import SalienceAugment
from dataprocessor import *
# The previous training loader relied on ``DynamicMixingDataset`` which used a
# complex ``FileCache`` for storing pre-processed audio.  This proved to be
# slow and memory hungry.  We now use ``AugmentedSalienceDataset`` which
# performs lightweight on-the-fly loading and augmentation inspired by the
# efficient pipeline in ZFTurbo's codebase.
from datasets.augmentation_dataset import AugmentedSalienceDataset


def NOOP_LOG(*args, **kwargs):
    pass


class PatchDataset(Dataset):
    # This class remains UNCHANGED.
    def __init__(
        self, track_groups, root_dir, cache_dir, config, is_train, log_callback=print
    ):
        self.config, self.is_train = config, is_train
        self.log = log_callback or NOOP_LOG
        self.root_dir = root_dir
        self.cache_dir = cache_dir

        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            # We are in the main process. Use the provided logger.
            self.log = log_callback
        else:
            # We are in a worker process. Workers should be silent.
            # Replace the logger with a dummy function that does nothing.
            self.log = NOOP_LOG

        self.tp, self.dp = self.config["training_params"], self.config["data_params"]
        self.patch_width_frames = self.tp.get(
            "patch_width", 64
        )  # Use .get for safety with new configs
        self.step_size = int(
            self.patch_width_frames * (1 - self.tp.get("patch_overlap", 0.5))
        )

        self.augmenter = None
        self.mixup_alpha = 0.0
        if self.is_train:
            self.augmenter = SalienceAugment()
            self.mixup_alpha = self.tp.get("mixup_alpha", 0.4)

        processor = DataProcessor(config, root_dir, cache_dir, log_callback)

        self.index = []
        self.cache_data = []

        for group in tqdm(track_groups, desc="Checking and loading cache", leave=False):
            canonical_name = "_".join(sorted(group))
            group_hash = hashlib.sha1(canonical_name.encode("utf-8")).hexdigest()
            cache_fname = f"{group_hash}.npz"
            cache_path = os.path.join(self.cache_dir, cache_fname)

            if not os.path.exists(cache_path):
                processor.process_and_cache_group(group)

            try:
                data = np.load(cache_path)
                self.cache_data.append(
                    {"log_hcqt": data["log_hcqt"], "f0_map": data["f0_map"]}
                )
                n_frames = data["log_hcqt"].shape[2]
                current_data_idx = len(self.cache_data) - 1
                for start in range(
                    0, n_frames - self.patch_width_frames + 1, self.step_size
                ):
                    self.index.append((current_data_idx, start))
            except Exception as e:
                self.log(
                    f"ERROR: Could not load or process cache file {cache_path}. Error: {e}"
                )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        # This method's logic is UNCHANGED.
        data_idx, start_frame = self.index[idx]
        hcqt_full = self.cache_data[data_idx]["log_hcqt"]
        f0_map_full = self.cache_data[data_idx]["f0_map"]
        end_frame = start_frame + self.patch_width_frames
        cqt_patch = hcqt_full[:, :, start_frame:end_frame]
        f0_patch = f0_map_full[:, start_frame:end_frame]
        cqt_patch = torch.from_numpy(cqt_patch.copy()).float()
        f0_patch = torch.from_numpy(f0_patch.copy()).unsqueeze(0).float()
        if self.is_train:
            if self.mixup_alpha > 0 and random.random() < 0.5:
                mix_idx = random.randint(0, len(self.index) - 1)
                mix_data_idx, mix_start_frame = self.index[mix_idx]
                mix_hcqt_full = self.cache_data[mix_data_idx]["log_hcqt"]
                mix_f0_map_full = self.cache_data[mix_data_idx]["f0_map"]
                mix_end_frame = mix_start_frame + self.patch_width_frames
                mix_cqt_patch = mix_hcqt_full[:, :, mix_start_frame:mix_end_frame]
                mix_f0_patch = mix_f0_map_full[:, mix_start_frame:mix_end_frame]
                mix_cqt_patch = torch.from_numpy(mix_cqt_patch.copy()).float()
                mix_f0_patch = (
                    torch.from_numpy(mix_f0_patch.copy()).unsqueeze(0).float()
                )
                lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                cqt_patch = lam * cqt_patch + (1 - lam) * mix_cqt_patch
                f0_patch = lam * f0_patch + (1 - lam) * mix_f0_patch
            if self.augmenter:
                cqt_patch = self.augmenter(cqt_patch)
        return cqt_patch, f0_patch


class DatasetManager:
    def __init__(self, root_dir, cache_dir, log_callback=print):
        self.root_dir, self.cache_dir = root_dir, cache_dir
        self.log = log_callback or NOOP_LOG
        self.track_groups = {}  # Populated by run_discovery
        self.all_stems = defaultdict(list)  # Populated by run_discovery
        self.device = "cpu"  # Will be set by the main app

    def generate_train_val_split(self, config):
        """
        Creates a persistent, stratified train/validation split from the user-selected
        datasets and saves the split into the config dictionary.
        """
        self.log(
            "Generating a new, persistent, and proportional train/validation split."
        )
        tp = config["training_params"]
        val_split_ratio = tp.get("val_split_ratio", 0.15)

        final_train_groups, final_val_groups = [], []

        train_dataset_names = config["data_params"].get("train_datasets", [])
        for name in train_dataset_names:
            dataset_specific_groups = list(self.track_groups.get(name, {}).values())
            if not dataset_specific_groups:
                continue

            random.shuffle(dataset_specific_groups)

            num_val = int(len(dataset_specific_groups) * val_split_ratio)
            # Ensure at least one validation track if possible
            if (
                len(dataset_specific_groups) > 1
                and num_val == 0
                and val_split_ratio > 0
            ):
                num_val = 1

            val_subset, train_subset = (
                dataset_specific_groups[:num_val],
                dataset_specific_groups[num_val:],
            )
            final_val_groups.extend(val_subset)
            final_train_groups.extend(train_subset)
            self.log(
                f"- Split for '{name}': {len(train_subset)} train, {len(val_subset)} val."
            )

        config["data_params"]["train_groups"] = final_train_groups
        config["data_params"]["val_groups"] = final_val_groups
        self.log(
            f"Total split: {len(final_train_groups)} train, {len(final_val_groups)} val."
        )
        return config

    def _get_dataset_folder(self, stem_name):
        """Helper to find the dataset subfolder for a Cantoria or DCS stem."""
        # This logic remains the same as provided.
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
        return ""

    def _discover_datasets(self):
        self.log("Scanning for datasets using hybrid discovery strategy...")
        self.log(
            f"--- Root directory being scanned: '{os.path.abspath(self.root_dir)}'"
        )
        all_groups = {"Cantoria": {}, "DCS": {}, "ChoralSynth": {}}

        try:
            dataset_folders = [
                d
                for d in os.listdir(self.root_dir)
                if os.path.isdir(os.path.join(self.root_dir, d))
            ]
            if not dataset_folders:
                self.log(
                    "--- WARNING: No subdirectories found in the root directory. Make sure dataset folders are directly inside the root."
                )
        except FileNotFoundError:
            self.log(
                f"--- FATAL ERROR: Dataset root directory not found at {self.root_dir}"
            )
            return {}, defaultdict(list)

        self.log(
            f"--- Found {len(dataset_folders)} potential dataset folders: {dataset_folders}"
        )

        for dataset_folder in dataset_folders:
            base_path = os.path.join(self.root_dir, dataset_folder)
            self.log(f"\n--- Checking folder: '{dataset_folder}'")

            # --- Strategy 1: For Cantoria ---
            if "cantoria" in dataset_folder.lower():
                self.log("-> Applying Cantoria discovery logic...")
                audio_dir = os.path.join(base_path, "Audio")
                crepe_dir = os.path.join(base_path, "F0_crepe")
                pyin_dir = os.path.join(base_path, "F0_pyin")

                if not os.path.exists(audio_dir):
                    self.log("  -> SKIPPING: 'Audio' subfolder not found.")
                    continue
                if not os.path.exists(crepe_dir):
                    self.log("  -> SKIPPING: 'F0_crepe' subfolder not found.")
                    continue
                if not os.path.exists(pyin_dir):
                    self.log("  -> SKIPPING: 'F0_pyin' subfolder not found.")
                    continue

                self.log("  -> Found required subfolders (Audio, F0_crepe, F0_pyin).")
                voice_parts = ["S", "A", "T", "B"]
                audio_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
                track_bases = set(
                    f.rsplit("_", 1)[0]
                    for f in audio_files
                    if f.startswith("Cantoria_") and len(f.split("_")) == 3
                )
                self.log(
                    f"  -> Found {len(track_bases)} potential track bases. Verifying completeness..."
                )

                for base in sorted(list(track_bases)):
                    # Check that all required audio and F0 files exist for this group
                    if all(
                        os.path.exists(os.path.join(audio_dir, f"{base}_{p}.wav"))
                        and os.path.exists(os.path.join(crepe_dir, f"{base}_{p}.csv"))
                        and os.path.exists(os.path.join(pyin_dir, f"{base}_{p}.csv"))
                        for p in voice_parts
                    ):
                        all_groups["Cantoria"][base] = [
                            f"{base}_{p}" for p in voice_parts
                        ]

                self.log(
                    f"  -> SUCCESS: Found {len(all_groups['Cantoria'])} complete Cantoria groups."
                )

            # --- Strategy 2: For DCS ---
            elif (
                "dcs" in dataset_folder.lower() or "dagstuhl" in dataset_folder.lower()
            ):
                self.log("-> Applying DCS discovery logic...")
                audio_dir = os.path.join(base_path, "Audio")
                crepe_dir = os.path.join(base_path, "F0_crepe")
                pyin_dir = os.path.join(base_path, "F0_pyin")

                if not os.path.exists(audio_dir):
                    self.log("  -> SKIPPING: 'Audio' subfolder not found.")
                    continue
                if not os.path.exists(crepe_dir):
                    self.log("  -> SKIPPING: 'F0_crepe' subfolder not found.")
                    continue
                if not os.path.exists(pyin_dir):
                    self.log("  -> SKIPPING: 'F0_pyin' subfolder not found.")
                    continue

                self.log("  -> Found required subfolders (Audio, F0_crepe, F0_pyin).")
                potential_groups = defaultdict(list)
                for filename in os.listdir(audio_dir):
                    if not filename.endswith(".wav"):
                        continue
                    base_name = filename.rsplit(".", 1)[0]
                    parts = base_name.split("_")
                    mic_type = parts[-1]
                    if mic_type.upper() in {"DYN", "LRX", "HSM"}:
                        group_id = "_".join(parts[:-2] + [mic_type])
                        potential_groups[group_id].append(base_name)

                self.log(
                    f"  -> Found {len(potential_groups)} potential groups. Verifying completeness..."
                )
                for group_id, stems in potential_groups.items():
                    if all(
                        os.path.exists(os.path.join(audio_dir, f"{s}.wav"))
                        and os.path.exists(os.path.join(crepe_dir, f"{s}.csv"))
                        and os.path.exists(os.path.join(pyin_dir, f"{s}.csv"))
                        for s in stems
                    ):
                        all_groups["DCS"][group_id] = stems

                self.log(
                    f"  -> SUCCESS: Found {len(all_groups['DCS'])} complete DCS groups."
                )

            # --- Strategy 3: For ChoralSynth ---
            elif "choralsynth" in dataset_folder.lower():
                self.log("-> Applying ChoralSynth discovery logic...")
                track_folders = [
                    d
                    for d in os.listdir(base_path)
                    if os.path.isdir(os.path.join(base_path, d))
                ]
                self.log(f"  -> Found {len(track_folders)} potential track subfolders.")

                for track_folder in track_folders:
                    voices_dir = os.path.join(base_path, track_folder, "voices")
                    if not os.path.exists(voices_dir):
                        self.log(
                            f"    -> SKIPPING track '{track_folder}': 'voices' subfolder not found."
                        )
                        continue

                    stems_in_group = []
                    for filename in os.listdir(voices_dir):
                        if filename.endswith(".wav"):
                            stem_name = filename[:-4]
                            # The unique identifier for a ChoralSynth stem includes its track folder path
                            full_stem_path_prefix = os.path.join(
                                track_folder, "voices", stem_name
                            )
                            f0_path = os.path.join(
                                base_path, f"{full_stem_path_prefix}.f0.csv"
                            )
                            if os.path.exists(f0_path):
                                stems_in_group.append(full_stem_path_prefix)

                    if stems_in_group:
                        all_groups["ChoralSynth"][track_folder] = stems_in_group

                self.log(
                    f"  -> SUCCESS: Found {len(all_groups['ChoralSynth'])} complete ChoralSynth groups."
                )

            else:
                self.log(
                    f"-> No specific logic found for '{dataset_folder}'. Skipping."
                )

        # --- Final Stem Aggregation ---
        self.log("\n--- Aggregating all found stems for training pool ---")
        flat_stems = defaultdict(list)
        for dataset_name, groups in all_groups.items():
            if not groups:
                continue
            self.log(f"-> Processing {len(groups)} groups from '{dataset_name}'...")
            for group_id, stems in groups.items():
                for stem_name in stems:
                    is_choralsynth_style = os.sep in stem_name
                    if is_choralsynth_style:
                        # For ChoralSynth, paths are relative to the dataset root
                        audio_path = os.path.join(
                            self.root_dir, "ChoralSynth", f"{stem_name}.wav"
                        )
                        f0_path = os.path.join(
                            self.root_dir, "ChoralSynth", f"{stem_name}.f0.csv"
                        )
                    else:
                        # For Cantoria/DCS, find the specific dataset subfolder
                        dataset_folder = self._get_dataset_folder(stem_name)
                        if not dataset_folder:
                            continue
                        audio_path = os.path.join(
                            self.root_dir, dataset_folder, "Audio", f"{stem_name}.wav"
                        )
                        # For Cantoria/DCS we use crepe as the primary F0 for training data
                        f0_path = os.path.join(
                            self.root_dir,
                            dataset_folder,
                            "F0_crepe",
                            f"{stem_name}.csv",
                        )

                    # Final check to ensure both audio and F0 files actually exist before adding to the pool
                    if os.path.exists(audio_path) and os.path.exists(f0_path):
                        flat_stems[dataset_name].append(
                            {
                                "audio_path": audio_path,
                                "f0_path": f0_path,
                                "stem_name": stem_name,
                            }
                        )

        for name, stems in flat_stems.items():
            self.log(f"-> Found {len(stems)} valid stems for '{name}'.")

        return all_groups, flat_stems

    def run_discovery(self):
        """
        Scans the root directory, discovers all compatible datasets (Cantoria, DCS, ChoralSynth),
        and populates the instance variables `self.track_groups` (for evaluation grouping)
        and `self.all_stems` (for the dynamic training pool).
        """
        # Call the main private method that contains all the detailed discovery logic
        discovered_groups, discovered_stems = self._discover_datasets()

        # Store the results in the instance variables for other methods to use
        self.track_groups = discovered_groups
        self.all_stems = discovered_stems

        # Provide a clear, summary log message to the user
        num_cantoria = len(self.track_groups.get("Cantoria", {}))
        num_dcs = len(self.track_groups.get("DCS", {}))
        num_choralsynth = len(self.track_groups.get("ChoralSynth", {}))

        self.log(
            f"Discovery complete. Found {num_cantoria} Cantoria, {num_dcs} DCS, and {num_choralsynth} ChoralSynth groups."
        )

    def get_dataloaders(self, config):
        """
        Creates and returns the appropriate training and validation dataloaders
        based on the persistent split stored in the config.
        """
        # --- This method is now corrected ---

        def _worker_init_fn(worker_id):
            seed = torch.initial_seed() % 2**32
            np.random.seed(seed)
            random.seed(seed)

        def _collate_raw(batch):
            audios = np.stack([item[0] for item in batch], axis=0)
            f0_lists = [item[1] for item in batch]
            return torch.from_numpy(audios), f0_lists

        train_groups = config["data_params"].get("train_groups")
        val_groups = config["data_params"].get("val_groups")

        if train_groups is None or val_groups is None:
            self.log(
                "FATAL: Config is missing the train/val split. Cannot create dataloaders."
            )
            return None, None

        # ---------------- Validation (cache-based PatchDataset) ----------------
        val_loader = None
        if val_groups:
            self.log(
                f"Preparing validation loader with {len(val_groups)} groups (from cache)."
            )
            val_dataset = PatchDataset(
                val_groups,
                self.root_dir,
                self.cache_dir,
                config,
                is_train=False,
                log_callback=None,
            )
            if len(val_dataset) > 0:
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=config.get("evaluation_params", {}).get(
                        "eval_batch_size", 8
                    ),
                    shuffle=False,
                    num_workers=0,
                    # num_workers=max(1, os.cpu_count() // 4),
                    # pin_memory=True,
                    # prefetch_factor=2,  # (remove/omit when num_workers=0)
                    # persistent_workers=True,  # (must be False when num_workers=0)
                    drop_last=False,
                    worker_init_fn=_worker_init_fn,
                )

        # ---------------- Training (on-the-fly augmented dataset) ----------------
        if not train_groups:
            self.log("ERROR: No training groups found in the split.")
            return None, val_loader

        training_stem_names = {stem for group in train_groups for stem in group}
        train_stems_with_info = []
        for dataset_name in config["data_params"]["train_datasets"]:
            for stem_info_dict in self.all_stems.get(dataset_name, []):
                if stem_info_dict["stem_name"] in training_stem_names:
                    train_stems_with_info.append(stem_info_dict)

        if not train_stems_with_info:
            self.log("ERROR: Could not find detailed info for any training stems.")
            return None, val_loader

        self.log(
            f"Creating training loader with {len(train_stems_with_info)} stems using on-the-fly augmentation."
        )

        train_dataset = AugmentedSalienceDataset(
            train_stems_with_info, config, log_callback=print
        )

        tp = config["training_params"]

        train_loader = DataLoader(
            train_dataset,
            batch_size=tp.get("batch_size", 8),
            shuffle=False,
            num_workers=0,
            # num_workers=tp.get("num_workers", max(1, os.cpu_count() // 2)),
            # pin_memory=True,
            # prefetch_factor=4,
            # persistent_workers=True,
            drop_last=True,
            worker_init_fn=_worker_init_fn,
            collate_fn=_collate_raw,
        )

        return train_loader, val_loader
