import os
import sys
import json
import random
import threading
import shutil
from datetime import datetime
import argparse
from queue import Queue, Empty

import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.profiler
import contextlib
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tqdm import tqdm
from scipy.signal import find_peaks
import mir_eval
import customtkinter as ctk
import hashlib

from augmentations import *
from datasetmanager import DatasetManager

from helpers import *

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# Use the 'Agg' backend for Matplotlib to make it thread-safe for Tkinter
matplotlib.use("Agg")

# -----------------------------------------------------------------------------
# SECTION 1: CORE CONFIGURATION & DEFAULT SETTINGS
# -----------------------------------------------------------------------------


def get_default_config():
    """Returns a dictionary with the default configuration for the entire project."""
    return {
        "run_id": None,
        "data_params": {
            "sr": 22050,
            "hop_length": 256,
            "fmin": 32.703,
            "harmonics": [1, 2, 3, 4, 5],
            "bins_per_octave": 60,
            "n_octaves": 6,
            "gaussian_sigma": 1.0,
            "train_dataset": "Cantoria",
        },
        "model_params": {
            "architecture_name": "SalienceNetV3",
            "input_channels": 5,  # Should match len(harmonics)
            "layers": [
                {"type": "conv_in", "filters": 32, "kernel": 5},
                {"type": "conv", "filters": 32, "kernel": 5},
                {"type": "conv", "filters": 32, "kernel": 5},
                {"type": "conv", "filters": 32, "kernel": (69, 1)},
                {"type": "conv_out", "filters": 1, "kernel": 1},
            ],
            "activation": "GELU",
            "rnn_hidden_size": 48,
            # --- Parameters specific to the BSRoformerForSalience model ---
            "dim": 200,  # Internal dimension of the transformer
            "depth": 4,  # Number of axial transformer blocks
            "heads": 4,  # Number of attention heads
            "dim_head": 32,  # Dimension of each attention head
            "time_transformer_depth": 1,  # Layers within each time transformer
            "freq_transformer_depth": 1,  # Layers within each frequency transformer
            "num_bands": 1,  # How many bands to split the CQT into.
            # (n_bins * input_channels) must be divisible by this.
            # (360 * 5) = 1800. 1800 is divisible by 6.
            # --- Parameters for SpecTNTForSalience (Lite version for 6 hours of data) ---
            "fe_out_channels": 64,  # Channels after the ResNet frontend
            "fe_freq_pooling": (2, 2, 2),  # Downsample frequency by 2*2*2 = 8x
            "fe_time_pooling": (2, 2, 1),  # Downsample time by 2*2*1 = 4x
            "spectral_dmodel": 128,  # Internal dimension of spectral transformer
            "spectral_nheads": 4,  # Attention heads
            "spectral_dimff": 256,  # Feed-forward layer size
            "temporal_dmodel": 128,  # Internal dimension of temporal transformer
            "temporal_nheads": 4,  # Attention heads
            "temporal_dimff": 256,  # Feed-forward layer size
            "embed_dim": 128,  # Shared embedding dimension
            "n_blocks": 4,  # Number of SpecTNT blocks
            "dropout": 0.1,
        },
        "training_params": {
            "learning_rate": 8e-4,
            "batch_size": 22,
            "num_epochs": 200,
            "optimizer": "AdamW",
            "patch_width": 64,
            "patch_overlap": 0.5,
            "val_split_ratio": 0.1,
            "weight_decay": 1e-3,
            "mixup_alpha": 0.4,
            "steps_per_epoch": 500,  # Number of batches to generate per "epoch"
            "segment_duration_sec": 2.0,  # Duration of audio segments in the pool (in seconds)
            "pool_size": 128,  # Number of stem segments to keep in the dynamic pool
            "patch_width": 64,
            "stems_per_mix": 4,  # Number of stems to mix for each training example
            "loudness_threshold_db": -50.0,  # Min loudness for a segment to be added to the pool
            "val_split_ratio": 0.15,
        },
        "evaluation_params": {"eval_batch_size": 8, "peak_threshold": 0.3},
        "tuning_params": {
            "population_size": 8,
            "num_generations": 5,
            "epochs_per_eval": 5,
            "mutation_rate": 0.2,
            "crossover_rate": 0.7,
            # Fitness function weights
            "fitness_performance_weight": 0.8,  # w1
            "fitness_efficiency_weight": 0.3,  # w2
            "search_space": {
                "layer_filters": {"type": "choice", "choices": [8, 12, 16, 20, 24, 32]},
                "learning_rate": {"type": "log_uniform", "range": [1e-5, 1e-1]},
                "gaussian_sigma": {
                    "type": "uniform",
                    "choices": [1.0, 1.5, 2.0, 2.5, 3.0],
                },
            },
        },
    }


def get_config_hash(config):
    """Generates a stable SHA-256 hash for a configuration dictionary."""
    # Create a deep copy to avoid modifying the original config
    import copy

    config_copy = copy.deepcopy(config)

    # Remove volatile keys that shouldn't affect the hash
    config_copy.pop("run_id", None)

    # Convert the dictionary to a sorted JSON string to ensure consistency
    config_string = json.dumps(config_copy, sort_keys=True, indent=None)

    # Calculate and return the SHA-256 hash
    # --- MODIFICATION: Shorten the hash to 8 characters ---
    return hashlib.sha256(config_string.encode("utf-8")).hexdigest()[:8]


# -----------------------------------------------------------------------------
# SECTION 2: MODELS, LOSS, & METRICS
# -----------------------------------------------------------------------------


class ParabolicCone(nn.Module):
    def forward(self, input):
        return input * (2 - input)


class Cone(nn.Module):
    def forward(self, input):
        return 1 - torch.abs(input - 1)


ACTIVATIONS = {
    "GELU": nn.GELU,
    "GEGLU": lambda: GEGLU(),
    "SwiGLU": lambda: SwiGLU(),
    "ReLU": lambda: nn.ReLU(inplace=True),
    "SiLU": lambda: nn.SiLU(inplace=True),
    "ParabolicCone": ParabolicCone,
    "Cone": ParabolicCone,
}


class SalienceCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_cfg = config["model_params"]
        layers = []
        in_channels = model_cfg["input_channels"]

        activation_name = model_cfg.get("activation", "GELU")
        activation_fn = ACTIVATIONS[activation_name]()

        for i, layer_cfg in enumerate(model_cfg["layers"]):
            out_channels = layer_cfg["filters"]
            kernel = layer_cfg["kernel"]
            padding = layer_cfg.get("padding", "same")

            # --- THE FIX PART 2: Double channels for GEGLU ---
            conv_out_channels = (
                out_channels * 2
                if activation_name == "GEGLU"
                and layer_cfg["type"] not in ["conv_out", "conv_in"]
                else out_channels
            )

            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=conv_out_channels,
                    kernel_size=kernel,
                    padding=padding,
                )
            )

            if layer_cfg["type"] not in ["conv_out", "conv_in"]:
                # The BatchNorm must also be on the doubled channel count, *before* GEGLU halves it
                layers.append(nn.BatchNorm2d(conv_out_channels))
                layers.append(activation_fn)

            in_channels = out_channels

        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class SalienceCNNLogits(nn.Module):
    def __init__(self, config, version=3):
        super().__init__()
        model_cfg = config["model_params"]
        data_cfg = config["data_params"]
        training_cfg = config["training_params"]
        layers = []
        in_channels = len(data_cfg["harmonics"])

        activation_name = model_cfg.get("activation", "GELU")
        activation_fn = ACTIVATIONS[activation_name]()

        for i, layer_cfg in enumerate(model_cfg["layers"]):
            out_channels = layer_cfg["filters"]
            kernel = layer_cfg["kernel"]
            padding = layer_cfg.get("padding", "same")

            # --- THE FIX PART 2: Double conv output channels if using GEGLU ---
            # This applies to intermediate layers, not the final output layer.
            conv_out_channels = (
                out_channels * 2
                if activation_name == "GEGLU"
                and layer_cfg["type"] not in ["conv_out", "conv_in", "conv_n"]
                else out_channels
            )

            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=conv_out_channels,
                    kernel_size=kernel,
                    padding=padding,
                )
            )

            if layer_cfg["type"] not in ["conv_out", "conv_in", "conv_n"]:
                # BatchNorm must operate on the doubled channels before they are halved by GEGLU
                layers.append(nn.BatchNorm2d(conv_out_channels))
                layers.append(activation_fn)

            # The next layer's input channel count is the *halved* count after GEGLU
            in_channels = out_channels

        self.network = nn.Sequential(*layers)

        # This dynamic calculation logic remains the same
        with torch.no_grad():
            dummy_input = torch.randn(
                1,
                len(data_cfg["harmonics"]),
                data_cfg["bins_per_octave"] * data_cfg["n_octaves"],
                training_cfg["patch_width"],
            )
            if version == 4:
                cnn_feature_extractor = self.network
            else:
                cnn_feature_extractor = self.network
            cnn_out_shape = cnn_feature_extractor(dummy_input).shape
            self.gru_input_size = cnn_out_shape[1] * cnn_out_shape[2]

        self.gru = None
        self.linear_out = None
        if version == 4:
            # GRU setup logic...
            self.gru = nn.GRU(
                input_size=self.gru_input_size,
                hidden_size=model_cfg["rnn_hidden_size"],
                num_layers=1,
                bidirectional=False,
                batch_first=True,
            )
            self.linear_out = nn.Linear(
                model_cfg["rnn_hidden_size"],
                data_cfg["bins_per_octave"] * data_cfg["n_octaves"],
            )

    def forward(self, x):
        # Forward pass logic...
        x = self.network(x)
        if self.gru:
            x = x.permute(0, 3, 1, 2)
            N, W, C, H = x.shape
            x = x.reshape(N, W, C * H)
            x, _ = self.gru(x)
            x = self.linear_out(x)
            x = x.permute(0, 2, 1).unsqueeze(1)
        return x


def bkld_loss(y_pred, y_true, eps=1e-7):
    y_pred = torch.clamp(y_pred, eps, 1 - eps)
    y_true = torch.clamp(y_true, eps, 1 - eps)
    bce = -y_true * torch.log(y_pred) - (1 - y_true) * torch.log(1 - y_pred)
    return bce.mean(dim=(1, 2)).mean()


# Import necessary components from the bsroformer git
from bs_roformer.bs_roformer import Transformer, BandSplit, RMSNorm
from bs_roformer.attend import Attend
from einops import rearrange, pack, unpack, reduce


class BSRoformerForSalience(nn.Module):
    """
    An adapter for the BS-Roformer architecture to predict salience maps from HCQT inputs.
    """

    def __init__(self, config):
        super().__init__()
        model_params = config["model_params"]
        data_params = config["data_params"]

        # --- BS-Roformer Specific Hyperparameters ---
        dim = model_params.get("dim", 192)
        depth = model_params.get("depth", 6)
        heads = model_params.get("heads", 8)
        dim_head = model_params.get("dim_head", 64)
        time_transformer_depth = model_params.get("time_transformer_depth", 2)
        freq_transformer_depth = model_params.get("freq_transformer_depth", 2)
        num_bands = model_params.get(
            "num_bands", 6
        )  # How many bands to split the CQT into

        # --- Input/Output Dimensions ---
        input_channels = model_params["input_channels"]
        n_bins = data_params["n_octaves"] * data_params["bins_per_octave"]

        # Validate that the CQT bins can be evenly split into the desired number of bands
        assert (
            n_bins * input_channels
        ) % num_bands == 0, f"Total features ({n_bins * input_channels}) must be divisible by the number of bands ({num_bands})."

        # --- Model Components ---

        # 1. BandSplit: This module reshapes the input and projects it into the model's dimension.
        # It splits the flattened frequency+harmonic dimension into multiple 'bands'.
        dims_per_band = (n_bins * input_channels) // num_bands
        self.band_split = BandSplit(
            dim=dim, dim_inputs=tuple([dims_per_band] * num_bands)
        )

        # 2. Transformer Blocks: The core of the model, copied from the original BSRoformer.
        self.layers = nn.ModuleList([])
        transformer_kwargs = dict(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            flash_attn=True,  # Assuming flash attention is available
            norm_output=False,
        )
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Transformer(depth=time_transformer_depth, **transformer_kwargs),
                        Transformer(depth=freq_transformer_depth, **transformer_kwargs),
                    ]
                )
            )

        # 3. Final Normalization
        self.final_norm = RMSNorm(dim)

        # 4. Salience Estimator: This replaces the original MaskEstimator.
        # It's an MLP that projects the transformer's output to the desired salience map shape.
        self.salience_estimator = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, n_bins),  # Output one value per CQT bin
        )

    def forward(self, x):
        """
        x: Input HCQT of shape (batch, channels, bins, time)
        """
        # Reshape for BS-Roformer's transformer: (b, c, f, t) -> (b, t, c*f)
        x = x.permute(0, 3, 1, 2)  # -> (b, t, c, f)
        x = rearrange(x, "b t c f -> b t (c f)")

        # Pass through the BandSplitter to get shape (b, t, num_bands, dim)
        x = self.band_split(x)

        # Axial attention (Time and Frequency Transformers)
        for time_transformer, freq_transformer in self.layers:
            # Time attention
            x = rearrange(x, "b t f d -> b f t d")
            x, ps = pack([x], "* t d")
            x, _ = time_transformer(x)
            (x,) = unpack(x, ps, "* t d")

            # Frequency attention
            x = rearrange(x, "b f t d -> b t f d")
            x, ps = pack([x], "* f d")
            x, _ = freq_transformer(x)
            (x,) = unpack(x, ps, "* f d")

        x = self.final_norm(x)  # (b, t, num_bands, dim)

        # Average across the bands to get a single feature vector per time step
        x = reduce(x, "b t num_bands d -> b t d", "mean")

        # Use the salience estimator to predict the CQT bins for each time step
        salience_map = self.salience_estimator(x)  # (b, t, n_bins)

        # Reshape to the required output format: (b, t, f) -> (b, 1, f, t)
        salience_map = rearrange(salience_map, "b t f -> b 1 f t")

        return salience_map


from spectnt import SpecTNTForSalience
from ftanet_pytorch import FTAnet


class ConvViTForSalience(nn.Module):
    """
    A Vision Transformer with a 2-layer convolutional frontend for salience prediction.
    The convolutional layers act as a learned patch embedding mechanism.
    """

    def __init__(self, config):
        super().__init__()
        model_params = config["model_params"]
        data_params = config["data_params"]
        training_params = config["training_params"]

        # --- ViT Hyperparameters (with defaults from config) ---
        d_model = model_params.get("vit_d_model", 128)
        nhead = model_params.get("vit_nhead", 4)
        num_layers = model_params.get("vit_num_layers", 4)
        dim_feedforward = model_params.get("vit_dim_feedforward", 256)
        dropout = model_params.get("vit_dropout", 0.1)
        conv_channels = model_params.get("vit_conv_channels", [16, 32])

        # --- Input/Output Dimensions ---
        input_channels = model_params["input_channels"]
        n_bins = data_params["n_octaves"] * data_params["bins_per_octave"]
        patch_width = training_params["patch_width"]

        # --- 1. Convolutional Frontend (2x 5x5 layers) ---
        # This will act as our patch embedding layer, downsampling the input.
        self.conv_frontend = nn.Sequential(
            # Layer 1
            nn.Conv2d(
                input_channels, conv_channels[0], kernel_size=5, stride=2, padding=2
            ),
            nn.BatchNorm2d(conv_channels[0]),
            nn.GELU(),
            # Layer 2
            nn.Conv2d(
                conv_channels[0], conv_channels[1], kernel_size=5, stride=2, padding=2
            ),
            nn.BatchNorm2d(conv_channels[1]),
            nn.GELU(),
        )

        # --- 2. Calculate patch dimensions after convolution ---
        # Run a dummy input through the frontend to determine the resulting shape.
        with torch.no_grad():
            dummy_input = torch.randn(1, input_channels, n_bins, patch_width)
            conv_output = self.conv_frontend(dummy_input)
            _, patch_dim, num_patches_freq, num_patches_time = conv_output.shape
            num_patches = num_patches_freq * num_patches_time

        # --- 3. Linear projection to d_model ---
        # The output channel count of the convs (patch_dim) might not match the desired
        # d_model for the transformer, so we add a linear layer to project it.
        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c f t -> b (f t) c"), nn.Linear(patch_dim, d_model)
        )

        # --- 4. Positional Embedding & CLS Token ---
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, d_model))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.dropout = nn.Dropout(dropout)

        # --- 5. Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # --- 6. Prediction Head ---
        # Takes the CLS token output and predicts the full flattened salience map.
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, n_bins * patch_width)
        )

        # Store dimensions for reshaping the final output
        self.n_bins = n_bins
        self.patch_width = patch_width

    def forward(self, x):
        # x shape: (batch, channels, bins, time)

        # 1. Pass through convolutional frontend to create patch embeddings
        x = self.conv_frontend(
            x
        )  # -> (b, patch_dim, num_patches_freq, num_patches_time)

        # 2. Flatten and project patches to the transformer's dimension (d_model)
        x = self.to_patch_embedding(x)  # -> (b, num_patches, d_model)
        b, n, _ = x.shape

        # 3. Prepend the [CLS] token to the sequence of patches
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)  # -> (b, num_patches + 1, d_model)

        # 4. Add learnable positional embedding
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)

        # 5. Pass through the Transformer Encoder
        x = self.transformer_encoder(x)  # -> (b, num_patches + 1, d_model)

        # 6. Isolate the output of the [CLS] token and pass it to the prediction head
        cls_output = x[:, 0]
        salience_flat = self.mlp_head(cls_output)  # -> (b, n_bins * patch_width)

        # 7. Reshape the flat output back into the desired 2D salience map format
        salience_map = rearrange(
            salience_flat, "b (f t) -> b 1 f t", f=self.n_bins, t=self.patch_width
        )

        return salience_map


# --- MODEL REGISTRY ---
# Place this dictionary right after your model class definitions
MODEL_REGISTRY = {
    "SalienceNetV1": SalienceCNN,
    "SalienceNetV2": SalienceCNN,
    "SalienceNetV3": SalienceCNNLogits,
    "SalienceNetV4": lambda config: SalienceCNNLogits(config, version=4),
    "BSRoformer": BSRoformerForSalience,
    "SpecTNT": SpecTNTForSalience,
    "FTAnet": FTAnet,
    "ConvViT": ConvViTForSalience,
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
        if abs(t - cur_t) <= atol:  # If it's the same time frame, append frequency
            cur_freqs.append(f)
        else:  # If it's a new time frame, save the old one and start a new one
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
        scores = {"Precision": 0.0, "Recall": 0.0, "Accuracy": 0.0}
    else:
        scores = mir_eval.multipitch.evaluate(
            ref_time_seq, ref_freqs_seq, est_time_seq, est_freqs_seq
        )

    # Manually calculate F1-score for consistency
    p, r = scores["Precision"], scores["Recall"]
    if p + r == 0:
        f1 = 0.0
    else:
        f1 = 2 * p * r / (p + r)

    scores["F1-score"] = f1
    return scores


# -----------------------------------------------------------------------------
# SECTION 4: TRAINING & EVALUATION ENGINE
# -----------------------------------------------------------------------------

from dataprocessor import OnTheFlyProcessor


class Trainer:
    """Handles the model training and validation loop with checkpoint resuming."""

    def __init__(
        self,
        config,
        device,
        data_manager,
        log_callback,
        progress_callback,
        epoch_end_callback,
    ):
        self.config, self.device, self.data_manager = config, device, data_manager
        self.log, self.progress, self.epoch_end_callback = (
            log_callback,
            progress_callback,
            epoch_end_callback,
        )

        self.checkpoint_dir = os.path.join("checkpoints", config["run_id"])
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train(self, pretrain_cnn=False):
        """
        Main training controller. Handles one-time data split for new runs
        and can "heal" old configs from previous versions.
        """
        config_path = os.path.join(self.checkpoint_dir, "config.json")

        if os.path.exists(config_path):
            self.log("Found existing config.json, loading...")
            with open(config_path, "r") as f:
                self.config = json.load(f)

            # --- SELF-HEALING MECHANISM ---
            # Check if the loaded config is from an old version (missing the split)
            if "train_groups" not in self.config["data_params"]:
                self.log("--- WARNING: Old config format detected. ---")
                self.log(
                    "--- Generating a new persistent split and updating the config file. ---"
                )
                # Generate the split and overwrite the old config
                self.config = self.data_manager.generate_train_val_split(self.config)
                with open(config_path, "w") as f:
                    json.dump(self.config, f, indent=4)
                self.log("--- Config file has been upgraded successfully. ---")
            else:
                self.log("Persistent data split loaded successfully.")

        else:
            # This is a brand-new run.
            self.log(
                "No existing config found. Generating and saving a new data split."
            )
            self.config = self.data_manager.generate_train_val_split(self.config)
            with open(config_path, "w") as f:
                json.dump(self.config, f, indent=4)

        # The rest of the function proceeds as normal, now guaranteed to have a valid config
        if pretrain_cnn:
            return self._train_two_phase()
        else:
            return self._train_end_to_end()

    def _train_end_to_end(self):
        """The original, consolidated training method with resume capability."""
        self.log(f"Starting end-to-end training for run: {self.config['run_id']}")

        self.data_manager.device = self.device
        self.data_manager.processor = OnTheFlyProcessor(self.config, self.device)

        end_to_end_config = json.loads(json.dumps(self.config))  # Deep copy
        model_name = end_to_end_config["model_params"]["architecture_name"]

        model_class = MODEL_REGISTRY[model_name]
        model = model_class(end_to_end_config).to(self.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=end_to_end_config["training_params"]["learning_rate"],
            weight_decay=end_to_end_config["training_params"]["weight_decay"],
        )

        return self._run_training_loop(model, optimizer, phase_name="EndToEnd")

    def _train_two_phase(self):
        """Orchestrates the two-phase CNN pre-training and GRU training."""
        self.log(f"Starting two-phase training for run: {self.config['run_id']}")

        self.data_manager.device = self.device
        self.data_manager.processor = OnTheFlyProcessor(self.config, self.device)

        # --- PHASE 1: PRE-TRAIN CNN ---
        self.log("\n" + "=" * 20 + " PHASE 1: Pre-training CNN " + "=" * 20)

        pretrain_config = json.loads(json.dumps(self.config))
        total_epochs = pretrain_config["training_params"]["num_epochs"]
        pretrain_config["training_params"]["num_epochs"] = max(1, total_epochs // 10)

        cnn_model = SalienceCNNLogits(pretrain_config, version=3).to(self.device)
        cnn_optimizer = torch.optim.AdamW(
            cnn_model.parameters(),
            lr=pretrain_config["training_params"]["learning_rate"],
            weight_decay=pretrain_config["training_params"]["weight_decay"],
        )

        # This part is unchanged
        self._run_training_loop(cnn_model, cnn_optimizer, phase_name="Phase1_CNN")

        # --- PHASE 2: TRAIN GRU ---
        self.log("\n" + "=" * 20 + " PHASE 2: Training GRU " + "=" * 20)

        gru_train_config = json.loads(json.dumps(self.config))
        gru_train_config["training_params"]["num_epochs"] = total_epochs

        full_model = SalienceCNNLogits(gru_train_config, version=4).to(self.device)

        best_cnn_path = os.path.join(self.checkpoint_dir, "best_model_Phase1_CNN.pth")
        if os.path.exists(best_cnn_path):
            self.log("Loading best pre-trained CNN weights for Phase 2.")

            # 1. Load the state dict from the Phase 1 model.
            # Its keys might look like '_orig_mod.network.0.weight'
            pretrained_state_dict = torch.load(best_cnn_path, map_location=self.device)

            # 2. Get the state dict of the destination network.
            # Its keys look like '0.weight'
            target_network_dict = full_model.network.state_dict()

            # 3. Create a new, clean dictionary for loading.
            from collections import OrderedDict

            new_state_to_load = OrderedDict()

            for k, v in pretrained_state_dict.items():
                # First, handle the prefix from torch.compile
                if k.startswith("_orig_mod."):
                    k = k[len("_orig_mod.") :]

                # Second, handle the 'network.' prefix from the submodule
                if k.startswith("network."):
                    # Strip the prefix to get the key for the nn.Sequential module
                    new_key = k[len("network.") :]

                    # IMPORTANT: Only add the key if it exists in the target network.
                    # This automatically and safely handles the removal of the 'conv_out' layer.
                    if new_key in target_network_dict:
                        new_state_to_load[new_key] = v

            # 4. Load the cleaned and filtered state dict into the network submodule
            full_model.network.load_state_dict(new_state_to_load)
            self.log("  > Successfully loaded matching layers.")

        else:
            self.log(
                "WARNING: No best CNN model from Phase 1 found. Training GRU with random CNN weights."
            )

        # 1. Define the main learning rate
        main_lr = gru_train_config["training_params"]["learning_rate"]

        # 2. Create parameter groups with different learning rates
        #    The CNN backbone gets a much smaller LR (e.g., 1/10th) than the new GRU head.
        param_groups = [
            {"params": full_model.network.parameters(), "lr": main_lr / 500},
            {"params": full_model.gru.parameters(), "lr": main_lr},
            {"params": full_model.linear_out.parameters(), "lr": main_lr},
        ]

        # 3. Create the optimizer with these specific parameter groups
        fine_tune_optimizer = torch.optim.AdamW(
            param_groups,
            lr=main_lr,
            weight_decay=gru_train_config["training_params"]["weight_decay"],
        )

        self.log(
            f"Optimizer configured for fine-tuning. CNN LR: {main_lr/500:.1e}, GRU LR: {main_lr:.1e}"
        )

        return self._run_training_loop(
            full_model, fine_tune_optimizer, phase_name="Phase2_FineTune"
        )

    def _run_training_loop(self, model, optimizer, phase_name):
        """
        Generic training loop with phase-specific checkpointing, resuming, and profiling.
        """
        # --- Setup paths and configs ---
        tp = self.config["training_params"]
        num_epochs_total = tp.get("num_epochs", 30)

        if phase_name == "Phase1_CNN":
            num_epochs = max(1, num_epochs_total // 10)
        elif phase_name == "Phase2_GRU":
            num_epochs = num_epochs_total
        else:
            num_epochs = num_epochs_total

        # Use steps_per_epoch for iterable datasets
        steps_per_epoch = tp.get("steps_per_epoch", 1000)
        total_train_steps = steps_per_epoch * num_epochs
        steps_per_checkpoint = max(1, steps_per_epoch // 2)
        augmenter = SalienceAugment()

        train_loader, val_loader = self.data_manager.get_dataloaders(self.config)
        if train_loader is None:
            self.log(
                f"[{phase_name}] Failed to create dataloaders. Aborting training phase."
            )
            return self.checkpoint_dir, {}

        latest_model_path = os.path.join(
            self.checkpoint_dir, f"latest_model_{phase_name}.pth"
        )
        best_model_path = os.path.join(
            self.checkpoint_dir, f"best_model_{phase_name}.pth"
        )
        metrics_path = os.path.join(self.checkpoint_dir, f"metrics_{phase_name}.json")

        # --- Resume Logic ---
        global_step = 0
        best_val_loss = float("inf")
        metrics = {"train_loss": [], "val_loss": [], "steps": []}
        use_amp = self.device == "cuda"
        scaler = (
            torch.cuda.amp.GradScaler(enabled=use_amp)
            if self.device == "cuda"
            else torch.amp.GradScaler("cpu", enabled=False)
        )

        if os.path.exists(latest_model_path):
            self.log(f"Resuming training for '{phase_name}' from checkpoint.")
            checkpoint = torch.load(latest_model_path, map_location=self.device)

            from collections import OrderedDict

            model_state_dict = checkpoint["model_state_dict"]
            new_state_dict = OrderedDict()
            for k, v in model_state_dict.items():
                name = k.replace("_orig_mod.", "")
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)

            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            global_step = checkpoint.get("global_step", 0)
            best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            if "scaler_state_dict" in checkpoint and use_amp:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            if os.path.exists(metrics_path):
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)
            self.log(
                f"Resumed from step {global_step}. Best validation loss so far: {best_val_loss:.4f}"
            )

        if hasattr(torch, "compile"):
            model = torch.compile(model)
        self.log(
            f"[{phase_name}] Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters."
        )
        self.log(
            f"[{phase_name}] Will train for {num_epochs} epochs ({total_train_steps} steps)."
        )

        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass  # Fails on older PyTorch versions, safe to ignore
        model = model.to(memory_format=torch.channels_last)

        # --- PROFILER SETUP ---
        profiling_enabled = tp.get("enable_profiler", False)

        if profiling_enabled:
            self.log("--- PYTORCH PROFILER IS ENABLED ---")
            profiler_dir = os.path.join(self.checkpoint_dir, "profiler_logs")
            os.makedirs(profiler_dir, exist_ok=True)
            self.log(f"Profiler traces will be saved to: {profiler_dir}")
            schedule = torch.profiler.schedule(wait=1, warmup=3, active=2, repeat=1)
            trace_handler = torch.profiler.tensorboard_trace_handler(profiler_dir)
            profiler_context = torch.profiler.profile(
                schedule=schedule,
                on_trace_ready=trace_handler,
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            )
        else:
            profiler_context = contextlib.nullcontext()

        # --- Main Training Loop ---
        if global_step >= total_train_steps:
            self.log(f"[{phase_name}] already completed. Skipping.")
            return self.checkpoint_dir, metrics

        with profiler_context as prof:
            model.train()
            train_iterator = iter(train_loader)
            pbar = tqdm(
                initial=global_step, total=total_train_steps, desc=f"{phase_name} Steps"
            )
            train_loss_accumulator = []
            device_type_str = "cuda" if self.device == "cuda" else "cpu"

            # --- Asynchronous Pipeline Setup (Suggestion #9) ---
            use_overlap = (self.device == "cuda") and tp.get("overlap_prep", True)
            if not use_overlap:
                self.log("Running in synchronous mode (no data prep overlap).")

            if use_overlap:
                prep_stream = torch.cuda.Stream()
                curr = {"cqt": None, "f0": None}
                nxt = {"cqt": None, "f0": None}

                # --- Prime the pipeline ---
                audios_bt, f0_lists_b = next(train_iterator)
                with torch.cuda.stream(prep_stream):
                    h, s = self.data_manager.processor.batch_generate_features_on_gpu(
                        audios_bt, f0_lists_b
                    )
                    nxt["cqt"] = h.contiguous(memory_format=torch.channels_last)
                    nxt["f0"] = s

            while global_step < total_train_steps:
                if use_overlap:
                    # 1. Wait for next batch to be ready, then swap buffers
                    torch.cuda.current_stream().wait_stream(prep_stream)
                    curr, nxt = nxt, curr
                    cqt_batch, f0_batch = curr["cqt"], curr["f0"]

                    # 2. Kick off preparation for the *next* batch ASAP
                    try:
                        audios_bt, f0_lists_b = next(train_iterator)
                    except StopIteration:
                        train_iterator = iter(train_loader)
                        audios_bt, f0_lists_b = next(train_iterator)

                    with torch.cuda.stream(prep_stream):
                        h, s = (
                            self.data_manager.processor.batch_generate_features_on_gpu(
                                audios_bt, f0_lists_b
                            )
                        )
                        nxt["cqt"] = h.contiguous(memory_format=torch.channels_last)
                        nxt["f0"] = s
                else:  # Synchronous path for CPU/debugging
                    try:
                        audios_bt, f0_lists_b = next(train_iterator)
                    except StopIteration:
                        train_iterator = iter(train_loader)
                        audios_bt, f0_lists_b = next(train_iterator)

                    cqt_batch, f0_batch = (
                        self.data_manager.processor.batch_generate_features_on_gpu(
                            audios_bt, f0_lists_b
                        )
                    )
                    cqt_batch = cqt_batch.contiguous(memory_format=torch.channels_last)

                # --- Crop to patch width if necessary (optional) ---
                if cqt_batch.shape[-1] > self.tp["patch_width"]:
                    start = torch.randint(
                        0, cqt_batch.shape[-1] - self.tp["patch_width"] + 1, (1,)
                    ).item()
                    end = start + self.tp["patch_width"]
                    cqt_batch = cqt_batch[..., start:end]
                    f0_batch = f0_batch[..., start:end]

                # --- Augment & Train (tensors are already on GPU) ---
                cqt_batch, f0_batch = mixup_data(cqt_batch, f0_batch, 0.4)
                cqt_batch = augmenter(cqt_batch)

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=("cuda" if self.device == "cuda" else "cpu"),
                    dtype=torch.float16,
                    enabled=use_amp,
                ):
                    logits = model(cqt_batch)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits, f0_batch
                    )

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss_accumulator.append(loss.item())

                if profiling_enabled:
                    prof.step()
                global_step += 1
                pbar.update(1)

                if (global_step + 1) % steps_per_checkpoint == 0 or (
                    global_step + 1
                ) == total_train_steps:
                    avg_train_loss = (
                        sum(train_loss_accumulator) / len(train_loss_accumulator)
                        if train_loss_accumulator
                        else 0
                    )
                    metrics["train_loss"].append(avg_train_loss)
                    train_loss_accumulator = []

                    avg_val_loss = self._run_validation(
                        model, val_loader, use_amp=use_amp
                    )
                    metrics["val_loss"].append(avg_val_loss)
                    metrics["steps"].append(global_step + 1)

                    self.log(
                        f"Step {global_step+1}/{total_train_steps} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
                    )

                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        torch.save(model.state_dict(), best_model_path)
                        self.log(
                            f"  > New best val_loss for '{phase_name}'. Best model saved."
                        )

                    torch.save(
                        {
                            "global_step": global_step + 1,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "best_val_loss": best_val_loss,
                            "scaler_state_dict": scaler.state_dict(),
                        },
                        latest_model_path,
                    )

                    with open(metrics_path, "w") as f:
                        json.dump(metrics, f, indent=4)

                    self.epoch_end_callback(metrics.copy())
                    model.train()

                global_step += 1
                pbar.update(1)
                self.progress(global_step, total_train_steps)

            pbar.close()

        if profiling_enabled:
            self.log("--- PROFILER: Trace files have been saved. ---")
            self.log("--- Use 'tensorboard --logdir=./checkpoints' to view them. ---")

        self.log(f"[{phase_name}] finished.")
        return self.checkpoint_dir, metrics

    def _run_validation(self, model, val_loader, use_amp=False):
        # This method is unchanged
        model.eval()
        losses = []
        criterion = torch.nn.BCEWithLogitsLoss()
        with torch.no_grad():
            for cqt_batch, f0_batch in val_loader:
                cqt_batch = cqt_batch.to(self.device, non_blocking=True)
                f0_batch = f0_batch.to(self.device, non_blocking=True)
                with torch.autocast(
                    device_type=self.device, dtype=torch.float16, enabled=use_amp
                ):
                    logits = model(cqt_batch)
                    loss = criterion(logits, f0_batch)
                losses.append(loss.item())
        return float(sum(losses) / max(1, len(losses)))


class Evaluator:
    """Same as the provided complete version."""

    def __init__(self, checkpoint_path, device, log_callback, progress_callback):
        self.device = device
        self.log = log_callback
        self.progress = progress_callback

        config_path = os.path.join(checkpoint_path, "config.json")
        model_path = os.path.join(checkpoint_path, "best_model.pth")

        with open(config_path, "r") as f:
            self.config = json.load(f)

        # --- DYNAMIC MODEL INSTANTIATION ---
        model_name = self.config["model_params"]["architecture_name"]
        if model_name not in MODEL_REGISTRY:
            self.log(
                f"ERROR: Model '{model_name}' from checkpoint not found in registry. Cannot evaluate."
            )
            # Handle this gracefully, maybe by raising an exception or setting a flag
            self.model = None
            return

        model_class = MODEL_REGISTRY[model_name]
        self.model = model_class(self.config).to(self.device)

        from collections import OrderedDict

        checkpoint = torch.load(model_path, map_location=device)
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k.replace("_orig_mod.", "")  # remove `_orig_mod.`
            new_state_dict[name] = v

        self.model.load_state_dict(new_state_dict)
        self.model.eval()
        self.log(f"Evaluator ready. Loaded model '{model_name}' from {checkpoint_path}")

    def _run_inference(self, log_hcqt):
        """
        Runs model inference on a pre-computed log_hcqt representation.

        This version correctly handles overlapping patches by calculating a true
        average of their salience values. It uses two maps: one to sum the
        predictions and another to count the number of contributions for each bin.
        """
        tp = self.config["training_params"]
        n_bins = (
            self.config["data_params"]["n_octaves"]
            * self.config["data_params"]["bins_per_octave"]
        )
        print(log_hcqt.shape)
        patch_width, n_f = tp["patch_width"], log_hcqt.shape[2]
        step = int(patch_width * (1 - tp["patch_overlap"]))

        if n_f < patch_width:
            self.log(
                "Warning: Number of frames is smaller than patch width. Returning empty map."
            )
            return np.zeros((n_bins, n_f))

        patches = np.stack(
            [
                log_hcqt[:, :, st : st + patch_width]
                for st in range(0, n_f - patch_width + 1, step)
            ]
        )
        dataset = TensorDataset(torch.from_numpy(patches).float())
        loader = DataLoader(
            dataset,
            batch_size=self.config["evaluation_params"]["eval_batch_size"],
            shuffle=False,
        )

        # The output map's size is determined by the total span of all patches.
        total_frames = (len(patches) - 1) * step + patch_width

        # --- KEY CHANGE: Use two maps for averaging ---
        # 1. A map to accumulate the sum of predictions.
        sum_map = np.zeros((n_bins, total_frames), dtype=np.float32)
        # 2. A map to count how many patches contributed to each bin.
        count_map = np.zeros((n_bins, total_frames), dtype=np.float32)

        with torch.no_grad():
            for i, (batch,) in enumerate(loader):
                # Get model predictions, already in [0, 1] from the sigmoid.
                preds = (
                    torch.sigmoid(self.model(batch.to(self.device)))
                    .squeeze(1)
                    .cpu()
                    .numpy()
                )
                for j, p in enumerate(preds):
                    start_frame = (i * loader.batch_size + j) * step
                    end_frame = start_frame + patch_width

                    # Add the model's output to the sum map.
                    sum_map[:, start_frame:end_frame] += p
                    # Increment the count for the region covered by this patch.
                    count_map[:, start_frame:end_frame] += 1

        # --- Final Averaging Step ---
        # To avoid division by zero for any bins that were not covered,
        # we set their count to 1. Since their sum is also 0, this results in 0.
        count_map[count_map == 0] = 1.0

        # Element-wise division to get the final averaged salience map.
        salience_map = sum_map / count_map

        return salience_map

    def _get_ground_truth(self, processor, track_stems, root_dir, n_frames):
        """
        Gets ground truth (time, frequency) pairs for mir_eval using a robust
        heuristic to distinguish between dataset path styles.
        """
        dp = self.config["data_params"]
        frame_times = librosa.frames_to_time(
            np.arange(n_frames), sr=dp["sr"], hop_length=dp["hop_length"]
        )

        ref_times_list, ref_freqs_list = [], []

        for stem in track_stems:
            interp_freqs = None
            active = None

            # --- START OF THE FIX ---
            # The reliable heuristic: ChoralSynth stems contain path separators.
            is_choralsynth_style = os.sep in stem

            if is_choralsynth_style:
                # It's a ChoralSynth stem. Construct the path directly.
                crepe_path = os.path.join(root_dir, "ChoralSynth", f"{stem}.f0.csv")
                if not os.path.exists(crepe_path):
                    self.log(f"WARN: ChoralSynth F0 file not found at {crepe_path}")
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
                active = (interp_freqs >= dp["fmin"]) & (interp_confidence > 0.5)

            else:
                # It's a Cantoria/DCS style stem. Now it is safe to get the dataset folder.
                dataset_folder = processor.get_dataset_folder(stem)
                if not dataset_folder:
                    self.log(
                        f"WARN: Could not find dataset folder for stem {stem}. Skipping."
                    )
                    continue

                base_path = os.path.join(root_dir, dataset_folder)
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
                active = (interp_freqs >= dp["fmin"]) & (interp_voiced > 0.5)
            # --- END OF THE FIX ---

            if active is not None and np.any(active):
                ref_times_list.extend(frame_times[active])
                ref_freqs_list.extend(interp_freqs[active])

        ref_times = np.array(ref_times_list)
        ref_freqs = np.array(ref_freqs_list)
        if ref_times.size == 0:
            return ref_times, ref_freqs

        sort_indices = np.argsort(ref_times)
        return ref_times[sort_indices], ref_freqs[sort_indices]

    def _extract_pitches(self, salience_map, threshold):
        """
        Extracts pitch contours from a salience map. It finds the top 4 peaks in each
        time frame and refines the frequency of each peak by calculating a weighted
        average of frequencies in a 3-bin neighborhood around the peak.
        """
        # --- Basic setup ---
        dp = self.config["data_params"]
        n_bins = dp["n_octaves"] * dp["bins_per_octave"]
        times = librosa.times_like(
            salience_map, sr=dp["sr"], hop_length=dp["hop_length"]
        )
        # Get the center frequency for each CQT bin
        freqs = librosa.cqt_frequencies(
            n_bins=n_bins, fmin=dp["fmin"], bins_per_octave=dp["bins_per_octave"]
        )

        est_times, est_freqs = [], []

        # --- Iterate through each time frame ---
        for t_idx in range(salience_map.shape[1]):
            # 1. Find all peaks above the given threshold
            peaks, properties = find_peaks(salience_map[:, t_idx], height=threshold)

            if peaks.size > 0:
                # 2. Limit to the top 4 most salient peaks
                peak_heights = properties["peak_heights"]
                # Sort peaks by height (salience) in descending order and take the top 4
                sorted_peak_indices = np.argsort(peak_heights)[::-1][:4]
                top_peaks = peaks[sorted_peak_indices]

                # 3. For each of the top peaks, refine the frequency estimate
                for peak_bin in top_peaks:
                    # Define the 3-bin neighborhood around the peak bin
                    # Handle edge cases where the peak is at the lowest or highest bin
                    start_bin = max(0, peak_bin - 1)
                    end_bin = min(n_bins, peak_bin + 2)  # slice is exclusive at the end

                    # Isolate the salience values and corresponding frequencies in the neighborhood
                    salience_window = salience_map[start_bin:end_bin, t_idx]
                    freq_window = freqs[start_bin:end_bin]

                    # Normalize the salience values in the window to act as probabilities/weights
                    salience_sum = salience_window.sum()
                    if salience_sum > 1e-9:  # Avoid division by zero
                        weights = salience_window / salience_sum
                        # Calculate the refined frequency as a weighted average
                        refined_freq = np.sum(weights * freq_window)
                    else:
                        # Fallback to the center frequency if saliences are all zero (highly unlikely for a peak)
                        refined_freq = freqs[peak_bin]

                    # Append the time and the newly refined frequency
                    est_times.append(times[t_idx])
                    est_freqs.append(refined_freq)

        return np.array(est_times), np.array(est_freqs)

    def evaluate_track(self, track_stems, root_dir, threshold):
        canonical_name = "_".join(sorted(track_stems))
        # Create a SHA-1 hash of this string to get a safe, fixed-length filename.
        group_hash = hashlib.sha1(canonical_name.encode("utf-8")).hexdigest()
        self.log(
            f"Evaluating: {', '.join(track_stems)} ({group_hash}) @ thresh {threshold:.2f}"
        )

        # 1. Use the DataProcessor to ensure the data is cached and get the path.
        processor = DataProcessor(self.config, root_dir, "./cache", self.log)
        cache_path = processor.process_and_cache_group(track_stems)

        if not cache_path or not os.path.exists(cache_path):
            self.log(
                f"ERROR: Could not process or find cache for group. Aborting evaluation for this track."
            )
            empty_scores = {
                "F1-score": 0.0,
                "Precision": 0.0,
                "Recall": 0.0,
                "Accuracy": 0.0,
            }
            empty_data = (np.array([]), np.array([]))
            return empty_scores, empty_data, empty_data

        # 2. Load the pre-computed log_hcqt from the cache.
        cached_data = np.load(cache_path)
        log_hcqt = cached_data["log_hcqt"]

        # 3. Run inference on the pre-computed features.
        salience_map = self._run_inference(log_hcqt)

        # 4. Get ground truth pitches by re-analyzing the original F0 files for mir_eval.
        ref_times, ref_freqs = self._get_ground_truth(
            processor, track_stems, root_dir, salience_map.shape[1]
        )

        # 5. Extract estimated pitches from the salience map.
        est_times, est_freqs = self._extract_pitches(salience_map, threshold)

        # 6. Calculate scores.
        scores = calculate_f1_score(ref_times, ref_freqs, est_times, est_freqs)
        self.log(
            f"Scores: F1={scores['F1-score']:.3f}, P={scores['Precision']:.3f}, R={scores['Recall']:.3f}"
        )

        return scores, (ref_times, ref_freqs), (est_times, est_freqs)

    def tune_threshold(self, all_eval_track_groups, root_dir):
        """
        Tunes the peak-picking threshold using an efficient ternary search to find
        the value that maximizes the AVERAGE F1-score across the evaluation set.
        """
        num_tracks = len(all_eval_track_groups)
        self.log(f"Tuning threshold across {num_tracks} tracks using Ternary Search...")

        processor = DataProcessor(self.config, root_dir, "./cache", self.log)

        # --- Step 1: Pre-compute salience maps ---
        salience_maps = {}
        ground_truths = {}

        # Define necessary parameters for CQT calculation
        dp = self.config["data_params"]
        n_bins = dp["n_octaves"] * dp["bins_per_octave"]

        track_values = list(all_eval_track_groups.values())
        for i, track_stems in enumerate(
            tqdm(track_values, desc="Pre-computing Salience Maps", leave=False)
        ):
            group_id = "_".join(sorted(track_stems))

            # --- Audio Mixing (remains the same) ---
            max_len, all_y = 0, []
            for stem in track_stems:
                # This logic is specific to Cantoria/DCS and needs to be more robust
                # For now, we assume it works for the intended dataset
                is_choralsynth_style = os.sep in stem
                if is_choralsynth_style:
                    # This logic might need adjustment if tuning ChoralSynth tracks
                    audio_path = os.path.join(
                        self.root_dir, "ChoralSynth", f"{stem}.wav"
                    )
                else:
                    dataset_folder = processor.get_dataset_folder(stem)
                    audio_path = os.path.join(
                        root_dir, dataset_folder, "Audio", f"{stem}.wav"
                    )

                if not os.path.exists(audio_path):
                    continue
                y, _ = librosa.load(audio_path, sr=dp["sr"])
                all_y.append(y)
                if len(y) > max_len:
                    max_len = len(y)

            if not all_y:
                continue
            audio_mix = np.sum(
                [np.pad(y, (0, max_len - len(y))) for y in all_y], axis=0
            )

            # --- START: ADDED CQT CALCULATION BLOCK ---
            # This block was missing. It converts the 1D audio_mix into the 3D log_hcqt.
            cqt_list = [
                librosa.cqt(
                    y=audio_mix,
                    sr=dp["sr"],
                    hop_length=dp["hop_length"],
                    fmin=dp["fmin"] * h,
                    n_bins=n_bins,
                    bins_per_octave=dp["bins_per_octave"],
                )
                for h in dp["harmonics"]
            ]
            min_time = min(c.shape[1] for c in cqt_list)
            hcqt = np.stack([c[:, :min_time] for c in cqt_list])
            log_hcqt = (1.0 / 80.0) * librosa.amplitude_to_db(
                np.abs(hcqt), ref=np.max
            ) + 1.0
            # --- END: ADDED CQT CALCULATION BLOCK ---

            # Now, call _run_inference with the CORRECT data type
            salience_map = self._run_inference(log_hcqt)
            salience_maps[group_id] = salience_map

            ref_times, ref_freqs = self._get_ground_truth(
                processor, track_stems, root_dir, salience_map.shape[1]
            )
            ground_truths[group_id] = (ref_times, ref_freqs)

            self.progress(i + 1, num_tracks)

        self.log("All salience maps pre-computed. Starting efficient search...")

        # --- Step 2 & 3: Ternary Search (this part is unchanged) ---
        memo = {}

        def get_avg_f1(thresh):
            if thresh in memo:
                return memo[thresh]

            f1_scores = []
            for track_stems in all_eval_track_groups.values():
                lookup_id = "_".join(sorted(track_stems))
                if lookup_id not in salience_maps:
                    continue  # Skip if processing failed
                salience_map = salience_maps[lookup_id]
                ref_times, ref_freqs = ground_truths[lookup_id]

                est_times, est_freqs = self._extract_pitches(salience_map, thresh)
                scores = calculate_f1_score(ref_times, ref_freqs, est_times, est_freqs)
                f1_scores.append(scores["F1-score"])

            if not f1_scores:
                return 0.0  # Avoid error if all tracks failed
            avg_f1 = np.mean(f1_scores)
            self.log(f"  - Testing Threshold {thresh:.4f}: Average F1 = {avg_f1:.4f}")
            memo[thresh] = avg_f1
            return avg_f1

        low, high, iterations = 0.1, 0.8, 10
        for i in range(iterations):
            if (high - low) < 0.01:
                break
            self.progress(i + 1, iterations)
            m1, m2 = low + (high - low) / 3, high - (high - low) / 3
            f1_m1, f1_m2 = get_avg_f1(m1), get_avg_f1(m2)
            if f1_m1 < f1_m2:
                low = m1
            else:
                high = m2

        best_thresh = (low + high) / 2
        best_avg_f1 = get_avg_f1(best_thresh)

        self.log(f"--- Optimal threshold found via Ternary Search ---")
        self.log(f"  > Optimal Threshold: {best_thresh:.4f}")
        self.log(f"  > Best Average F1-score: {best_avg_f1:.4f}")

        return best_thresh, best_avg_f1


class HyperparameterTuner:
    def __init__(
        self, base_config, device, data_manager, log_callback, progress_callback
    ):
        self.base_config = base_config
        self.device = device
        self.data_manager = data_manager
        self.log = log_callback
        self.progress = progress_callback
        self.tuning_params = base_config["tuning_params"]
        self.search_space = self.tuning_params["search_space"]
        self.history = {"best_fitness": [], "avg_fitness": []}

    def _create_individual(self):
        ind = {}
        for key, space in self.search_space.items():
            if space["type"] == "log_uniform":
                ind[key] = np.exp(
                    random.uniform(np.log(space["range"][0]), np.log(space["range"][1]))
                )
            elif space["type"] == "uniform":
                ind[key] = random.uniform(space["range"][0], space["range"][1])
            elif space["type"] == "choice":
                ind[key] = random.choice(space["choices"])
        return ind

    def _evaluate_fitness(self, individual_params):
        # Create a temporary config for this evaluation run
        eval_config = self.base_config.copy()
        eval_config["training_params"]["learning_rate"] = individual_params[
            "learning_rate"
        ]
        eval_config["data_params"]["gaussian_sigma"] = individual_params[
            "gaussian_sigma"
        ]
        eval_config["training_params"]["num_epochs"] = self.tuning_params[
            "epochs_per_eval"
        ]

        run_id = f"tune_{datetime.now().strftime('%H%M%S_%f')}"
        eval_config["run_id"] = run_id

        # --- MODIFICATION: The trainer will generate and save the split ---
        # The Trainer's .train() method now handles generating the train/val split
        # and saving it to the config within the checkpoint directory.
        trainer = Trainer(
            eval_config,
            self.device,
            self.data_manager,
            lambda msg: None,
            lambda c, t: None,
            lambda metrics: None,
        )
        checkpoint_dir, _ = trainer.train()

        # Evaluation on one validation track
        evaluator = Evaluator(
            checkpoint_dir, self.device, lambda msg: None, lambda c, t: None
        )

        # --- MODIFICATION: Get validation tracks from the run's saved config ---
        # The evaluator.config now holds the config from the checkpoint,
        # including the 'val_groups' list that was just created.
        val_groups = evaluator.config["data_params"].get("val_groups", [])
        if not val_groups:
            self.log(
                "Tuner Warning: No validation groups found in the temp config. Cannot evaluate fitness."
            )
            shutil.rmtree(checkpoint_dir)
            return 0.0

        val_track_stems = random.choice(val_groups)

        # Use a fixed threshold for fair comparison during tuning
        scores, _, _ = evaluator.evaluate_track(
            val_track_stems, self.data_manager.root_dir, 0.3
        )

        shutil.rmtree(checkpoint_dir)  # Clean up temporary checkpoint
        return scores["F1-score"]

    def _selection(self, population, fitnesses):
        # Tournament selection
        selected = []
        for _ in range(len(population)):
            i1, i2 = random.sample(range(len(population)), 2)
            winner = i1 if fitnesses[i1] > fitnesses[i2] else i2
            selected.append(population[winner])
        return selected

    def _crossover(self, p1, p2):
        if random.random() > self.tuning_params["crossover_rate"]:
            return p1.copy(), p2.copy()
        c1, c2 = p1.copy(), p2.copy()
        for key in self.search_space:
            if random.random() < 0.5:
                c1[key], c2[key] = c2[key], c1[key]
        return c1, c2

    def _mutation(self, individual):
        if random.random() > self.tuning_params["mutation_rate"]:
            return individual
        mutated_ind = individual.copy()
        key_to_mutate = random.choice(list(self.search_space.keys()))
        space = self.search_space[key_to_mutate]
        if space["type"] == "log_uniform":
            mutated_ind[key_to_mutate] = np.exp(
                random.uniform(np.log(space["range"][0]), np.log(space["range"][1]))
            )
        elif space["type"] == "uniform":
            mutated_ind[key_to_mutate] = random.uniform(
                space["range"][0], space["range"][1]
            )
        elif space["type"] == "choice":
            mutated_ind[key_to_mutate] = random.choice(space["choices"])
        return mutated_ind

    def run_tuning(self):
        population = [
            self._create_individual()
            for _ in range(self.tuning_params["population_size"])
        ]

        for gen in range(self.tuning_params["num_generations"]):
            self.log(
                f"--- Generation {gen + 1}/{self.tuning_params['num_generations']} ---"
            )

            fitnesses = []
            for i, ind in enumerate(population):
                self.log(
                    f"  Evaluating individual {i+1}/{len(population)}: { {k: f'{v:.2e}' if 'rate' in k else f'{v:.2f}' for k,v in ind.items()} }"
                )
                fitness = self._evaluate_fitness(ind)
                fitnesses.append(fitness)
                self.log(f"    > Fitness (F1-score): {fitness:.4f}")

            best_fitness_gen = max(fitnesses)
            avg_fitness_gen = sum(fitnesses) / len(fitnesses)
            self.history["best_fitness"].append(best_fitness_gen)
            self.history["avg_fitness"].append(avg_fitness_gen)
            self.progress(gen + 1, self.tuning_params["num_generations"])

            self.log(
                f"Generation {gen+1} Summary | Best F1: {best_fitness_gen:.4f}, Avg F1: {avg_fitness_gen:.4f}"
            )

            parents = self._selection(population, fitnesses)
            next_population = []
            for i in range(0, len(parents), 2):
                p1, p2 = parents[i], parents[i + 1]
                c1, c2 = self._crossover(p1, p2)
                next_population.extend([self._mutation(c1), self._mutation(c2)])

            population = next_population

        best_gen_idx = np.argmax(self.history["best_fitness"])
        final_best_fitness = self.history["best_fitness"][best_gen_idx]
        self.log(
            f"Tuning finished. Best F1-score of {final_best_fitness:.4f} achieved in generation {best_gen_idx+1}."
        )
        return self.history


# -----------------------------------------------------------------------------
# SECTION 5: GUI APPLICATION
# -----------------------------------------------------------------------------


class SalienceStudioApp(ctk.CTk):
    def __init__(self, root_dir):
        super().__init__()
        self.title("Deep Salience Studio")
        self.geometry("1200x850")
        ctk.set_appearance_mode("dark")
        self.root_dir, self.cache_dir = root_dir, "./cache"
        self.config = get_default_config()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ui_queue = Queue()

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.console = None  # Will be created in _create_train_tab

        # 1. Initialize the DatasetManager (it does no work yet)
        self.data_manager = DatasetManager(self.root_dir, self.cache_dir, self.log)
        self.data_manager.device = self.device

        self.tab_view = ctk.CTkTabview(self, anchor="w")
        self.tab_view.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.tab_view.add("Train")
        self.tab_view.add("Evaluate")
        self.tab_view.add("Hyper-parameter Tuning")

        # 2. Build the static UI layout for all tabs.
        #    The train tab will create the console and data-widget placeholders.
        self._create_train_tab()
        self._create_evaluate_tab()
        self._create_tuning_tab()

        # 3. NOW that the console exists, run the discovery.
        self.data_manager.run_discovery()

        # 4. NOW that discovery is complete, populate the widgets with the found datasets.
        self._update_dataset_widgets()

        self.check_ui_queue()
        self.log(f"Welcome to Deep Salience Studio! Using device: {self.device}")

        self._on_model_select(
            self.train_widgets["architecture_name"].get()
        )  # Allign invisible button

    def _compare_configs(self, old_config, new_config, prefix=""):
        """
        Recursively compares two configuration dictionaries and returns a list
        of human-readable differences.
        """
        diffs = []
        # Get a set of all unique keys from both dictionaries
        all_keys = set(old_config.keys()) | set(new_config.keys())

        for key in sorted(list(all_keys)):
            old_value = old_config.get(key, "<<MISSING>>")
            new_value = new_config.get(key, "<<MISSING>>")

            # If the values are nested dictionaries, recurse
            if isinstance(old_value, dict) and isinstance(new_value, dict):
                nested_diffs = self._compare_configs(
                    old_value, new_value, prefix=f"{prefix}{key}."
                )
                diffs.extend(nested_diffs)
            # If the values are different, record the change
            elif old_value != new_value:
                diff_line = (
                    f"- Key '{prefix}{key}': "
                    f"'{str(old_value)}' (existing) != '{str(new_value)}' (new)"
                )
                diffs.append(diff_line)

        return diffs

    def _create_train_tab(self):
        tab = self.tab_view.tab("Train")

        tab.grid_columnconfigure(0, weight=1)
        tab.grid_columnconfigure(1, weight=2)
        tab.grid_rowconfigure(0, weight=1)

        left_frame = ctk.CTkFrame(tab)
        left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        left_frame.grid_columnconfigure(0, weight=1)
        settings_frame = ctk.CTkScrollableFrame(
            left_frame, label_text="Training Configuration"
        )
        settings_frame.pack(padx=10, pady=10, fill="both", expand=True)

        self.train_widgets = {}
        # (All other widgets are created as before)
        ctk.CTkLabel(
            settings_frame, text="Run Name (Optional)", font=ctk.CTkFont(weight="bold")
        ).pack(anchor="w", padx=10, pady=(10, 0))
        self.train_widgets["run_name"] = ctk.CTkEntry(
            settings_frame, placeholder_text="e.g., Experiment-With-GELU"
        )
        self.train_widgets["run_name"].pack(fill="x", padx=10, pady=(0, 10))
        ctk.CTkLabel(
            settings_frame, text="Model Architecture", font=ctk.CTkFont(weight="bold")
        ).pack(anchor="w", padx=10, pady=(10, 0))
        model_names = list(MODEL_REGISTRY.keys())
        self.train_widgets["architecture_name"] = ctk.CTkOptionMenu(
            settings_frame, values=model_names, command=self._on_model_select
        )
        self.train_widgets["architecture_name"].set(
            self.config["model_params"]["architecture_name"]
        )
        self.train_widgets["architecture_name"].pack(fill="x", padx=10, pady=(0, 10))

        # --- Create placeholders for data-dependent widgets ---
        ctk.CTkLabel(
            settings_frame, text="Training Datasets", font=ctk.CTkFont(weight="bold")
        ).pack(anchor="w", padx=10, pady=(10, 0))
        # This frame will be populated later by _update_dataset_widgets
        self.train_dataset_checkbox_frame = ctk.CTkFrame(
            settings_frame, fg_color="transparent"
        )
        self.train_dataset_checkbox_frame.pack(anchor="w", padx=20, fill="x")
        self.train_dataset_checkboxes = {}

        ctk.CTkLabel(
            settings_frame,
            text="Validation Split Ratio (e.g., 0.15)",
            font=ctk.CTkFont(weight="bold"),
        ).pack(anchor="w", padx=10, pady=(10, 0))
        self.train_widgets["val_split_ratio"] = ctk.CTkEntry(
            settings_frame,
            placeholder_text=str(self.config["training_params"]["val_split_ratio"]),
        )
        self.train_widgets["val_split_ratio"].pack(fill="x", padx=10, pady=(0, 10))

        # (The rest of the widgets are created as before)
        ctk.CTkLabel(
            settings_frame,
            text="Training Hyperparameters",
            font=ctk.CTkFont(weight="bold"),
        ).pack(anchor="w", padx=10, pady=(10, 0))
        ctk.CTkLabel(settings_frame, text="Learning Rate").pack(anchor="w", padx=10)
        self.train_widgets["learning_rate"] = ctk.CTkEntry(
            settings_frame,
            placeholder_text=f"{self.config['training_params']['learning_rate']:.1e}",
        )
        self.train_widgets["learning_rate"].pack(fill="x", padx=10)
        ctk.CTkLabel(settings_frame, text="Epochs").pack(
            anchor="w", padx=10, pady=(10, 0)
        )
        self.train_widgets["num_epochs"] = ctk.CTkEntry(
            settings_frame,
            placeholder_text=str(self.config["training_params"]["num_epochs"]),
        )
        self.train_widgets["num_epochs"].pack(fill="x", padx=10)
        ctk.CTkLabel(settings_frame, text="Batch Size").pack(
            anchor="w", padx=10, pady=(10, 0)
        )
        self.train_widgets["batch_size"] = ctk.CTkEntry(
            settings_frame,
            placeholder_text=str(self.config["training_params"]["batch_size"]),
        )
        self.train_widgets["batch_size"].pack(fill="x", padx=10)

        self.train_widgets["enable_profiler"] = ctk.StringVar(value="off")
        profiler_checkbox = ctk.CTkCheckBox(
            settings_frame,
            text="Enable Profiler (for performance debugging)",
            variable=self.train_widgets["enable_profiler"],
            onvalue="on",
            offvalue="off",
        )
        profiler_checkbox.pack(anchor="w", padx=10, pady=(10, 10))

        # (All other frames and widgets are created as before)
        self.train_button = ctk.CTkButton(
            left_frame, text="Start Training", command=self.start_training_thread
        )
        self.train_button.pack(padx=10, pady=(0, 5), fill="x")

        # Create the button, but don't pack it yet if the default model isn't SalienceNetV4
        self.pretrain_button = ctk.CTkButton(
            left_frame,
            text="Pre-train CNN, then Train GRU",
            command=lambda: self.start_training_thread(pretrain_cnn=True),
        )

        # This progress bar will act as our positional anchor
        self.progress_bar = ctk.CTkProgressBar(left_frame)
        self.progress_bar.pack(padx=10, pady=10, fill="x")

        right_frame = ctk.CTkFrame(tab)
        right_frame.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="nsew")
        right_frame.grid_rowconfigure(0, weight=1)
        right_frame.grid_rowconfigure(1, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)
        self.train_viz_frame = ctk.CTkFrame(right_frame)
        self.train_viz_frame.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="nsew")
        ctk.CTkLabel(self.train_viz_frame, text="Training & Validation Loss").pack()
        self.train_canvas_widget = None
        console_frame = ctk.CTkFrame(right_frame)
        console_frame.grid(row=1, column=0, padx=10, pady=(5, 10), sticky="nsew")
        console_frame.grid_rowconfigure(1, weight=1)
        console_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(console_frame, text="Log Console").grid(
            row=0, column=0, sticky="w", padx=5
        )
        self.console = ctk.CTkTextbox(console_frame, wrap="word")
        self.console.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self._on_model_select(self.train_widgets["architecture_name"].get())

    def _update_dataset_widgets(self):
        """Populates the dataset selection widgets with data from the manager."""
        # Get the list of discovered dataset names
        dataset_names = list(self.data_manager.track_groups.keys())

        # If no datasets were found, display a message
        if not dataset_names:
            ctk.CTkLabel(
                self.train_dataset_checkbox_frame, text="No datasets found."
            ).pack()
            return

        # Populate the checkboxes for training datasets
        for name in dataset_names:
            var = ctk.StringVar(value="on")
            chk = ctk.CTkCheckBox(
                self.train_dataset_checkbox_frame,
                text=name,
                variable=var,
                onvalue="on",
                offvalue="off",
            )
            chk.pack(anchor="w")
            self.train_dataset_checkboxes[name] = var

    def _on_model_select(self, model_name):
        """Shows or hides the pre-training button in the correct position."""
        if model_name.startswith("SalienceNetV4"):
            # Show the button by packing it specifically *before* the progress bar.
            self.pretrain_button.pack(
                padx=10, pady=5, fill="x", before=self.progress_bar
            )
        else:
            # Hide the button by removing it from the layout manager.
            self.pretrain_button.pack_forget()

    def _create_evaluate_tab(self):
        tab = self.tab_view.tab("Evaluate")
        tab.grid_columnconfigure(0, weight=2)
        tab.grid_columnconfigure(1, weight=1)
        tab.grid_rowconfigure(0, weight=2)
        tab.grid_rowconfigure(1, weight=1)

        settings_frame = ctk.CTkFrame(tab)
        settings_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # --- CREATE WIDGETS FIRST ---
        ctk.CTkLabel(settings_frame, text="Checkpoint").pack(
            anchor="w", padx=10, pady=(10, 0)
        )
        self.checkpoint_menu = ctk.CTkOptionMenu(
            settings_frame, values=["None"], command=self.on_checkpoint_selected
        )
        self.checkpoint_menu.pack(fill="x", padx=10)

        ctk.CTkLabel(settings_frame, text="Evaluation Track").pack(
            anchor="w", padx=10, pady=(10, 0)
        )
        self.eval_track_menu = ctk.CTkOptionMenu(settings_frame, values=["None"])
        self.eval_track_menu.pack(fill="x", padx=10)

        ctk.CTkLabel(settings_frame, text="Threshold").pack(
            anchor="w", padx=10, pady=(10, 0)
        )
        self.threshold_entry = ctk.CTkEntry(settings_frame, placeholder_text="0.3")
        self.threshold_entry.pack(fill="x", padx=10)

        self.eval_button = ctk.CTkButton(
            settings_frame,
            text="Evaluate Track",
            command=self.start_evaluation_thread,
            state="disabled",
        )
        self.eval_button.pack(fill="x", padx=10, pady=20)

        self.tune_single_track_button = ctk.CTkButton(
            settings_frame,
            text="Tune Threshold for This Track",
            command=self.start_single_track_tuning_thread,
            state="disabled",
        )
        self.tune_single_track_button.pack(fill="x", padx=10, pady=10)

        self.tune_button = ctk.CTkButton(
            settings_frame,
            text="Auto-Tune Threshold (All Tracks)",
            command=self.start_tuning_thread,
            state="disabled",
        )
        self.tune_button.pack(fill="x", padx=10, pady=(0, 10))

        self.eval_full_dataset_button = ctk.CTkButton(
            settings_frame,
            text="Evaluate Full Dataset",
            command=self.start_full_dataset_evaluation_thread,
            state="disabled",
        )
        self.eval_full_dataset_button.pack(fill="x", padx=10, pady=(0, 20))

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

        ctk.CTkLabel(
            settings_frame,
            text="Genetic Algorithm Parameters",
            font=ctk.CTkFont(weight="bold"),
        ).pack(anchor="w", padx=10, pady=(10, 0))

        ctk.CTkLabel(settings_frame, text="Population Size").pack(anchor="w", padx=10)
        self.tuning_widgets["population_size"] = ctk.CTkEntry(
            settings_frame,
            placeholder_text=str(self.config["tuning_params"]["population_size"]),
        )
        self.tuning_widgets["population_size"].pack(fill="x", padx=10)

        ctk.CTkLabel(settings_frame, text="Number of Generations").pack(
            anchor="w", padx=10, pady=(10, 0)
        )
        self.tuning_widgets["num_generations"] = ctk.CTkEntry(
            settings_frame,
            placeholder_text=str(self.config["tuning_params"]["num_generations"]),
        )
        self.tuning_widgets["num_generations"].pack(fill="x", padx=10)

        ctk.CTkLabel(settings_frame, text="Epochs per Evaluation").pack(
            anchor="w", padx=10, pady=(10, 0)
        )
        self.tuning_widgets["epochs_per_eval"] = ctk.CTkEntry(
            settings_frame,
            placeholder_text=str(self.config["tuning_params"]["epochs_per_eval"]),
        )
        self.tuning_widgets["epochs_per_eval"].pack(fill="x", padx=10)

        ctk.CTkLabel(
            settings_frame,
            text="Note: Search space (LR, etc.) is currently hard-coded in config.py.",
        ).pack(anchor="w", padx=10, pady=(10, 0))

        # --- Control Buttons and Progress Bar ---
        self.tune_run_button = ctk.CTkButton(
            tab,
            text="Start Hyper-parameter Tuning",
            command=self.start_hp_tuning_thread,
        )
        self.tune_run_button.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

        self.tune_progress_bar = ctk.CTkProgressBar(tab)
        self.tune_progress_bar.set(0)
        self.tune_progress_bar.grid(row=2, column=0, padx=10, pady=10, sticky="ew")

        # --- Visualization Panel ---
        self.tune_viz_frame = ctk.CTkFrame(tab)
        self.tune_viz_frame.grid(
            row=0, column=1, rowspan=3, padx=10, pady=10, sticky="nsew"
        )
        ctk.CTkLabel(self.tune_viz_frame, text="Tuning Fitness History").pack()
        self.tune_canvas_widget = None

    def start_training_thread(self, pretrain_cnn=False):
        self.update_config_from_ui()

        run_name_raw = self.train_widgets["run_name"].get()
        if run_name_raw:
            import re

            run_name_sanitized = (
                re.sub(r"[^\w\-_\. ]", "", run_name_raw).strip().replace(" ", "_")
            )
        else:
            run_name_sanitized = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        config_hash = get_config_hash(self.config)

        checkpoints_dir = "checkpoints"
        os.makedirs(checkpoints_dir, exist_ok=True)
        for existing_folder in os.listdir(checkpoints_dir):
            if existing_folder.endswith(f"-{run_name_sanitized}"):
                existing_hash = existing_folder.split("-")[0]
                if existing_hash != config_hash:
                    # --- CONFLICT DETECTED: INITIATE DIFF-CHECK ---
                    self.log("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    self.log("ERROR: Name conflict detected!")

                    # 1. Load the existing configuration file
                    old_config_path = os.path.join(
                        checkpoints_dir, existing_folder, "config.json"
                    )
                    try:
                        with open(old_config_path, "r") as f:
                            old_config = json.load(f)

                        # 2. Compare with the new config and get differences
                        diffs = self._compare_configs(old_config, self.config)

                        # 3. Log the differences clearly
                        if diffs:
                            self.log("--- Configuration Differences ---")
                            for diff_line in diffs:
                                self.log(diff_line)
                            self.log("---------------------------------")
                        else:
                            self.log(
                                "--- No parameter differences found (hash may differ due to formatting/volatile keys)."
                            )

                    except (FileNotFoundError, json.JSONDecodeError) as e:
                        self.log(
                            f"--- Could not load or parse existing config to check differences: {e}"
                        )

                    # 4. Log the original error message and abort
                    self.log(
                        f"The name '{run_name_sanitized}' is already used by a checkpoint with a DIFFERENT configuration."
                    )
                    self.log(f"  > Existing folder: {existing_folder}")
                    self.log(f"  > Your current config hash: {config_hash}")
                    self.log(
                        "Please choose a unique name for this new configuration, or delete the old checkpoint folder."
                    )
                    self.log("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    return

        final_run_id = f"{config_hash}-{run_name_sanitized}"
        self.config["run_id"] = final_run_id

        self.log(f"Starting run. Checkpoint ID: {final_run_id}")

        trainer = Trainer(
            self.config,
            self.device,
            self.data_manager,
            self.log_threadsafe,
            self.progress_threadsafe,
            self.epoch_end_threadsafe,
        )

        threading.Thread(
            target=self.run_training_and_get_results,
            args=(trainer, pretrain_cnn),
            daemon=True,
        ).start()

        self.train_button.configure(state="disabled", text="Training...")
        self.pretrain_button.configure(state="disabled")

    def run_training_and_get_results(self, trainer, pretrain_cnn):
        # The trainer's train method now handles the logic
        checkpoint_dir, final_metrics = trainer.train(pretrain_cnn=pretrain_cnn)
        self.ui_queue.put(("training_complete", (checkpoint_dir, final_metrics)))

    def start_evaluation_thread(self):
        try:
            threshold = float(self.threshold_entry.get())
        except:
            threshold = 0.3
        checkpoint_path = os.path.join("checkpoints", self.checkpoint_menu.get())
        evaluator = Evaluator(
            checkpoint_path,
            self.device,
            self.log_threadsafe,
            self.eval_progress_threadsafe,
        )

        threading.Thread(
            target=self.run_evaluation,
            args=(evaluator, track_stems, threshold),
            daemon=True,
        ).start()
        self.eval_button.configure(state="disabled")
        self.tune_button.configure(state="disabled")

    def run_evaluation(self, evaluator, track_stems, threshold):
        self.ui_queue.put(
            (
                "evaluation_complete",
                evaluator.evaluate_track(track_stems, self.root_dir, threshold),
            )
        )

    def start_tuning_thread(self):
        checkpoint_path = os.path.join("checkpoints", self.checkpoint_menu.get())
        evaluator = Evaluator(
            checkpoint_path,
            self.device,
            self.log_threadsafe,
            self.eval_progress_threadsafe,
        )

    def run_tuning(self, evaluator, all_tracks_for_eval):
        # We need to pass the dictionary values (the lists of stems) to the method
        best_thresh, best_f1 = evaluator.tune_threshold(
            all_tracks_for_eval, self.root_dir
        )
        self.ui_queue.put(("tuning_complete", (best_thresh, best_f1)))

    def start_single_track_tuning_thread(self):
        """Starts threshold tuning for only the currently selected track."""
        checkpoint_path = os.path.join("checkpoints", self.checkpoint_menu.get())
        evaluator = Evaluator(
            checkpoint_path,
            self.device,
            self.log_threadsafe,
            self.eval_progress_threadsafe,
        )

        # --- GET ONLY THE SELECTED TRACK ---
        selected_track_id = self.eval_track_menu.get()
        if selected_track_id == "None":
            self.log("ERROR: No track selected to tune for. Please select a track.")
            return

        # Create a dictionary containing only the selected track's stems
        # This format is required by the tune_threshold method
        track_stems = self.data_manager.track_groups[eval_dataset_name].get(
            selected_track_id
        )
        single_track_dict = {selected_track_id: track_stems}

        self.log(f"Starting threshold tuning for single track: {selected_track_id}")

        # The run_tuning method is generic enough to work with our single-item dictionary
        threading.Thread(
            target=self.run_tuning, args=(evaluator, single_track_dict), daemon=True
        ).start()

        # Disable all buttons during operation
        self.eval_button.configure(state="disabled")
        self.tune_single_track_button.configure(state="disabled")
        self.tune_button.configure(state="disabled")

    def start_full_dataset_evaluation_thread(self):
        """Starts a thread to evaluate the model on the entire evaluation dataset."""
        try:
            threshold = float(self.threshold_entry.get())
        except (ValueError, TypeError):
            threshold = 0.3  # Use default if empty or invalid

        checkpoint_name = self.checkpoint_menu.get()
        if checkpoint_name == "None":
            self.log("ERROR: No checkpoint selected for evaluation.")
            return

        checkpoint_path = os.path.join("checkpoints", checkpoint_name)

        self.log(f"Starting full dataset evaluation for model: {checkpoint_name}")

        # Disable buttons
        self.eval_button.configure(state="disabled")
        self.tune_button.configure(state="disabled")
        self.tune_single_track_button.configure(state="disabled")
        self.eval_full_dataset_button.configure(state="disabled")

        # The worker function will need access to the evaluator and track info
        evaluator = Evaluator(
            checkpoint_path,
            self.device,
            self.log_threadsafe,
            self.eval_progress_threadsafe,
        )
        eval_dataset_name = evaluator.config["data_params"]["eval_dataset"]
        all_tracks_for_eval = self.data_manager.track_groups[eval_dataset_name]

        if not all_tracks_for_eval:
            self.log(
                f"ERROR: No tracks found in the evaluation dataset '{eval_dataset_name}'."
            )
            self.handle_full_evaluation_completion(None)  # Re-enable buttons
            return

        threading.Thread(
            target=self.run_full_dataset_evaluation,
            args=(evaluator, all_tracks_for_eval, threshold, checkpoint_path),
            daemon=True,
        ).start()

    def run_full_dataset_evaluation(
        self, evaluator, all_tracks_for_eval, threshold, checkpoint_path
    ):
        """Worker thread to perform evaluation, calculate stats, and save plots/scores."""
        # Create a unique results folder
        results_dir = os.path.join(
            checkpoint_path,
            f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        plots_dir = os.path.join(results_dir, "pitch_plots")
        os.makedirs(plots_dir, exist_ok=True)

        self.log_threadsafe(f"Results will be saved in: {results_dir}")

        all_scores = []
        track_items = list(all_tracks_for_eval.items())  # To get track names and stems
        num_tracks = len(track_items)

        for i, (track_id, track_stems) in enumerate(track_items):
            self.log_threadsafe(
                f"--- Evaluating track {i+1}/{num_tracks}: {track_id} ---"
            )

            scores, ref_data, est_data = evaluator.evaluate_track(
                track_stems, self.root_dir, threshold
            )

            score_entry = {"track_id": track_id, **scores}
            all_scores.append(score_entry)

            # Sanitize filename for PDF
            sanitized_track_id = track_id.replace(os.sep, "_").replace("/", "_")
            plot_filename = f"{sanitized_track_id}.pdf"
            plot_filepath = os.path.join(plots_dir, plot_filename)
            self._save_plot_to_pdf(ref_data, est_data, track_id, plot_filepath)

            self.eval_progress_threadsafe(i + 1, num_tracks)

        self.log_threadsafe(
            "--- Full dataset evaluation finished. Calculating statistics. ---"
        )

        if all_scores:
            df = pd.DataFrame(all_scores)
            stats = {}
            for metric in ["F1-score", "Precision", "Recall", "Accuracy"]:
                description = df[metric].describe()
                stats[metric] = {
                    "mean": description.get("mean", 0),
                    "std": description.get("std", 0),
                    "min": description.get("min", 0),
                    "q1": description.get("25%", 0),
                    "median": description.get("50%", 0),
                    "q3": description.get("75%", 0),
                    "max": description.get("max", 0),
                }

            results_data = {
                "model_checkpoint_hash": os.path.basename(checkpoint_path),
                "evaluation_timestamp": datetime.now().isoformat(),
                "threshold_used": threshold,
                "box_plot_statistics": stats,
                "track_by_track_scores": all_scores,
            }
            results_filepath = os.path.join(results_dir, "summary_scores.json")
            try:
                with open(results_filepath, "w") as f:
                    json.dump(results_data, f, indent=4)
                self.log_threadsafe(
                    f"Successfully saved summary statistics to {results_filepath}"
                )
            except Exception as e:
                self.log_threadsafe(f"ERROR: Failed to save summary file. Reason: {e}")
        else:
            self.log_threadsafe(
                "WARNING: No scores were generated, summary file not created."
            )

        self.ui_queue.put(("full_evaluation_complete", results_dir))

    def start_hp_tuning_thread(self):
        self.update_tuning_config_from_ui()
        tuner = HyperparameterTuner(
            self.config,
            self.device,
            self.data_manager,
            self.log_threadsafe,
            self.tune_progress_threadsafe,
        )
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
            if message == "epoch_complete":
                self.handle_epoch_completion(data)
            elif message == "training_complete":
                self.handle_training_completion(data)
            elif message == "evaluation_complete":
                self.handle_evaluation_completion(data)
            elif message == "tuning_complete":
                self.handle_tuning_completion(data)
            elif message == "hp_tuning_complete":
                self.handle_hp_tuning_completion(data)
            elif message == "full_evaluation_complete":
                self.handle_full_evaluation_completion(data)

        self.after(100, self.check_ui_queue)

    def handle_epoch_completion(self, metrics):
        """Called after each epoch to update the plot."""
        self.plot_training_loss(metrics)

    def handle_training_completion(self, data):
        """Called only when all epochs are finished."""
        checkpoint_dir, metrics = data
        self.log(f"Training run completed. Final results in {checkpoint_dir}")
        # Re-enable both buttons
        self.train_button.configure(state="normal", text="Train End-to-End")
        self.pretrain_button.configure(state="normal")
        self.progress_bar.set(1.0)
        self.plot_training_loss(metrics)
        self.refresh_checkpoints()

    def _save_evaluation_data_to_wav_files(self, ref_data, est_data):
        """
        A wrapper function that calls the main POLYPHONIC synthesizer for
        both ground truth and estimated pitch data, and logs completion.
        """
        self.log("Synthesizing evaluation results to .wav files...")
        pass  # TODO: Fix/implement polyphonic synthesis for demo
        try:
            ref_times, ref_freqs = ref_data
            est_times, est_freqs = est_data

            # Synthesize Ground Truth
            self.log("--- Synthesizing Ground Truth ---")
            self.synthesize_polyphonic_from_f0(
                ref_times, ref_freqs, "tmp_groundtruth.wav"
            )

            # Synthesize Model Estimation
            self.log("--- Synthesizing Model Estimation ---")
            self.synthesize_polyphonic_from_f0(
                est_times, est_freqs, "tmp_estimated.wav"
            )

            self.log("Finished synthesizing tmp_groundtruth.wav and tmp_estimated.wav.")

        except Exception as e:
            # The error log remains for debugging purposes
            self.log(f"ERROR: Failed to save synthesized WAV files. Reason: {e}")

    def handle_evaluation_completion(self, data):
        """
        Handles the UI update after a single track evaluation is complete.
        This now synthesizes the pitch data to MIDI and then to WAV files.
        """
        scores, ref_data, est_data = data

        # Call the new function to save the data
        self._save_evaluation_data_to_wav_files(ref_data, est_data)

        # Existing functionality remains the same
        self.plot_evaluation_result(ref_data, est_data)
        self.eval_button.configure(state="normal")
        self.tune_button.configure(state="normal")
        self.tune_single_track_button.configure(state="normal")
        self.eval_progress_bar.set(0)

    def handle_tuning_completion(self, data):
        best_thresh, best_f1 = data
        self.threshold_entry.delete(0, "end")
        self.threshold_entry.insert(0, f"{best_thresh:.2f}")
        self.eval_button.configure(state="normal")
        self.tune_button.configure(state="normal")
        self.tune_single_track_button.configure(state="normal")
        self.eval_progress_bar.set(0)

    def handle_hp_tuning_completion(self, data):
        history = data
        self.plot_tuning_history(history)
        self.tune_run_button.configure(state="normal")
        self.tune_progress_bar.set(0)

    def handle_full_evaluation_completion(self, results_dir):
        """Handles the UI update after a full dataset evaluation is complete."""
        if results_dir:
            self.log(
                f"Full dataset evaluation is complete. Results saved in {results_dir}"
            )
        # Re-enable all evaluation buttons
        self.eval_button.configure(state="normal")
        self.tune_button.configure(state="normal")
        self.tune_single_track_button.configure(state="normal")
        self.eval_full_dataset_button.configure(state="normal")
        self.eval_progress_bar.set(0)

    # --- Plotting methods ---
    def plot_training_loss(self, metrics):
        if self.train_canvas_widget:
            self.train_canvas_widget.get_tk_widget().destroy()

        # Extract data from the metrics dictionary
        steps = metrics.get("steps", [])
        train_loss = metrics.get("train_loss", [])
        val_loss = metrics.get("val_loss", [])

        # Create the plot with a dark facecolor
        fig, ax1 = plt.subplots(figsize=(6, 4), facecolor="#2B2B2B")

        # --- Configure the primary Y-axis (for Loss) ---
        color = "#1F6AA5"  # Blue for training loss
        ax1.set_xlabel("Training Steps", color="white")
        ax1.set_ylabel("Loss", color="white")

        # Plot Train Loss if data exists
        if steps and train_loss:
            ax1.plot(steps, train_loss, label="Train Loss", color=color, marker=".")

        # Plot Validation Loss if data exists
        if steps and val_loss:
            ax1.plot(
                steps, val_loss, label="Val Loss", color="#FFA500", marker="."
            )  # Orange for validation loss

        # Style the tick parameters for the primary axis
        ax1.tick_params(axis="y", colors="white")
        ax1.tick_params(axis="x", colors="white")
        ax1.grid(True, linestyle=":", color="gray", alpha=0.6)

        # --- Create a shared legend at the bottom ---
        # Get handles and labels from the axis
        lines1, labels1 = ax1.get_legend_handles_labels()

        # If there's anything to plot, create the legend
        if lines1:
            ax1.legend(
                lines1,
                labels1,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.15),  # Position the legend below the plot
                ncol=2,  # Number of columns in the legend
                labelcolor="white",  # Text color for the legend
                frameon=False,
            )  # No border around the legend

        # Set the main title and adjust layout to prevent labels from being cut off
        # fig.suptitle("Training History", color="white")
        fig.tight_layout(
            rect=[0, 0.05, 1, 1]
        )  # Adjust rect to make space for the bottom legend

        # --- Embed the plot in the Tkinter canvas ---
        canvas = FigureCanvasTkAgg(fig, master=self.train_viz_frame)
        self.train_canvas_widget = canvas
        canvas.draw()
        canvas.get_tk_widget().pack(side=ctk.TOP, fill=ctk.BOTH, expand=1)

        # Close the figure to free up memory
        plt.close(fig)

    def plot_evaluation_result(self, ref_data, est_data):
        if self.eval_canvas_widget:
            self.eval_canvas_widget.get_tk_widget().destroy()
        ref_times, ref_freqs = ref_data
        est_times, est_freqs = est_data
        fig, ax = plt.subplots(figsize=(14, 6), facecolor="#2B2B2B")
        if ref_times.size > 0:
            ax.scatter(
                ref_times,
                ref_freqs,
                c="black",
                marker=".",
                s=50,
                label="Reference",
                zorder=2,
            )
        if est_times.size > 0:
            ax.scatter(
                est_times,
                est_freqs,
                c="#e60000",
                marker=".",
                s=25,
                alpha=0.9,
                label="Prediction",
                zorder=3,
            )
        ax.set_yscale("log")
        ax.set_yticks([128, 256, 512, 1024], labels=["128", "256", "512", "1024"])
        ax.set_ylabel("Frequency (Hz)", color="white")
        ax.set_xlabel("Time (s)", color="white")
        ax.set_title("Pitch Estimation Result", color="white")
        ax.legend()
        ax.grid(True, linestyle=":", color="gray")
        ax.tick_params(axis="x", colors="white")
        ax.tick_params(axis="y", colors="white")
        if ref_times.size > 0:
            ax.set_xlim(ref_times.min() - 1, ref_times.max() + 1)
            ax.set_ylim(bottom=60)
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.eval_viz_frame)
        self.eval_canvas_widget = canvas
        canvas.draw()
        canvas.get_tk_widget().pack(side=ctk.TOP, fill=ctk.BOTH, expand=1)
        plt.close(fig)

    def _save_plot_to_pdf(self, ref_data, est_data, track_name, output_path):
        """Saves a pitch estimation plot to a PDF file. Designed to be thread-safe."""
        ref_times, ref_freqs = ref_data
        est_times, est_freqs = est_data

        fig, ax = plt.subplots(
            figsize=(14, 7), facecolor="#FFFFFF"
        )  # Use white background for PDF

        if ref_times.size > 0:
            ax.scatter(
                ref_times,
                ref_freqs,
                c="black",
                marker=".",
                s=50,
                label="Reference",
                zorder=2,
            )
        if est_times.size > 0:
            ax.scatter(
                est_times,
                est_freqs,
                c="#e60000",
                marker=".",
                s=25,
                alpha=0.7,
                label="Prediction",
                zorder=3,
            )

        ax.set_yscale("log")
        ax.set_yticks([128, 256, 512, 1024], labels=["128", "256", "512", "1024"])
        ax.set_ylabel("Frequency (Hz)")
        ax.set_xlabel("Time (s)")
        ax.set_title(f"Pitch Estimation for: {track_name}")
        ax.legend()
        ax.grid(True, linestyle=":", color="gray")

        if ref_times.size > 0 and ref_times.max() > ref_times.min():
            ax.set_xlim(ref_times.min() - 1, ref_times.max() + 1)
        ax.set_ylim(bottom=60)

        fig.tight_layout()

        try:
            fig.savefig(output_path, format="pdf")
            self.log_threadsafe(f"  > Saved plot to {os.path.basename(output_path)}")
        except Exception as e:
            self.log_threadsafe(
                f"  > ERROR: Failed to save plot to {output_path}. Reason: {e}"
            )
        finally:
            plt.close(fig)  # Important: release memory

    def plot_tuning_history(self, history):
        if self.tune_canvas_widget:
            self.tune_canvas_widget.get_tk_widget().destroy()
        fig, ax = plt.subplots(figsize=(6, 4), facecolor="#2B2B2B")
        ax.plot(
            history["best_fitness"], label="Best Fitness", color="#1F6AA5", marker="o"
        )
        ax.plot(
            history["avg_fitness"],
            label="Average Fitness",
            color="#FFA500",
            linestyle="--",
        )
        ax.set_title("Tuning Fitness History", color="white")
        ax.set_xlabel("Generation", color="white")
        ax.set_ylabel("F1-Score", color="white")
        ax.legend()
        ax.grid(True, linestyle=":", color="gray")
        ax.tick_params(axis="x", colors="white")
        ax.tick_params(axis="y", colors="white")
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.tune_viz_frame)
        self.tune_canvas_widget = canvas
        canvas.draw()
        canvas.get_tk_widget().pack(side=ctk.TOP, fill=ctk.BOTH, expand=1)

    # --- UI update and callback methods ---
    def refresh_checkpoints(self):
        if not os.path.exists("checkpoints"):
            self.checkpoint_menu.configure(values=["None"])
            return
        checkpoints = sorted(
            [
                d
                for d in os.listdir("checkpoints")
                if os.path.isdir(os.path.join("checkpoints", d))
            ],
            reverse=True,
        )
        if not checkpoints:
            checkpoints = ["None"]
        self.checkpoint_menu.configure(values=checkpoints)
        self.on_checkpoint_selected(checkpoints[0])

    def on_checkpoint_selected(self, name):
        is_none = name == "None"
        state = "disabled" if is_none else "normal"

        self.eval_button.configure(state=state)
        self.tune_button.configure(state=state)
        self.tune_single_track_button.configure(state=state)
        self.eval_full_dataset_button.configure(state=state)

        if is_none:
            self.eval_track_menu.configure(values=["None"])
            return

        try:
            with open(os.path.join("checkpoints", name, "config.json"), "r") as f:
                chk_config = json.load(f)
            eval_dataset_name = chk_config["data_params"]["eval_dataset"]
            track_ids = list(self.data_manager.track_groups[eval_dataset_name].keys())
            if not track_ids:
                track_ids = ["None"]
            self.eval_track_menu.configure(values=track_ids)
            self.eval_track_menu.set(random.choice(track_ids))
        except Exception as e:
            self.log(f"Error loading checkpoint config: {e}")
            self.eval_track_menu.configure(values=["None"])

    def update_config_from_ui(self):
        """Update the main config dictionary from the UI widgets before a run."""
        # --- READ MODEL ARCHITECTURE ---
        self.config["model_params"]["architecture_name"] = self.train_widgets[
            "architecture_name"
        ].get()

        # --- READ DATASET CONFIG (CORRECTED LOGIC) ---

        # This is the new logic to read from the checkboxes
        selected_train_datasets = [
            name
            for name, var in self.train_dataset_checkboxes.items()
            if var.get() == "on"
        ]
        self.config["data_params"][
            "train_datasets"
        ] = selected_train_datasets  # Note: plural 'datasets'

        # The old line that caused the error is no longer needed.
        # self.config['data_params']['train_dataset'] = self.train_widgets['train_dataset'].get() # REMOVED

        # --- Training Params ---
        try:
            self.config["training_params"]["learning_rate"] = float(
                self.train_widgets["learning_rate"].get()
            )
        except (ValueError, TypeError):
            pass  # Keep default if entry is empty/invalid
        try:
            self.config["training_params"]["num_epochs"] = int(
                self.train_widgets["num_epochs"].get()
            )
        except (ValueError, TypeError):
            pass
        try:
            self.config["training_params"]["batch_size"] = int(
                self.train_widgets["batch_size"].get()
            )
        except (ValueError, TypeError):
            pass

        # might want to add entries for the new step-based training parameters here too
        # Example:
        # try:
        #     self.config['training_params']['steps_per_checkpoint'] = int(self.train_widgets['steps_per_checkpoint'].get())
        # except (ValueError, TypeError): pass

        self.config["training_params"]["enable_profiler"] = (
            self.train_widgets["enable_profiler"].get() == "on"
        )

        self.log("Configuration updated from UI settings.")

    def update_tuning_config_from_ui(self):
        """Update the tuning_params in the config from the UI widgets."""
        try:
            self.config["tuning_params"]["population_size"] = int(
                self.tuning_widgets["population_size"].get()
            )
        except (ValueError, TypeError):
            pass
        try:
            self.config["tuning_params"]["num_generations"] = int(
                self.tuning_widgets["num_generations"].get()
            )
        except (ValueError, TypeError):
            pass
        try:
            self.config["tuning_params"]["epochs_per_eval"] = int(
                self.tuning_widgets["epochs_per_eval"].get()
            )
        except (ValueError, TypeError):
            pass

        self.log("Tuning configuration updated from UI settings.")

    def epoch_end_threadsafe(self, metrics):
        self.ui_queue.put(("epoch_complete", metrics))

    def log_threadsafe(self, message):
        self.after(0, self.log, message)

    def progress_threadsafe(self, current, total):
        self.after(0, lambda: self.progress_bar.set(current / total))

    def eval_progress_threadsafe(self, current, total):
        self.after(0, lambda: self.eval_progress_bar.set(current / total))

    def tune_progress_threadsafe(self, current, total):
        self.after(0, lambda: self.tune_progress_bar.set(current / total))

    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.console.insert("end", f"[{timestamp}] {message}\n")
        self.console.see("end")


# -----------------------------------------------------------------------------
# SECTION 6: MAIN EXECUTION
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="Deep Salience Studio")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="./datasets",
        help="Path to the root directory containing dataset folders (e.g., CantoriaDataset_v1.0.0).",
    )
    args = parser.parse_args()
    if not os.path.exists(args.dataset_root):
        print(f"Error: Dataset root path does not exist: {args.dataset_root}")
        sys.exit(1)

    app = SalienceStudioApp(root_dir=args.dataset_root)
    app.mainloop()
