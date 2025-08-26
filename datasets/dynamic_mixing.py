import os
import random
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import soundfile as sf
from torch.utils.data import IterableDataset

from dataprocessor import precompute_rms_env

try:
    from audiomentations import (
        Compose,
        AddGaussianNoise,
        ClippingDistortion,
        PolarityInversion,
    )
except Exception:  # pragma: no cover - optional dependency
    Compose = None


class FileCache:
    """Persistent LRU cache for preprocessed audio."""

    def __init__(self, max_items=256, dp_config=None, log_callback=print):
        self.max_items = max_items
        self.dp = dp_config or {}
        self.log = log_callback
        self.processed_audio_cache: OrderedDict[str, tuple] = OrderedDict()

    def get_processed_audio(self, audio_path: str):
        if audio_path in self.processed_audio_cache:
            self.processed_audio_cache.move_to_end(audio_path)
            return self.processed_audio_cache[audio_path]
        try:
            y, file_sr = sf.read(audio_path, dtype="float32", always_2d=False)
            if y.ndim > 1:
                y = np.mean(y, axis=1)
            if file_sr != self.dp.get("sr"):
                import resampy  # lazy import

                y = resampy.resample(y, file_sr, self.dp["sr"])
            rms_hop = self.dp.get("rms_hop", 256)
            rms_frame = self.dp.get("rms_frame", 1024)
            rms_env = precompute_rms_env(
                y, frame_length=rms_frame, hop_length=rms_hop
            )
            if len(self.processed_audio_cache) >= self.max_items:
                self.processed_audio_cache.popitem(last=False)
            self.processed_audio_cache[audio_path] = (y, rms_env)
            return y, rms_env
        except Exception as e:  # pragma: no cover - I/O errors
            if self.log:
                self.log(f"ERROR caching {os.path.basename(audio_path)}: {e}")
            return None, None


class DynamicMixingDataset(IterableDataset):
    """On-the-fly audio mixing with optional augmentation."""

    def __init__(self, stem_info_list, config, log_callback=print):
        super().__init__()
        self.stem_info_list = stem_info_list
        self.config = config
        self.dp = config["data_params"]
        self.tp = config["training_params"]
        self.segment_len_samples = int(
            self.tp.get("segment_duration_sec", 2.0) * self.dp["sr"]
        )
        self.pool_size = self.tp.get("pool_size", 128)
        self.steps_per_epoch = self.tp.get("steps_per_epoch", 500)
        self.stems_per_mix = self.tp["stems_per_mix"]
        self.loudness_thresh = self.tp["loudness_threshold_db"]

        self.log = log_callback
        self.cache: FileCache | None = None
        self.executor: ThreadPoolExecutor | None = None

        if Compose is not None:
            self.augment = Compose(
                [
                    AddGaussianNoise(p=0.3),
                    ClippingDistortion(p=0.2),
                    PolarityInversion(p=0.1),
                ]
            )
        else:  # pragma: no cover - augmentation library missing
            self.augment = None

    def _initialize_worker(self):
        self.cache = FileCache(
            max_items=self.pool_size, dp_config=self.dp, log_callback=self.log
        )
        self.executor = ThreadPoolExecutor(max_workers=4)

    # ------------------------------------------------------------------
    def _load_random_segment(self):
        for _ in range(20):
            stem_info = random.choice(self.stem_info_list)
            audio_path = stem_info["audio_path"]
            y, rms_env = self.cache.get_processed_audio(audio_path)
            if y is None or len(y) < self.segment_len_samples:
                continue
            rms_db = 20 * np.log10(np.maximum(rms_env, 1e-7))
            good = np.where(rms_db > self.loudness_thresh)[0]
            if good.size == 0:
                continue
            frame = random.choice(good)
            rms_hop = self.dp.get("rms_hop", 256)
            start = frame * rms_hop
            if start + self.segment_len_samples > len(y):
                start = len(y) - self.segment_len_samples
            segment = y[start : start + self.segment_len_samples]
            f0_df = pd.read_csv(stem_info["f0_path"])
            f0_df.columns = [c.strip() for c in f0_df.columns]
            start_time = start / self.dp["sr"]
            end_time = (start + self.segment_len_samples) / self.dp["sr"]
            pad = 0.1
            f0_seg = f0_df[
                (f0_df["time"] >= start_time - pad)
                & (f0_df["time"] <= end_time + pad)
            ]
            if f0_seg.empty:
                f0_times, f0_freqs = np.array([]), np.array([])
            else:
                f0_times = f0_seg["time"].values - start_time
                f0_freqs = f0_seg["frequency"].values
            return segment, (f0_times, f0_freqs)
        self.log(
            "Warning: Could not find a valid segment after many attempts. Yielding silence."
        )
        return np.zeros(self.segment_len_samples, dtype=np.float32), (
            np.array([]),
            np.array([]),
        )

    # ------------------------------------------------------------------
    def __iter__(self):
        if self.cache is None:
            self._initialize_worker()
        assert self.executor is not None

        futures = [
            self.executor.submit(self._load_random_segment)
            for _ in range(self.pool_size)
        ]
        pool = [f.result() for f in futures]

        for _ in range(self.steps_per_epoch):
            stems = random.sample(pool, self.stems_per_mix)
            # refill pool asynchronously
            refill = [
                self.executor.submit(self._load_random_segment)
                for _ in range(self.stems_per_mix)
            ]
            for fut in refill:
                pool[random.randint(0, self.pool_size - 1)] = fut.result()

            mixed = np.zeros(self.segment_len_samples, dtype=np.float32)
            f0_list = []
            for audio, f0 in stems:
                gain = 10.0 ** (random.uniform(-3.0, 3.0) / 20.0)
                mixed += audio * gain
                f0_list.append(f0)
            if self.augment is not None:
                mixed = self.augment(samples=mixed, sample_rate=self.dp["sr"])
            yield mixed, f0_list
