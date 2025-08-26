import os
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
from torch.utils.data import Dataset

try:  # optional dependency
    from audiomentations import (
        Compose,
        AddGaussianNoise,
        ClippingDistortion,
        PolarityInversion,
    )
except Exception:  # pragma: no cover - augmentation library may be missing
    Compose = None


class AugmentedSalienceDataset(Dataset):
    """Dataset that mixes random stems on-the-fly with lightweight augmentation.

    This dataset replaces the previous ``DynamicMixingDataset`` which relied on
    an expensive :class:`FileCache` and threading machinery.  The new
    implementation follows a much simpler strategy inspired by the dataset
    pipeline used in ZFTurbo's source separation code.  Each ``__getitem__``
    call loads small random segments directly from disk using ``soundfile`` and
    performs mixing/augmentation in-memory without any persistent caching.

    The dataset returns a tuple ``(mixture, f0_list)`` where ``mixture`` is a
    mono audio segment and ``f0_list`` is a list of ``(times, freqs)`` arrays for
    each contributing stem.  Downstream code is responsible for converting these
    into HCQT features and salience maps; the salience map generation logic
    remains unchanged and lives in :mod:`dataprocessor`.
    """

    def __init__(self, stem_info_list: List[dict], config: dict, log_callback=print):
        super().__init__()
        self.stems = []
        self.log = log_callback
        self.config = config
        self.dp = config["data_params"]
        self.tp = config["training_params"]

        self.sr = self.dp["sr"]
        self.segment_len = int(self.tp.get("segment_duration_sec", 2.0) * self.sr)
        self.steps_per_epoch = self.tp.get("steps_per_epoch", 500)
        self.stems_per_mix = self.tp.get("stems_per_mix", 2)
        self.loudness_thresh = self.tp.get("loudness_threshold_db", -40.0)

        # Pre-load lightweight metadata: audio lengths and F0 CSVs
        for stem in stem_info_list:
            try:
                info = sf.info(stem["audio_path"])
                length = int(info.frames)
            except Exception:
                length = 0
            try:
                f0_df = pd.read_csv(stem["f0_path"])
                f0_df.columns = [c.strip() for c in f0_df.columns]
                times = f0_df["time"].values.astype(np.float32)
                freqs = f0_df["frequency"].values.astype(np.float32)
            except Exception:
                times = np.zeros(0, dtype=np.float32)
                freqs = np.zeros(0, dtype=np.float32)
            self.stems.append(
                {
                    "audio_path": stem["audio_path"],
                    "times": times,
                    "freqs": freqs,
                    "length": length,
                }
            )

        # Simple augmentation chain (all optional)
        if Compose is not None:
            self.augment = Compose(
                [
                    AddGaussianNoise(p=0.3),
                    ClippingDistortion(p=0.2),
                    PolarityInversion(p=0.1),
                ]
            )
        else:  # pragma: no cover
            self.augment = None

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.steps_per_epoch

    # ------------------------------------------------------------------
    def _load_segment(self, stem: dict) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Load a random loud segment from ``stem``.

        The method attempts several random offsets until the RMS of the segment
        exceeds ``self.loudness_thresh``.  If no valid segment is found a silent
        segment is returned.
        """

        max_offset = max(0, stem["length"] - self.segment_len)
        for _ in range(20):
            offset = random.randint(0, max_offset) if max_offset > 0 else 0
            audio, _ = sf.read(
                stem["audio_path"],
                start=offset,
                frames=self.segment_len,
                dtype="float32",
                always_2d=False,
            )
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            if len(audio) < self.segment_len:
                audio = np.pad(audio, (0, self.segment_len - len(audio)))
            rms = np.sqrt(np.mean(audio ** 2) + 1e-9)
            db = 20 * np.log10(rms)
            if db > self.loudness_thresh:
                break
        start_time = offset / self.sr
        end_time = start_time + self.segment_len / self.sr
        times = stem["times"]
        freqs = stem["freqs"]
        mask = (times >= start_time) & (times <= end_time)
        return audio.astype(np.float32), (times[mask] - start_time, freqs[mask])

    # ------------------------------------------------------------------
    def __getitem__(self, index: int):  # index is ignored; dataset is random
        chosen = random.sample(self.stems, self.stems_per_mix)
        mixture = np.zeros(self.segment_len, dtype=np.float32)
        f0_list: List[Tuple[np.ndarray, np.ndarray]] = []
        for stem in chosen:
            audio, f0 = self._load_segment(stem)
            gain = 10.0 ** (random.uniform(-3.0, 3.0) / 20.0)
            mixture += audio * gain
            f0_list.append(f0)
        if self.augment is not None:
            mixture = self.augment(samples=mixture, sample_rate=self.sr)
        return mixture, f0_list
