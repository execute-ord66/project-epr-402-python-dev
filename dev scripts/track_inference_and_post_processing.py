import os
import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import find_peaks
import soundfile as sf

# ── Configuration ─────────────────────────────────────────────────────────────
file = "mary.mp3"
checkpoint_path = "candidate.pth"
should_plot = False
sr = 22050
hop_length = 256
fmin = 32.703                    # C1
harmonics = [1, 2, 3, 4, 5]
bins_per_octave = 60
n_octaves = 6
n_bins = bins_per_octave * n_octaves

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
                {"type": "conv", "filters": 16, "kernel": 5, "padding": 2},
                {"type": "conv", "filters": 16, "kernel": 5, "padding": 2},
                {"type": "conv", "filters": 16, "kernel": 5, "padding": 2},
                {"type": "conv", "filters": 16, "kernel": (69, 5), "padding": (34, 2)},
                {"type": "conv", "filters": 16, "kernel": (69, 5), "padding": (34, 2)},
                {"type": "conv_out", "filters": 1, "kernel": 1},
            ], "activation": "GELU"
        },
        "training_params": {
            "learning_rate": 0.03125, "batch_size": 22, "num_epochs": 30,
            "optimizer": "AdamW", "patch_width": 50, "patch_overlap": 0.5, "val_split_ratio": 0.1,
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

def get_new_config():
    """Returns a dictionary with the default configuration for the entire project."""
    return {
        "run_id": None,
        "data_params": {
            "sr": 22050, "hop_length": 256, "fmin": 32.703, "harmonics": [1, 2, 3, 4, 5],
            "bins_per_octave": 60, "n_octaves": 5, "gaussian_sigma": 1.0,
            "train_dataset": "Cantoria", "eval_dataset": "DCS",
        },
        "model_params": {
            "architecture_name": "SalienceNetV2",
            "input_channels": 5, # Should match len(harmonics)
            "layers": [
                {"type": "conv_in", "filters": 16, "kernel": 5, "padding": 2},
                {"type": "conv", "filters": 16, "kernel": 5, "padding": 2},
                {"type": "conv", "filters": 16, "kernel": 5, "padding": 2},
                {"type": "conv", "filters": 16, "kernel": 5, "padding": 2},
                {"type": "conv", "filters": 4, "kernel": (69, 5), "padding": (34, 2)},
                {"type": "conv_out", "filters": 1, "kernel": 1},
            ], "activation": "GELU"
        },
        "training_params": {
            "learning_rate": 2e-3, "batch_size": 22, "num_epochs": 40,
            "optimizer": "AdamW", "patch_width": 50, "patch_overlap": 0.5, "val_split_ratio": 0.1,
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

class SalienceCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_cfg = config['model_params']
        layers = []
        in_channels = model_cfg['input_channels']

        # Get the activation function from the dictionary
        activation_name = model_cfg.get("activation", "GELU")
        activation_fn = nn.GELU()

        for layer_cfg in model_cfg['layers']:
            out_channels, kernel, padding = layer_cfg['filters'], layer_cfg['kernel'], layer_cfg.get('padding', 'same')
            layers.append(nn.Conv2d(in_channels, out_channels, kernel, padding=padding))
            if layer_cfg['type'] != "conv_out" and layer_cfg['type'] != "conv_in":
                layers.extend([nn.BatchNorm2d(out_channels), activation_fn])
            in_channels = out_channels
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)
    
    def forward(self, x): return self.network(x)

model = SalienceCNN(get_new_config()).to(device)

# Load your checkpoint and pull out just the weights
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint)
# checkpoint = torch.load(checkpoint_path, map_location=device)
# model.load_state_dict(checkpoint['model_state_dict'])

model.eval()


def compute_hcqt(audio_fpath):
    y, fs = librosa.load(audio_fpath, sr=sr)
    cqt_list, shapes = [], []
    for h in harmonics:
        C = librosa.cqt(y, sr=fs, hop_length=hop_length,
                        fmin=fmin*h,
                        n_bins=bins_per_octave*n_octaves,
                        bins_per_octave=bins_per_octave)
        cqt_list.append(C); shapes.append(C.shape)
    # trim to shortest
    min_t = min(s[1] for s in shapes)
    cqt_list = [C[:, :min_t] for C in cqt_list]
    log_hcqt = (1/80.) * librosa.amplitude_to_db(
                  np.abs(np.stack(cqt_list)), ref=np.max) + 1.0
    return log_hcqt

def create_patches(hcqt_mag, patch_width=50):
    n_ch, n_b, n_f = hcqt_mag.shape
    step = patch_width // 2
    patches = []
    for st in range(0, n_f - patch_width + 1, step):
        patches.append(hcqt_mag[:, :, st:st+patch_width])
    return np.stack(patches)

# ensure input exists
if not os.path.exists(file):
    print(f"Creating dummy audio: {file}")
    dummy = np.random.randn(sr*5)
    librosa.output.write_wav(file, dummy, sr)

# ── prepare data ─────────────────────────────────────────────────────────────
hcqt = compute_hcqt(file)
patches = create_patches(hcqt)
print("Patches shape:", patches.shape)

class HcqtPatchDataset(Dataset):
    def __init__(self, patches): self.patches = patches
    def __len__(self): return len(self.patches)
    def __getitem__(self, i): 
        return torch.from_numpy(self.patches[i]).float()

ds = HcqtPatchDataset(patches)
loader = DataLoader(ds, batch_size=16, shuffle=False)

# ── run inference ─────────────────────────────────────────────────────────────
n_ch, n_bins, n_frames = hcqt.shape
pw = patches.shape[-1]
step = pw//2
total_frames = (len(patches)-1)*step + pw

out_map = np.zeros((n_bins, total_frames), dtype=np.float32)
ov_count = np.zeros_like(out_map)

with torch.no_grad():
    for batch_idx, batch in enumerate(tqdm(loader, desc="Inference")):
        batch = batch.to(device)            # (B, H, F, W)
        preds = model(batch).squeeze(1)     # (B, F, W)
        preds = preds.cpu().numpy()
        for j, p in enumerate(preds):
            start = batch_idx*loader.batch_size*step + j*step
            out_map[:, start:start+pw] += p
            ov_count[:, start:start+pw] += 1

ov_count[ov_count==0] = 1e-6
out_map /= ov_count
out_map /= out_map.max()
print("Output salience map shape:", out_map.shape)

# ── Polyphonic synthesis from multi-F0 and MIDI export ───────────────────────
def synthesize_polyphonic_with_debug(
    times, freqs, wav_out, midi_out,
    sr=16000, max_voices=4,
    smooth_ms=60.0, min_note_ms=80.0,
    max_harmonics=8, tilt_db_per_oct=-12.0, noise_level=0.000
):
    """
    End-to-end: raw multi-F0 -> voice tracking -> smooth -> quantize -> segment -> synth -> WAV+MIDI.
    Returns a `debug` dict with per-stage data for plotting.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from scipy import signal
    import soundfile as sf

    # Optional MIDI libs
    try:
        import pretty_midi as _pm
        _HAVE_PM = True
    except Exception:
        _HAVE_PM = False
    try:
        import mido as _mido
        _HAVE_MIDO = True
    except Exception:
        _HAVE_MIDO = False

    # ----------------- helpers -----------------
    def _freq_to_midi_vectorized(f):
        f = np.asarray(f, float)
        out = np.zeros_like(f, dtype=float); pos = f > 0
        out[pos] = 12.0 * np.log2(f[pos] / 440.0) + 69.0
        return out

    def _midi_to_freq_vectorized(m):
        m = np.asarray(m, float)
        out = np.zeros_like(m, dtype=float); pos = m > 0
        out[pos] = 440.0 * (2.0 ** ((m[pos] - 69.0) / 12.0))
        return out

    def _quantize_to_semitone(f):
        return _midi_to_freq_vectorized(np.round(_freq_to_midi_vectorized(f)))
    
    def _merge_same_pitch_notes(notes, merge_gap_ms=500.0, midi_tol=0):
        """
        Merge adjacent same-pitch notes if the gap between them is <= merge_gap_ms.
        Pitch equality is tested by MIDI (rounded), with optional tolerance in semitones.

        Args:
            notes: list of dicts with keys: start_time, end_time, quantized_freq
            merge_gap_ms: maximum silence between notes to merge (milliseconds)
            midi_tol: allow +/- this many semitones as 'same pitch' (0 = exact)

        Returns:
            merged list of notes (same schema)
        """
        import numpy as np

        if not notes:
            return notes

        def freq_to_midi(f):
            if f <= 0: return -np.inf
            return 12.0 * np.log2(f / 440.0) + 69.0

        gap_thr = merge_gap_ms / 1000.0

        merged = []
        prev = dict(notes[0])  # copy

        prev_midi = round(freq_to_midi(prev["quantized_freq"]))
        for n in notes[1:]:
            cur = dict(n)
            cur_midi = round(freq_to_midi(cur["quantized_freq"]))

            same_pitch = abs(cur_midi - prev_midi) <= midi_tol
            gap = cur["start_time"] - prev["end_time"]  # can be negative if overlap

            if same_pitch and gap <= gap_thr:
                # extend previous note through the current one
                prev["end_time"] = max(prev["end_time"], cur["end_time"])
                # keep prev_midi as-is
            else:
                merged.append(prev)
                prev = cur
                prev_midi = cur_midi

        merged.append(prev)
        return merged


    def _smooth_f0_contour(times_, freqs_, sr,
                       smooth_ms=20.0,
                       gap_factor=1.5,
                       hold_ms=1000.0,
                       semitone_tol=0.5,
                       bridge='hold'):
        """
        Dense smoothing for a SINGLE voice contour (sparse -> dense) with:
        • explicit unvoiced gaps (zeros), and
        • optional 'hold' bridging across short gaps (≤ hold_ms) if the next note
            is within ±semitone_tol semitones of the current note.

        Args:
            times_, freqs_: sparse detections for one voice
            sr:        sample rate for dense timeline
            smooth_ms: one-pole low-pass time constant (per voiced segment)
            gap_factor: gap is "large" if > gap_factor * median_frame_dt
            hold_ms:   max gap to bridge (milliseconds)
            semitone_tol: tolerance in semitones for pitch continuity
            bridge:    'hold' (flat) or 'interp' (linear glide) across bridged gap
        Returns:
            t_dense (s), f0_dense (Hz) with zeros in true unvoiced regions.
        """
        import numpy as np
        from scipy import signal

        times_ = np.asarray(times_, float)
        freqs_ = np.asarray(freqs_, float)

        if times_.size == 0:
            return np.array([0.0]), np.array([0.0])

        # Clean & sort
        mask = np.isfinite(times_) & np.isfinite(freqs_)
        times_, freqs_ = times_[mask], freqs_[mask]
        order = np.argsort(times_)
        times_, freqs_ = times_[order], freqs_[order]
        freqs_ = np.maximum(freqs_, 0.0)

        # Estimate nominal frame step
        if len(times_) >= 2:
            dt_med = float(np.median(np.diff(times_)))
            if not np.isfinite(dt_med) or dt_med <= 0:
                dt_med = 0.032
        else:
            dt_med = 0.032

        gap_thr = gap_factor * dt_med
        hold_thr = hold_ms / 1000.0
        cents = lambda f2, f1: 1200.0 * np.log2(max(f2, 1e-9) / max(f1, 1e-9))

        # Build augmented anchors
        t_aug = []
        f_aug = []

        # Leading zero anchor
        t0 = max(0.0, times_[0] - 0.5 * dt_med)
        t_aug.append(t0); f_aug.append(0.0)

        for i in range(len(times_)):
            t_i, f_i = float(times_[i]), float(freqs_[i])
            t_aug.append(t_i); f_aug.append(f_i)

            if i + 1 < len(times_):
                t_next, f_next = float(times_[i + 1]), float(freqs_[i + 1])
                gap = t_next - t_i

                if gap > gap_thr:
                    # Candidate for bridging?
                    same_pitch = (f_i > 0 and f_next > 0 and
                                abs(cents(f_next, f_i)) <= semitone_tol * 100.0)
                    if gap <= hold_thr and same_pitch:
                        # BRIDGE the gap
                        if bridge == 'hold':
                            # Flat hold of previous pitch until just before next
                            tL = t_i + 0.5 * dt_med
                            tR = t_next - 0.5 * dt_med
                            if tR > tL:
                                t_aug += [tL, tR]; f_aug += [f_i, f_i]
                        elif bridge == 'interp':
                            # Linear glide: no zeros; optional mid-anchor to stabilize
                            # (Leaving only endpoints will already interpolate linearly.)
                            pass
                    else:
                        # REAL silence: drop to zero inside the gap
                        tL = t_i + 0.5 * dt_med
                        tR = t_next - 0.5 * dt_med
                        if tR > tL:
                            t_aug += [tL, tR]; f_aug += [0.0, 0.0]

        # Trailing zero anchor
        tN = times_[-1] + 0.5 * dt_med
        t_aug.append(tN); f_aug.append(0.0)

        t_aug = np.asarray(t_aug, float)
        f_aug = np.asarray(f_aug, float)

        # Dense timeline
        T = float(max(t_aug.max(), times_.max()) + 0.25)
        N = int(np.ceil(T * sr))
        t_dense = np.arange(N) / sr

        # Interpolate; zeros persist where we placed 0-Hz anchors
        f0 = np.interp(t_dense, t_aug, f_aug, left=0.0, right=0.0)

        # Smooth ONLY within voiced segments
        if smooth_ms and smooth_ms > 0:
            fc = 1.0 / (smooth_ms / 1000.0)
            w = 2 * np.pi * fc / sr
            alpha = w / (1 + w)
            b, a = np.array([alpha]), np.array([1.0, -(1.0 - alpha)])

            voiced = f0 > 0.0
            v = voiced.astype(np.int8)
            dv = np.diff(v, prepend=0, append=0)
            starts = np.where(dv == 1)[0]
            ends   = np.where(dv == -1)[0]

            f0_s = f0.copy()
            for s, e in zip(starts, ends):
                if e > s:
                    f0_s[s:e] = signal.lfilter(b, a, f0[s:e])
            f0 = f0_s

        return t_dense, f0.astype(float)


    def _segment_notes(t_dense, f0_q, sr, min_duration_ms=80.0):
        stable, start = [], -1
        min_len = int((min_duration_ms/1000.0) * sr)
        for i in range(1, len(f0_q)):
            if f0_q[i] > 0 and start == -1:
                start = i
            elif start != -1 and f0_q[i] != f0_q[start]:
                if (i - start) >= min_len:
                    stable.append({'start_time': t_dense[start],
                                   'end_time': t_dense[i],
                                   'quantized_freq': f0_q[start]})
                start = i if f0_q[i] > 0 else -1
        if start != -1 and (len(f0_q) - start) >= min_len:
            stable.append({'start_time': t_dense[start],
                           'end_time': t_dense[-1],
                           'quantized_freq': f0_q[start]})
        # merge micro-gaps
        merged = []
        if stable:
            prev = stable[0]
            for n in stable[1:]:
                if (abs(n['quantized_freq'] - prev['quantized_freq']) < 1e-3) and \
                   (n['start_time'] - prev['end_time'] <= 0.03):
                    prev['end_time'] = n['end_time']
                else:
                    merged.append(prev); prev = n
            merged.append(prev)
        return merged

    def _synthesize_mono_from_notes(notes, sr=16000, max_harmonics=8, tilt_db_per_oct=-12.0, noise_level=0.004):
        if not notes:
            return np.zeros(sr, dtype=np.float32)
        T = max(n["end_time"] for n in notes) + 0.25
        N = int(np.ceil(T * sr))
        t = np.arange(N) / sr
        f0 = np.zeros(N, dtype=float)
        for n in notes:
            s = int(np.floor(n["start_time"] * sr))
            e = int(np.floor(n["end_time"] * sr))
            f0[s:e] = n["quantized_freq"]
        # shape edges
        edge_len = max(1, int(0.008 * sr))
        env = np.ones_like(f0)
        changes = np.where(np.diff(f0) != 0)[0]
        for idx in changes:
            lo = max(0, idx - edge_len + 1)
            if f0[idx] > 0:
                env[lo:idx + 1] *= np.linspace(1.0, 0.0, idx - lo + 1)
            hi = min(len(f0) - 1, idx + edge_len)
            if f0[idx + 1] > 0:
                env[idx + 1:hi + 1] *= np.linspace(0.0, 1.0, hi - idx)
        # additive with anti-alias
        phase = np.cumsum(2 * np.pi * f0 / sr)
        k = np.arange(1, max_harmonics + 1)[:, None]
        gains = (1.0 / k) * (10.0 ** (tilt_db_per_oct * np.log2(k) / 20.0))
        with np.errstate(divide='ignore', invalid='ignore'):
            max_k_inst = np.floor((sr / 2) / np.maximum(f0, 1e-9))
        alias = (k <= max_k_inst)
        y_h = np.sum(np.sin(k * phase) * gains * alias, axis=0)
        # gentle noise in rests
        noise = np.random.randn(N)
        b, a = signal.butter(1, 1200.0 / (sr / 2), btype='highpass')
        noise_hp = signal.lfilter(b, a, noise)
        y = np.where(f0 > 0, y_h * env, noise_level * noise_hp)
        peak = np.max(np.abs(y))
        if peak > 0: y = 0.95 * y / peak
        return y.astype(np.float32)

    # ----------------- Stage 0: raw -----------------
    times = np.asarray(times, float); freqs = np.asarray(freqs, float)
    debug = {"raw": {"times": times, "freqs": freqs}}

    # ----------------- Stage 1: frame + voice tracking -----------------
    mask = np.isfinite(times) & np.isfinite(freqs)
    times, freqs = times[mask], freqs[mask]
    order = np.lexsort((freqs, times)); times, freqs = times[order], freqs[order]

    uniq_times, idx_starts = np.unique(times, return_index=True)
    f_by_frame = []
    for k in range(len(uniq_times)):
        i0 = idx_starts[k]
        i1 = idx_starts[k + 1] if k + 1 < len(idx_starts) else len(times)
        fs = freqs[i0:i1]
        fs = fs[np.isfinite(fs) & (fs > 0)]
        f_by_frame.append(np.array(fs, float) if fs.size else np.array([], float))

    cents = lambda f1, f2: 1200.0 * np.log2(np.maximum(f1, 1e-9) / np.maximum(f2, 1e-9))
    V = max_voices
    voice_times = [[] for _ in range(V)]
    voice_freqs = [[] for _ in range(V)]
    last_freq = np.zeros(V, dtype=float)
    max_cents_jump = 300.0
    big_penalty = 1e6

    for t_k, obs in zip(uniq_times, f_by_frame):
        if obs.size == 0:
            continue
        if obs.size > V:
            if np.any(last_freq > 0):
                d = []
                for f in obs:
                    active = last_freq[last_freq > 0]
                    d.append(np.min(np.abs(cents(f, active))) if active.size else 0.0)
                obs = obs[np.argsort(d)[:V]]
            else:
                obs = np.sort(obs)[:V]
        O = len(obs)
        C = np.full((V, O), big_penalty, dtype=float)
        for v in range(V):
            lf = last_freq[v]
            if lf <= 0:
                C[v, :] = 200.0
            else:
                dist = np.abs(cents(obs, lf))
                C[v, :] = dist + np.maximum(0.0, dist - max_cents_jump) * 10.0
        row_ind, col_ind = linear_sum_assignment(C)
        assigned = [None] * V
        for v, j in zip(row_ind, col_ind):
            if j < O and C[v, j] < big_penalty * 0.5:
                assigned[v] = obs[j]
        for v in range(V):
            f_new = assigned[v]
            if f_new is not None and np.isfinite(f_new) and f_new > 0:
                voice_times[v].append(t_k)
                voice_freqs[v].append(float(f_new))
                last_freq[v] = float(f_new)

    packed = []
    for v in range(V):
        t_v = np.array(voice_times[v], float)
        f_v = np.array(voice_freqs[v], float)
        if t_v.size > 0:
            packed.append((t_v, f_v))
    debug["voices_tracked"] = [{"times": t, "freqs": f} for (t, f) in packed]

    # ----------------- Stages 2-4: smooth, quantize, segment -----------------
    voice_notes = []
    voice_debug = []
    for (t_v, f_v) in packed:
        t_dense, f0_smooth = _smooth_f0_contour(t_v, f_v, sr, smooth_ms=smooth_ms)
        f0_quant = _quantize_to_semitone(f0_smooth)
        notes = _segment_notes(t_dense, f0_quant, sr, min_duration_ms=min_note_ms)
        notes = _merge_same_pitch_notes(notes, merge_gap_ms=500.0, midi_tol=0)
        voice_notes.append(notes)
        voice_debug.append({
            "t_dense": t_dense, "f0_smooth": f0_smooth,
            "f0_quant": f0_quant, "notes": notes
        })
    debug["voices_processed"] = voice_debug

    # ----------------- Stage 5: synth + mix + write WAV -----------------
    if not voice_notes:
        sf.write(wav_out, np.zeros(sr, dtype=np.float32), sr)
    else:
        voice_wavs = []
        max_len = 0
        for notes in voice_notes:
            y = _synthesize_mono_from_notes(notes, sr=sr,
                                            max_harmonics=max_harmonics,
                                            tilt_db_per_oct=tilt_db_per_oct,
                                            noise_level=noise_level)
            voice_wavs.append(y)
            max_len = max(max_len, len(y))
        mix = np.zeros(max_len, dtype=np.float32)
        for y in voice_wavs:
            mix[:len(y)] += y
        peak = np.max(np.abs(mix))
        if peak > 0:
            mix = 0.98 * mix / peak
        sf.write(wav_out, mix, sr)

    # ----------------- MIDI writing -----------------
    if _HAVE_PM:
        pm = _pm.PrettyMIDI()
        for vi, notes in enumerate(voice_notes):
            inst = _pm.Instrument(program=_pm.instrument_name_to_program('Acoustic Grand Piano'))
            for n in notes:
                f = float(n['quantized_freq'])
                if f <= 0: continue
                midi_num = int(np.clip(np.round(12*np.log2(f/440.0)+69), 0, 127))
                inst.notes.append(_pm.Note(velocity=80, pitch=midi_num,
                                           start=float(n['start_time']),
                                           end=float(n['end_time'])))
            pm.instruments.append(inst)
        pm.write(midi_out)
    elif _HAVE_MIDO:
        mid = _mido.MidiFile(type=1)
        track = _mido.MidiTrack(); mid.tracks.append(track)
        ticks_per_beat = mid.ticks_per_beat
        tempo = _mido.bpm2tempo(120)
        track.append(_mido.MetaMessage('set_tempo', tempo=tempo, time=0))
        def sec_to_ticks(sec): return int(_mido.second2tick(sec, ticks_per_beat, tempo))
        events = []
        for notes in voice_notes:
            for n in notes:
                f = float(n['quantized_freq'])
                if f <= 0: continue
                midi_num = int(np.clip(np.round(12*np.log2(f/440.0)+69), 0, 127))
                events.append(('on',  float(n['start_time']), midi_num))
                events.append(('off', float(n['end_time']),   midi_num))
        events.sort(key=lambda x: x[1])
        last = 0
        for kind, t_sec, pitch in events:
            t_ticks = sec_to_ticks(t_sec)
            dt = max(0, t_ticks - last); last = t_ticks
            if kind == 'on':
                track.append(_mido.Message('note_on', note=pitch, velocity=80, time=dt))
            else:
                track.append(_mido.Message('note_off', note=pitch, velocity=0, time=dt))
        mid.save(midi_out)
    else:
        print("MIDI not written (install pretty_midi or mido).")

    return debug

# --- Build (times, freqs) from your salience map via peak picking ------------
def extract_multi_f0_stream(
    sal_map,
    fmin,
    bins_per_octave,
    hop_length,
    sr,
    peak_thresh=0.4,
    max_peaks=4,
):
    """
    Convert salience map (n_bins, n_frames) to a flat (times, freqs) stream.
    For each frame:
      1) find peaks >= peak_thresh
      2) keep top `max_peaks` by height
      3) refine each peak's freq by a weighted average over a 3-bin neighborhood
         around the peak (bins [peak-1, peak, peak+1], clipped to valid range).
    Returns:
      times: (K,) seconds (multiple entries can share the same timestamp)
      freqs: (K,) Hz (refined freq per entry)
    """
    import numpy as np
    from scipy.signal import find_peaks
    import librosa

    n_bins, n_frames = sal_map.shape

    # Center frequencies for each CQT bin
    freqs_per_bin = librosa.cqt_frequencies(
        n_bins=n_bins,
        fmin=fmin,
        bins_per_octave=bins_per_octave,
    )

    # Frame times (seconds)
    times_per_frame = librosa.times_like(
        sal_map,
        sr=sr,
        hop_length=hop_length
    )

    out_times, out_freqs = [], []

    for t_idx in range(n_frames):
        col = sal_map[:, t_idx]

        # 1) peaks above threshold
        peaks, props = find_peaks(col, height=peak_thresh)
        if peaks.size == 0:
            continue

        # 2) top-N by salience
        peak_heights = props['peak_heights']
        keep = np.argsort(peak_heights)[::-1][:max_peaks]
        top_peaks = peaks[keep]

        # 3) weighted 3-bin refinement around each peak
        for p in top_peaks:
            start_bin = max(0, p - 1)
            end_bin = min(n_bins, p + 2)  # slice end is exclusive
            sal_win = col[start_bin:end_bin]
            freq_win = freqs_per_bin[start_bin:end_bin]

            ssum = float(np.sum(sal_win))
            if ssum > 1e-9:
                weights = sal_win / ssum
                refined = float(np.sum(weights * freq_win))
            else:
                refined = float(freqs_per_bin[p])

            out_times.append(times_per_frame[t_idx])
            out_freqs.append(refined)

    return np.asarray(out_times, dtype=float), np.asarray(out_freqs, dtype=float)


synth_sr = 16000          # synthesis sample rate (you can change if you like)
peak_thresh = 0.55         # detection threshold (tune as desired)
max_voices = 8

# 1) Build the refined peak stream from your salience map:
times_f0, freqs_f0 = extract_multi_f0_stream(
    out_map, fmin, bins_per_octave, hop_length, sr=22050,
    peak_thresh=peak_thresh, max_peaks=max_voices
)

# 2) Run synthesis with debug capture (also writes output.wav + output.mid)
debug = synthesize_polyphonic_with_debug(
    times_f0, freqs_f0,
    wav_out="output.wav",
    midi_out="output.mid",
    sr=synth_sr,
    max_voices=max_voices
)



# ── plotting ────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt

def _freq_to_midi(f):
    f = np.asarray(f, float)
    out = np.zeros_like(f, dtype=float)
    pos = f > 0
    out[pos] = 12.0 * np.log2(f[pos] / 440.0) + 69.0
    return out

def plot_raw_multi_f0(times, freqs, y_axis='midi'):
    """
    Scatter of the raw extracted multi-F0 stream.
    y_axis: 'midi' or 'hz'
    """
    if y_axis == 'midi':
        y = _freq_to_midi(freqs)
        plt.figure()
        plt.scatter(times, y, s=6)
        plt.xlabel("Time (s)")
        plt.ylabel("Pitch (MIDI)")
        plt.title("Raw multi-F0 (weighted peaks)")
        plt.grid(True, alpha=0.3)
    else:
        plt.figure()
        plt.scatter(times, freqs, s=6)
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.title("Raw multi-F0 (weighted peaks)")
        plt.grid(True, alpha=0.3)
    plt.show()

def plot_voice_tracks_smooth(debug, y_axis='midi'):
    """
    One line per voice showing the smoothed continuous f0 contour.
    """
    plt.figure()
    for v in debug.get("voices_processed", []):
        t = v["t_dense"]; f = v["f0_smooth"]
        y = _freq_to_midi(f) if y_axis == 'midi' else f
        plt.plot(t, y, linewidth=1.0)
    plt.xlabel("Time (s)")
    plt.ylabel("Pitch (MIDI)" if y_axis=='midi' else "Frequency (Hz)")
    plt.title("Voice-tracked smoothed contours")
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_voice_tracks_quantized(debug, y_axis='midi'):
    """
    One (step-like) line per voice for semitone-quantized contour.
    """
    plt.figure()
    for v in debug.get("voices_processed", []):
        t = v["t_dense"]; f = v["f0_quant"]
        y = _freq_to_midi(f) if y_axis == 'midi' else f
        # step-ish: plotting dense quantized contour gives visual steps
        plt.plot(t, y, linewidth=1.0)
    plt.xlabel("Time (s)")
    plt.ylabel("Pitch (MIDI)" if y_axis=='midi' else "Frequency (Hz)")
    plt.title("Semitone-quantized contours")
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_segmented_notes(debug, y_axis='midi'):
    """
    Horizontal bars for segmented notes per voice.
    """
    plt.figure()
    for vi, v in enumerate(debug.get("voices_processed", [])):
        for n in v["notes"]:
            f = n["quantized_freq"]
            y = _freq_to_midi(f)[()] if y_axis == 'midi' else f
            plt.hlines(y, n["start_time"], n["end_time"], linewidth=2.0)
    plt.xlabel("Time (s)")
    plt.ylabel("Pitch (MIDI)" if y_axis=='midi' else "Frequency (Hz)")
    plt.title("Segmented notes (per voice)")
    plt.grid(True, alpha=0.3)
    plt.show()


if should_plot:
    # 3) Plot each stage (each call creates its own figure)
    plot_raw_multi_f0(debug["raw"]["times"], debug["raw"]["freqs"], y_axis='midi')
    # plot_voice_tracks_smooth(debug, y_axis='midi')
    # plot_voice_tracks_quantized(debug, y_axis='midi')
    plot_segmented_notes(debug, y_axis='midi')

print("Done.")
