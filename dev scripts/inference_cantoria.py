import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter
import mir_eval
import argparse
import random

# ─── Configuration & Dataset Discovery ─────────────────────────────────────────

def find_valid_tracks(root_dir):
    audio_dir = os.path.join(root_dir, "Audio")
    crepe_dir = os.path.join(root_dir, "F0_crepe")
    pyin_dir = os.path.join(root_dir, "F0_pyin")
    valid_tracks = {}
    voice_parts = ['S', 'A', 'T', 'B']
    if not os.path.exists(audio_dir): return {}
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    print(audio_files[100:120])
    track_bases = set(f[:-6] for f in audio_files if f.startswith('Cantoria_') and f[-5] in voice_parts)
    for base in sorted(list(track_bases)):
        if all(os.path.exists(os.path.join(d, f"{base}_{p}{ext}"))
               for p in voice_parts for d, ext in [(audio_dir, '.wav'), (crepe_dir, '.csv'), (pyin_dir, '.csv')]):
            valid_tracks[base] = [f"{base}_{p}" for p in voice_parts]
    return valid_tracks

# ─── Model & Inference Logic (from previous script) ───────────────────────────

class SalienceCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(5, 32, kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.bn3   = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.bn4   = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=(69,5), padding=(34,2))
        self.bn5   = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=(69,5), padding=(34,2))
        self.bn8   = nn.BatchNorm2d(32)
        self.conv8 = nn.Conv2d(32, 1, kernel_size=1)
        self.gelu  = nn.GELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.gelu(self.conv1(x))
        x = self.bn2(x);   x = self.gelu(self.conv2(x))
        x = self.bn3(x);   x = self.gelu(self.conv3(x))
        x = self.bn4(x);   x = self.gelu(self.conv4(x))
        x = self.bn5(x);   x = self.gelu(self.conv5(x))
        x = self.bn8(x);   x = self.sigmoid(self.conv8(x))
        return x

def get_model(checkpoint_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SalienceCNN().to(device)
    try:
        # Adjusting for the new checkpoint format from your training script
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else: # For final model saves
            model.load_state_dict(checkpoint)
        print(f"Successfully loaded model from: {checkpoint_path}")
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {checkpoint_path}"); exit()
    model.eval()
    return model

def run_inference_on_mix(model, audio_mix, config):
    # 1. Compute HCQT on the provided audio mix
    cqt_list = []
    for h in config['harmonics']:
        C = librosa.cqt(audio_mix, sr=config['sr'], hop_length=config['hop_length'],
                        fmin=config['fmin'] * h, n_bins=config['n_bins'],
                        bins_per_octave=config['bins_per_octave'])
        cqt_list.append(C)
    min_t = min(c.shape[1] for c in cqt_list)
    cqt_list = [c[:, :min_t] for c in cqt_list]
    log_hcqt = (1 / 80.) * librosa.amplitude_to_db(np.abs(np.stack(cqt_list)), ref=np.max) + 1.0

    # 2. Create Patches
    patch_width = 50
    n_ch, n_b, n_f = log_hcqt.shape
    step = patch_width // 2
    patches = np.stack([log_hcqt[:, :, st:st+patch_width] for st in range(0, n_f - patch_width + 1, step)])
    
    # 3. Dataloader & Inference
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(patches).float())
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    total_frames = (len(patches) - 1) * step + patch_width
    out_map = np.zeros((config['n_bins'], total_frames), dtype=np.float32)
    ov_count = np.zeros_like(out_map)

    with torch.no_grad():
        for i, (batch,) in enumerate(tqdm(loader, desc="Model Inference")):
            batch = batch.to(next(model.parameters()).device)
            preds = model(batch).squeeze(1).cpu().numpy()
            for j, p in enumerate(preds):
                start_frame = i * loader.batch_size * step + j * step
                out_map[:, start_frame:start_frame + patch_width] += p
                ov_count[:, start_frame:start_frame + patch_width] += 1
    
    ov_count[ov_count == 0] = 1e-6
    out_map = (out_map / ov_count) / (out_map.max() + 1e-6)
    return out_map

# ─── Ground Truth Generation (from your training script) ─────────────────────

def generate_ground_truth(track_group, root_dir, config):
    """
    Processes an SATB track group to create a mixed audio file and
    the corresponding ground truth reference pitches.
    """
    audio_dir = os.path.join(root_dir, "Audio")
    crepe_dir = os.path.join(root_dir, "F0_crepe")
    pyin_dir = os.path.join(root_dir, "F0_pyin")

    # ... (the beginning of the function is the same) ...
    all_y = []
    print(f"Creating audio mix for: {', '.join(track_group)}")
    for track in track_group:
        audio_path = os.path.join(audio_dir, track + ".wav")
        y, _ = librosa.load(audio_path, sr=config['sr'])
        all_y.append(y)
    max_len = max(len(y) for y in all_y)
    y_mix = np.sum([np.pad(y, (0, max_len - len(y))) for y in all_y], axis=0)
    y_mix /= (np.max(np.abs(y_mix)) + 1e-6)

    n_frames = int(np.ceil(len(y_mix) / config['hop_length']))
    frame_times = librosa.frames_to_time(np.arange(n_frames), sr=config['sr'], hop_length=config['hop_length'])
    
    ref_times_list, ref_freqs_list = [], []
    
    print("Generating ground truth from stems...")
    for track in track_group:
        # ... (the loop content is the same) ...
        crepe_path = os.path.join(crepe_dir, track + ".csv")
        pyin_path = os.path.join(pyin_dir, track + ".csv")
        crepe_df = pd.read_csv(crepe_path); crepe_df.columns = crepe_df.columns.str.strip()
        pyin_df = pd.read_csv(pyin_path, header=None, names=['time', 'frequency', 'confidence'])
        pyin_mask = pyin_df.frequency > 0
        interp_freqs = np.interp(frame_times, crepe_df['time'], crepe_df['frequency'], left=0.0, right=0.0)
        interp_voiced = np.interp(frame_times, pyin_df['time'], (pyin_mask).astype(float), left=0.0, right=0.0)
        active_mask = interp_voiced > 0.5
        ref_times_list.extend(frame_times[active_mask])
        ref_freqs_list.extend(interp_freqs[active_mask])

    # --- FIX IS HERE ---
    # Convert lists to NumPy arrays
    ref_times = np.array(ref_times_list)
    ref_freqs = np.array(ref_freqs_list)
    
    # Get the indices that would sort the time array
    sort_indices = np.argsort(ref_times)
    
    # Apply these indices to both arrays to sort them together
    sorted_ref_times = ref_times[sort_indices]
    sorted_ref_freqs = ref_freqs[sort_indices]
    
    return y_mix, sorted_ref_times, sorted_ref_freqs

# ─── Post-Processing, Evaluation & Plotting ───────────────────────────────────

def extract_pitches_from_salience(salience_map, config, threshold=0.3):
    """Performs peak-picking on the salience map."""
    times = librosa.times_like(salience_map, sr=config['sr'], hop_length=config['hop_length'])
    freq_bins = librosa.cqt_frequencies(n_bins=config['n_bins'], fmin=config['fmin'], bins_per_octave=config['bins_per_octave'])
    
    est_times, est_freqs = [], []
    for t_idx in range(salience_map.shape[1]):
        frame_salience = salience_map[:, t_idx]
        peaks, _ = find_peaks(frame_salience, height=threshold)
        if peaks.size > 0:
            est_times.extend([times[t_idx]] * len(peaks))
            est_freqs.extend(freq_bins[peaks])
    return np.array(est_times), np.array(est_freqs)

def _to_mir_eval_sequence(times, freqs, atol=1e-6):
    """
    Convert event lists (one (time,freq) per pitch) into the format expected by
    mir_eval.multipitch: a sorted array of unique times and a list of 1-D arrays
    of frequencies active at each time.
    """
    times = np.asarray(times, dtype=float)
    freqs = np.asarray(freqs, dtype=float)
    if times.size == 0:
        return np.asarray([]), []

    # Sort by time to make grouping stable
    order = np.argsort(times)
    times, freqs = times[order], freqs[order]

    out_times = []
    out_freqs = []
    cur_t = times[0]
    cur_freqs = [freqs[0]]

    for t, f in zip(times[1:], freqs[1:]):
        if abs(t - cur_t) <= atol:          # same frame (within tolerance)
            cur_freqs.append(f)
        else:
            out_times.append(cur_t)
            out_freqs.append(np.asarray(cur_freqs, dtype=float))
            cur_t = t
            cur_freqs = [f]

    # flush last group
    out_times.append(cur_t)
    out_freqs.append(np.asarray(cur_freqs, dtype=float))

    return np.asarray(out_times, dtype=float), out_freqs


def evaluate_and_plot(ref_times, ref_freqs, est_times, est_freqs, track_group_name):
    """Runs mir_eval and generates the final comparison plot."""
    print("\n--- MIREX Multipitch Evaluation Results ---")

    # --- FIX IS HERE ---
    # Defensively ensure all inputs are at least 1-dimensional arrays.
    # This handles the edge case where a function might return a single scalar.
    ref_times = np.atleast_1d(ref_times)
    ref_freqs = np.atleast_1d(ref_freqs)
    est_times = np.atleast_1d(est_times)
    est_freqs = np.atleast_1d(est_freqs)

    # Wrap each frequency in its own array (empty if no pitch)
    ref_time_seq, ref_freqs_seq = _to_mir_eval_sequence(ref_times, ref_freqs)
    est_time_seq, est_freqs_seq = _to_mir_eval_sequence(est_times, est_freqs)
    
    # Now, the evaluation can proceed safely.
    if est_time_seq.size == 0 or ref_time_seq.size == 0:
        if est_time_seq.size == 0: print("No pitches were predicted.")
        if ref_time_seq.size == 0: print("No reference pitches were found.")
        scores = {'Precision': 0, 'Recall': 0, 'F-measure': 0}
    else:
        assert all((isinstance(a, np.ndarray) and a.ndim == 1) for a in ref_freqs_seq)
        assert all((isinstance(a, np.ndarray) and a.ndim == 1) for a in est_freqs_seq)

        scores = mir_eval.multipitch.evaluate(ref_time_seq, ref_freqs_seq,
                                              est_time_seq, est_freqs_seq)
        print(scores)


    # print(f"Precision: {scores['Precision']:.3f}, Recall: {scores['Recall']:.3f}, F-measure: {scores['F-measure']:.3f}")

    plt.figure(figsize=(18, 8))
    # Check for empty arrays before plotting to avoid errors
    if ref_times.size > 0:
        plt.scatter(ref_times, ref_freqs, c='black', marker='.', s=50, label='Reference (All Stems)', zorder=1)
    if est_times.size > 0:
        plt.scatter(est_times, est_freqs, c='#e60000', marker='.', s=25, alpha=0.9, label='MF0 Prediction', zorder=2)
    
    plt.yscale('log'); plt.yticks([128, 256, 512, 1024], labels=['128', '256', '512', '1024'])
    plt.ylabel('Frequency (Hz)', fontsize=14); plt.xlabel('Time (sec)', fontsize=14)
    plt.title(f'Pitch Prediction vs. Reference for: {track_group_name}', fontsize=16)
    plt.legend(loc='upper right', fontsize=12); plt.grid(axis='y', linestyle=':', color='gray')
    
    if ref_times.size > 0:
        plt.xlim(ref_times.min() - 1, ref_times.max() + 1)
    
    plt.ylim(bottom=60); plt.show()

# ─── Main Execution ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate SalienceCNN on the Cantoria dataset.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("cantoria_root", help="Path to the root of the Cantoria dataset (e.g., './datasets/Cantoria/CantoriaDataset_v1.0.0').")
    parser.add_argument("checkpoint_path", help="Path to the model checkpoint (.pth).")
    parser.add_argument("--track_name", default=None, help="Specify a track base name (e.g., 'Cantoria_0001'). If None, a random track is chosen.")
    parser.add_argument("--threshold", type=float, default=0.3, help="Salience threshold for peak-picking (0.0 to 1.0).")
    args = parser.parse_args()

    config = {'sr': 22050, 'hop_length': 256, 'fmin': 32.7, 'harmonics': [1, 2, 3, 4, 5],
              'bins_per_octave': 60, 'n_octaves': 6, 'n_bins': 360}
    
    # 1. Discover tracks
    print("Scanning Cantoria dataset for complete SATB tracks...")
    complete_tracks = find_valid_tracks(args.cantoria_root)
    if not complete_tracks:
        print("No complete SATB tracks found. Exiting."); return
    print(f"Found {len(complete_tracks)} complete SATB track groups.")
    
    # 2. Select a track group for evaluation
    if args.track_name and args.track_name in complete_tracks:
        track_base = args.track_name
    else:
        track_base = random.choice(list(complete_tracks.keys()))
    track_group = complete_tracks[track_base]
    print(f"\nSelected track for evaluation: {track_base}")

    # 3. Generate ground truth and model prediction
    model = get_model(args.checkpoint_path)
    audio_mix, ref_times, ref_freqs = generate_ground_truth(track_group, args.cantoria_root, config)
    salience_map = run_inference_on_mix(model, audio_mix, config)
    est_times, est_freqs = extract_pitches_from_salience(salience_map, config, threshold=args.threshold)
    
    # 4. Evaluate and visualize
    evaluate_and_plot(ref_times, ref_freqs, est_times, est_freqs, track_base)

if __name__ == '__main__':
    main()