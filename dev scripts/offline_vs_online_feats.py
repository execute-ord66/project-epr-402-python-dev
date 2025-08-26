# HCQT comparison: librosa vs nnAudio on sawtooth + Gaussian noise
# Matches your scaling: amplitude_to_db(..., ref=max) -> (dB/80)+1, clamped to [0,1]

import numpy as np
import torch
import matplotlib.pyplot as plt
import librosa
from nnAudio.features import CQT1992v2

# -------- Config (align with your config) --------
dp = {
    "sr": 22050,
    "hop_length": 512,
    "fmin": 32.703195662574764,  # C1
    "bins_per_octave": 60,
    "n_octaves": 6,
    "harmonics": [1, 2, 3, 4, 5],   # test multiple harmonics in HCQT
}
n_bins = dp["n_octaves"] * dp["bins_per_octave"]
H = len(dp["harmonics"])

# Target: exactly N frames so comparing is easy
NFRAMES = 128
segment_len = NFRAMES * dp["hop_length"]
dur = segment_len / dp["sr"]

# -------- Signal: sawtooth + time-domain gaussian noise --------
f0 = 110.0  # A2 fundamental (well above fmin)
t = np.arange(segment_len) / dp["sr"]

# Classic sawtooth via Fourier partials (sum of odd/even with 1/k amplitude, alternating signs)
# y(t) = 2/pi * sum_{k=1..K} (-1)^{k+1} * (1/k) * sin(2*pi*k*f0*t)
K = 50  # number of partials
y = np.zeros_like(t, dtype=np.float64)
for k in range(1, K+1):
    y += ((-1)**(k+1)) * np.sin(2*np.pi*k*f0*t) / k
y *= (2/np.pi)

# Add time-domain Gaussian noise at desired SNR (e.g., 20 dB)
def add_noise_snr(x, snr_db):
    p_sig = np.mean(x**2)
    p_noise = p_sig / (10**(snr_db/10.0))
    n = np.random.randn(*x.shape) * np.sqrt(p_noise)
    return x + n

y_noisy = add_noise_snr(y, snr_db=20.0).astype(np.float32)

# -------- Helper: scale to [0,1] like your code --------
def log01_from_amplitude(mag):
    ref = np.max(mag) if np.max(mag) > 0 else 1.0
    db = 20.0 * np.log10(np.clip(mag, 1e-10, None) / ref)
    return np.clip((db / 80.0) + 1.0, 0.0, 1.0)

# -------- Offline HCQT via librosa.cqt for each harmonic --------
hcqt_librosa = []
for h in dp["harmonics"]:
    c = librosa.cqt(
        y_noisy, sr=dp["sr"], hop_length=dp["hop_length"],
        fmin=dp["fmin"] * h,
        n_bins=n_bins, bins_per_octave=dp["bins_per_octave"],
        # keep defaults or explicitly set these to match your training choices:
        # window='hann', center=True, pad_mode='reflect'
    )
    hcqt_librosa.append(np.abs(c))
# Stack as (H, F, T) and crop to same T across harmonics
min_T = min(C.shape[1] for C in hcqt_librosa)
hcqt_librosa = np.stack([C[:, :min_T] for C in hcqt_librosa], axis=0)
log_hcqt_librosa = log01_from_amplitude(hcqt_librosa)

# -------- On-the-fly HCQT via nnAudio (CQT1992v2) --------
device = "cuda" if torch.cuda.is_available() else "cpu"
hcqt_nna = []
with torch.no_grad():
    x = torch.from_numpy(y_noisy).to(device)[None, :]  # (1, samples)
    for h in dp["harmonics"]:
        cqt = CQT1992v2(
            sr=dp["sr"], hop_length=dp["hop_length"],
            fmin=dp["fmin"] * h,
            n_bins=n_bins, bins_per_octave=dp["bins_per_octave"],
            # Important: make sure the kernel/padding choices align conceptually with librosa
            # If available in your nnAudio version, you can set pad_mode='reflect'
            # and output_format='Magnitude' (else we take abs() below for safety).
        ).to(device)
        C = cqt(x)  # shape: (batch=1, freq, time)
        C = C.squeeze(0)  # (F, T)
        hcqt_nna.append(torch.abs(C).cpu().numpy())
hcqt_nna = np.stack([C[:, :min_T] for C in hcqt_nna], axis=0)
log_hcqt_nna = log01_from_amplitude(hcqt_nna)

# -------- Compare --------
def stats(a, b):
    d = a - b
    return {
        "shape_a": a.shape, "shape_b": b.shape,
        "mae": float(np.mean(np.abs(d))),
        "rmse": float(np.sqrt(np.mean(d**2))),
        "max_abs_diff": float(np.max(np.abs(d))),
        "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b)),
        "max_a": float(np.max(a)),   "max_b": float(np.max(b)),
    }

print("LOG-HCQT  [0,1]-scaled  comparison")
print(stats(log_hcqt_librosa, log_hcqt_nna))

# -------- Visual diagnostics --------
H_show = min(H, 3)  # show first few harmonics to keep it readable

for i in range(H_show):
    plt.figure(figsize=(7,5))
    plt.title(f"Harmonic {dp['harmonics'][i]} — librosa")
    plt.imshow(log_hcqt_librosa[i], aspect='auto', origin='lower')
    plt.xlabel("Frames"); plt.ylabel("CQT bins"); plt.tight_layout(); plt.show()

    plt.figure(figsize=(7,5))
    plt.title(f"Harmonic {dp['harmonics'][i]} — nnAudio")
    plt.imshow(log_hcqt_nna[i], aspect='auto', origin='lower')
    plt.xlabel("Frames"); plt.ylabel("CQT bins"); plt.tight_layout(); plt.show()

    plt.figure(figsize=(7,5))
    plt.title(f"Abs diff (librosa - nnAudio) — harmonic {dp['harmonics'][i]}")
    plt.imshow(np.abs(log_hcqt_librosa[i] - log_hcqt_nna[i]), aspect='auto', origin='lower')
    plt.xlabel("Frames"); plt.ylabel("CQT bins"); plt.tight_layout(); plt.show()

# Also compare time & freq marginals to catch systematic bias
lib_sum_f = log_hcqt_librosa.sum(axis=(0))  # (F, T) summed over H
nna_sum_f = log_hcqt_nna.sum(axis=(0))
plt.figure(figsize=(7,4))
plt.title("Time marginal (sum over freq & harmonics)")
plt.plot(lib_sum_f.sum(axis=0), label='librosa')
plt.plot(nna_sum_f.sum(axis=0), label='nnAudio', linestyle='--')
plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(7,4))
plt.title("Freq marginal (sum over time & harmonics)")
plt.plot(lib_sum_f.sum(axis=1), label='librosa')
plt.plot(nna_sum_f.sum(axis=1), label='nnAudio', linestyle='--')
plt.legend(); plt.tight_layout(); plt.show()
