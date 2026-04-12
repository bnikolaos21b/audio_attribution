"""
Noise floor analysis: isolate frames at or below the 10th-percentile RMS
for each file, compute the mean magnitude spectrum of those frames,
then compare Suno AI vs real folk recordings.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
AUDIO_DIR     = Path(os.environ.get("SUNO_AUDIO_DIR", str(_PROJECT_ROOT / "data" / "suno")))
OUT_PNG       = _PROJECT_ROOT / "results" / "plots" / "suno_noise_floor_analysis.png"
SR         = 22050
CLIP_SEC   = 30
HOP_LENGTH = 512
N_FFT      = 2048
PERCENTILE = 10      # RMS threshold percentile

# ── File lists ─────────────────────────────────────────────────────────────────
all_files  = sorted(AUDIO_DIR.glob("*.mp3"))
real_files = [f for f in all_files if f.name.startswith("real_")]
suno_files = [f for f in all_files if not f.name.startswith("real_")]
print(f"Suno: {len(suno_files)}  Real: {len(real_files)}")

freq_bins = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)   # (N_FFT//2 + 1,)

# ── Per-file extraction ────────────────────────────────────────────────────────

def get_noise_floor_spectrum(path):
    """
    Load the middle 30 s, find frames whose RMS is ≤ 10th percentile,
    return the mean magnitude spectrum (dB, normalised to 0 dB peak)
    of those quiet frames, plus the count of quiet frames used.
    """
    total  = librosa.get_duration(path=str(path))
    offset = max(0.0, (total - CLIP_SEC) / 2.0)
    y, _   = librosa.load(str(path), sr=SR, offset=offset,
                           duration=CLIP_SEC, mono=True)

    # STFT
    D   = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mag = np.abs(D)                              # (freq, time)

    # Per-frame RMS from the magnitude spectrum
    frame_rms = np.sqrt(np.mean(mag ** 2, axis=0))   # (time,)

    # Threshold: 10th percentile of RMS across this file's frames
    thresh        = np.percentile(frame_rms, PERCENTILE)
    quiet_mask    = frame_rms <= thresh
    n_quiet       = int(quiet_mask.sum())

    quiet_mag     = mag[:, quiet_mask]           # (freq, n_quiet)
    mean_mag      = np.mean(quiet_mag, axis=1)   # (freq,)

    # Convert to dB, normalise so peak = 0 dB
    mean_db       = librosa.amplitude_to_db(mean_mag, ref=np.max(mean_mag))

    return mean_db, n_quiet, thresh

print("\n── Suno files ──")
suno_spectra = []
for f in suno_files:
    spec, nq, thr = get_noise_floor_spectrum(f)
    suno_spectra.append(spec)
    print(f"  {f.name:<45}  quiet frames: {nq:3d}  RMS thresh: {thr:.5f}")

print("\n── Real files ──")
real_spectra = []
for f in real_files:
    spec, nq, thr = get_noise_floor_spectrum(f)
    real_spectra.append(spec)
    print(f"  {f.name:<45}  quiet frames: {nq:3d}  RMS thresh: {thr:.5f}")

suno_arr = np.array(suno_spectra)   # (26, freq)
real_arr = np.array(real_spectra)   # ( 5, freq)

suno_mean = np.mean(suno_arr, axis=0)
suno_var  = np.var(suno_arr,  axis=0)
suno_std  = np.std(suno_arr,  axis=0)
real_mean = np.mean(real_arr, axis=0)
real_std  = np.std(real_arr,  axis=0)

# ── Plot ───────────────────────────────────────────────────────────────────────
SUNO_COLOR = "#E05C5C"
REAL_COLOR = "#4C9BE8"
VAR_COLOR  = "#9B59B6"

fig, axes = plt.subplots(3, 1, figsize=(13, 14),
                          gridspec_kw={"height_ratios": [3, 2, 2]})
fig.suptitle(
    "Noise Floor Spectral Analysis\n"
    f"Frames ≤ {PERCENTILE}th-percentile RMS per file  |  Middle {CLIP_SEC}s  |  "
    f"Suno n={len(suno_files)}, Real n={len(real_files)}",
    fontsize=13, fontweight="bold"
)

# ── Panel 1: overlaid mean spectra ────────────────────────────────────────────
ax = axes[0]
# Individual Suno traces (faint)
for spec in suno_arr:
    ax.plot(freq_bins, spec, color=SUNO_COLOR, alpha=0.12, lw=0.7)
# Individual Real traces (faint)
for spec in real_arr:
    ax.plot(freq_bins, spec, color=REAL_COLOR, alpha=0.25, lw=0.9)

# Mean ± 1 std bands
ax.fill_between(freq_bins,
                suno_mean - suno_std, suno_mean + suno_std,
                color=SUNO_COLOR, alpha=0.18)
ax.fill_between(freq_bins,
                real_mean - real_std,  real_mean + real_std,
                color=REAL_COLOR,  alpha=0.22)
ax.plot(freq_bins, suno_mean, color=SUNO_COLOR, lw=2.2,
        label=f"Suno AI — mean (n={len(suno_files)})")
ax.plot(freq_bins, real_mean, color=REAL_COLOR,  lw=2.2,
        label=f"Real Folk — mean (n={len(real_files)})")

ax.set_xscale("log")
ax.set_xlim(freq_bins[1], freq_bins[-1])
ax.set_xlabel("Frequency (Hz)", fontsize=11)
ax.set_ylabel("Relative magnitude (dB, peak-normalised)", fontsize=11)
ax.set_title("Mean noise-floor magnitude spectrum (shaded = ±1 std)", fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, which="both", alpha=0.25)
ax.set_ylim(-80, 5)

# Annotate the difference at a few landmark frequencies
for fq in [500, 2000, 6000, 12000]:
    idx   = np.argmin(np.abs(freq_bins - fq))
    delta = suno_mean[idx] - real_mean[idx]
    y_pos = max(suno_mean[idx], real_mean[idx]) + 3
    ax.annotate(f"Δ{delta:+.1f} dB", xy=(fq, y_pos),
                fontsize=7.5, color="black", ha="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

# ── Panel 2: difference spectrum (Suno − Real) ────────────────────────────────
ax = axes[1]
diff = suno_mean - real_mean
ax.plot(freq_bins, diff, color="black", lw=1.8, label="Suno − Real (mean dB diff)")
ax.fill_between(freq_bins, diff, 0,
                where=(diff > 0), color=SUNO_COLOR, alpha=0.35,
                label="Suno louder in noise floor")
ax.fill_between(freq_bins, diff, 0,
                where=(diff < 0), color=REAL_COLOR, alpha=0.35,
                label="Real louder in noise floor")
ax.axhline(0, color="gray", lw=0.8, ls="--")
ax.set_xscale("log")
ax.set_xlim(freq_bins[1], freq_bins[-1])
ax.set_xlabel("Frequency (Hz)", fontsize=11)
ax.set_ylabel("Δ dB", fontsize=11)
ax.set_title("Suno − Real: difference in noise-floor spectral shape", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, which="both", alpha=0.25)

# ── Panel 3: per-bin variance across Suno files ───────────────────────────────
ax = axes[2]
ax.plot(freq_bins, suno_var,  color=SUNO_COLOR, lw=2.0,
        label=f"Suno variance (n={len(suno_files)})")
ax.plot(freq_bins, real_std**2, color=REAL_COLOR,  lw=2.0, ls="--",
        label=f"Real variance (n={len(real_files)})")
ax.fill_between(freq_bins, suno_var, color=SUNO_COLOR, alpha=0.2)
ax.set_xscale("log")
ax.set_xlim(freq_bins[1], freq_bins[-1])
ax.set_xlabel("Frequency (Hz)", fontsize=11)
ax.set_ylabel("Variance (dB²)", fontsize=11)
ax.set_title(
    "Per-frequency-bin variance of noise-floor spectrum\n"
    "Low Suno variance → consistent AI generation artefact", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, which="both", alpha=0.25)

# Mark frequency region of minimum Suno variance
low_var_idx = np.argmin(suno_var)
ax.axvline(freq_bins[low_var_idx], color=VAR_COLOR, lw=1.2, ls=":",
           label=f"Suno min-var @ {freq_bins[low_var_idx]:.0f} Hz")
ax.annotate(f"Min variance\n{freq_bins[low_var_idx]:.0f} Hz",
            xy=(freq_bins[low_var_idx], suno_var[low_var_idx]),
            xytext=(freq_bins[low_var_idx] * 1.6, np.max(suno_var) * 0.6),
            fontsize=8, color=VAR_COLOR,
            arrowprops=dict(arrowstyle="->", color=VAR_COLOR))

plt.tight_layout()
plt.savefig(str(OUT_PNG), dpi=150, bbox_inches="tight")
print(f"\nFigure saved → {OUT_PNG}")

# ── Numeric summary ────────────────────────────────────────────────────────────
print("\n" + "=" * 68)
print("  NOISE FLOOR SUMMARY")
print("=" * 68)

bands = [
    ("Sub-bass   (20–80 Hz)",    20,    80),
    ("Bass      (80–300 Hz)",    80,   300),
    ("Midrange (300–2k Hz)",    300,  2000),
    ("Upper-mid (2k–6k Hz)",   2000,  6000),
    ("Presence  (6k–12k Hz)", 6000, 12000),
    ("Air      (12k–22k Hz)", 12000, 22050),
]

print(f"\n{'Band':<26} {'Suno mean':>10} {'Real mean':>10} {'Δ dB':>8}  {'Suno var':>9}  {'Real var':>9}")
print("-" * 78)
for label, flo, fhi in bands:
    mask = (freq_bins >= flo) & (freq_bins < fhi)
    sm = float(np.mean(suno_mean[mask]))
    rm = float(np.mean(real_mean[mask]))
    sv = float(np.mean(suno_var[mask]))
    rv = float(np.mean(real_std[mask] ** 2))
    print(f"{label:<26} {sm:>10.2f} {rm:>10.2f} {sm-rm:>+8.2f}  {sv:>9.3f}  {rv:>9.3f}")

print("\n── Consistency of Suno noise floor ──")
print(f"  Mean per-bin variance  — Suno : {np.mean(suno_var):.3f} dB²")
print(f"  Mean per-bin variance  — Real : {np.mean(real_std**2):.3f} dB²")
ratio = np.mean(suno_var) / (np.mean(real_std**2) + 1e-12)
print(f"  Variance ratio (Suno/Real)    : {ratio:.3f}x  "
      f"({'more consistent' if ratio < 1 else 'less consistent'} across files)")
print(f"\n  Frequency of minimum Suno variance : {freq_bins[low_var_idx]:.1f} Hz")
print(f"  Frequency of maximum Suno variance : {freq_bins[np.argmax(suno_var)]:.1f} Hz")
print("=" * 68)
