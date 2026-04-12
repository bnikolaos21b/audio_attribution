"""
Zoom into 8–11 kHz noise floor: individual per-file lines, Suno vs real.
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
OUT_PNG       = _PROJECT_ROOT / "results" / "plots" / "suno_noise_floor_nyquist.png"
SR         = 22050
CLIP_SEC   = 30
HOP_LENGTH = 512
N_FFT      = 2048
PERCENTILE = 10
ZOOM_LO    = 8000
ZOOM_HI    = 11025
BAND_LO    = 9000    # band for variance ratio report
BAND_HI    = 11000

# ── Files ──────────────────────────────────────────────────────────────────────
all_files  = sorted(AUDIO_DIR.glob("*.mp3"))
real_files = [f for f in all_files if f.name.startswith("real_")]
suno_files = [f for f in all_files if not f.name.startswith("real_")]
print(f"Suno: {len(suno_files)}  Real: {len(real_files)}")

freq_bins = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)   # 1025 bins, 0–11025 Hz

# ── Extraction (same logic as before) ─────────────────────────────────────────

def get_noise_floor_spectrum(path):
    total  = librosa.get_duration(path=str(path))
    offset = max(0.0, (total - CLIP_SEC) / 2.0)
    y, _   = librosa.load(str(path), sr=SR, offset=offset,
                           duration=CLIP_SEC, mono=True)
    D      = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mag    = np.abs(D)
    frame_rms   = np.sqrt(np.mean(mag ** 2, axis=0))
    thresh      = np.percentile(frame_rms, PERCENTILE)
    quiet_mask  = frame_rms <= thresh
    mean_mag    = np.mean(mag[:, quiet_mask], axis=1)
    # Normalise: peak = 0 dB
    mean_db     = librosa.amplitude_to_db(mean_mag, ref=np.max(mean_mag))
    return mean_db

print("\nExtracting Suno…")
suno_spectra = []
suno_labels  = []
for f in suno_files:
    print(f"  {f.name}")
    suno_spectra.append(get_noise_floor_spectrum(f))
    # Short label: strip " (1)" variants and truncate
    label = f.stem.replace(" (1)", "*")
    suno_labels.append(label[:28])

print("\nExtracting Real…")
real_spectra = []
real_labels  = []
for f in real_files:
    print(f"  {f.name}")
    real_spectra.append(get_noise_floor_spectrum(f))
    real_labels.append(f.stem.replace("real_", "")[:28])

suno_arr = np.array(suno_spectra)   # (26, 1025)
real_arr = np.array(real_spectra)   # ( 5, 1025)

# ── Restrict to zoom window ────────────────────────────────────────────────────
zoom_mask  = (freq_bins >= ZOOM_LO) & (freq_bins <= ZOOM_HI)
band_mask  = (freq_bins >= BAND_LO) & (freq_bins <= BAND_HI)
fz         = freq_bins[zoom_mask]

suno_zoom  = suno_arr[:, zoom_mask]   # (26, n_zoom_bins)
real_zoom  = real_arr[:,  zoom_mask]  # ( 5, n_zoom_bins)

# ── Variance ratio in 9–11 kHz ────────────────────────────────────────────────
suno_band  = suno_arr[:, band_mask]   # (26, n_band_bins)
real_band  = real_arr[:,  band_mask]  # ( 5, n_band_bins)

# Per-file mean level in the band (scalar per file)
suno_band_means = np.mean(suno_band, axis=1)   # (26,)
real_band_means = np.mean(real_band, axis=1)   # ( 5,)

# Per-bin variance across files, then averaged over the band
suno_bin_var = np.var(suno_band, axis=0)       # (n_band_bins,)
real_bin_var = np.var(real_band, axis=0)       # (n_band_bins,)
suno_mean_var = float(np.mean(suno_bin_var))
real_mean_var = float(np.mean(real_bin_var))
var_ratio     = suno_mean_var / (real_mean_var + 1e-12)

# ── Colour maps ───────────────────────────────────────────────────────────────
import matplotlib.cm as cm
suno_cmap = cm.get_cmap("Reds",   len(suno_files) + 4)
real_cmap  = cm.get_cmap("Blues", len(real_files)  + 2)
suno_colors = [suno_cmap(i + 3) for i in range(len(suno_files))]
real_colors  = [real_cmap(i  + 2) for i in range(len(real_files))]

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    3, 1, figsize=(14, 15),
    gridspec_kw={"height_ratios": [4, 2, 2]}
)
fig.suptitle(
    f"Noise Floor: 8–11 kHz zoom  |  10th-percentile RMS frames  |  "
    f"Middle {CLIP_SEC}s per file",
    fontsize=13, fontweight="bold"
)

# ── Panel 1: individual lines ─────────────────────────────────────────────────
ax = axes[0]

for i, (spec, label, col) in enumerate(zip(suno_zoom, suno_labels, suno_colors)):
    ls = "--" if label.endswith("*") else "-"
    ax.plot(fz / 1000, spec, color=col, lw=1.1, alpha=0.85,
            linestyle=ls, label=label)

for i, (spec, label, col) in enumerate(zip(real_zoom, real_labels, real_colors)):
    ax.plot(fz / 1000, spec, color=col, lw=2.4, alpha=0.95,
            linestyle="-", label=f"REAL: {label}")

# Overlay means
ax.plot(fz / 1000, np.mean(suno_zoom, axis=0),
        color="#8B0000", lw=2.5, ls="-", zorder=10, label="Suno MEAN")
ax.plot(fz / 1000, np.mean(real_zoom, axis=0),
        color="#00008B", lw=2.5, ls="-", zorder=10, label="Real MEAN")

# Shade the 9–11 kHz analysis band
ax.axvspan(BAND_LO / 1000, BAND_HI / 1000, color="gold", alpha=0.12,
           label=f"{BAND_LO//1000}–{BAND_HI//1000} kHz analysis band")
ax.axvline(BAND_LO / 1000, color="goldenrod", lw=0.8, ls=":")
ax.axvline(BAND_HI / 1000, color="goldenrod", lw=0.8, ls=":")

ax.set_xlabel("Frequency (kHz)", fontsize=11)
ax.set_ylabel("Relative magnitude (dB, peak-normalised)", fontsize=11)
ax.set_title(
    "Individual file noise-floor spectra (8–11 kHz)\n"
    "Suno: red shades (dashed = variant '(1)'), Real: blue shades",
    fontsize=10
)
ax.set_xlim(ZOOM_LO / 1000, ZOOM_HI / 1000)
ax.grid(True, alpha=0.3)

# Two-column legend outside plot
leg = ax.legend(fontsize=6.5, ncol=3,
                loc="upper center", bbox_to_anchor=(0.5, -0.14),
                framealpha=0.9, columnspacing=0.8, handlelength=1.5)

# ── Panel 2: per-bin variance across files ────────────────────────────────────
ax = axes[1]
ax.plot(fz / 1000, np.var(suno_zoom, axis=0),
        color="#C0392B", lw=2.0, label=f"Suno variance (n={len(suno_files)})")
ax.plot(fz / 1000, np.var(real_zoom, axis=0),
        color="#2980B9",  lw=2.0, label=f"Real variance  (n={len(real_files)})")
ax.fill_between(fz / 1000, np.var(suno_zoom, axis=0),
                color="#C0392B", alpha=0.15)
ax.fill_between(fz / 1000, np.var(real_zoom, axis=0),
                color="#2980B9",  alpha=0.15)
ax.axvspan(BAND_LO / 1000, BAND_HI / 1000, color="gold", alpha=0.12)
ax.set_xlabel("Frequency (kHz)", fontsize=11)
ax.set_ylabel("Variance (dB²)", fontsize=11)
ax.set_title("Per-bin variance across files  (lower = more consistent)", fontsize=10)
ax.set_xlim(ZOOM_LO / 1000, ZOOM_HI / 1000)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ── Panel 3: per-file mean level in 9–11 kHz (strip plot) ────────────────────
ax = axes[2]

x_suno = np.zeros(len(suno_band_means))
x_real  = np.ones(len(real_band_means))
jitter_s = np.linspace(-0.18, 0.18, len(suno_band_means))
jitter_r = np.linspace(-0.10, 0.10, len(real_band_means))

for j, (val, label, col) in enumerate(zip(suno_band_means, suno_labels, suno_colors)):
    ax.scatter(x_suno[j] + jitter_s[j], val, color=col, s=55, zorder=3)
    ax.text(x_suno[j] + jitter_s[j], val + 0.3, label,
            fontsize=5.5, ha="center", va="bottom", color=col, rotation=60)

for j, (val, label, col) in enumerate(zip(real_band_means, real_labels, real_colors)):
    ax.scatter(x_real[j] + jitter_r[j], val, color=col, s=90, zorder=3,
               marker="D")
    ax.text(x_real[j] + jitter_r[j], val + 0.3, label,
            fontsize=6.5, ha="center", va="bottom", color=col, rotation=45)

# Mean bars
ax.hlines(np.mean(suno_band_means), -0.3, 0.3,
          colors="#8B0000", lw=2.5, zorder=4, label="Suno mean")
ax.hlines(np.mean(real_band_means),  0.7, 1.3,
          colors="#00008B",  lw=2.5, zorder=4, label="Real mean")

ax.set_xticks([0, 1])
ax.set_xticklabels(["Suno AI  (n=26)", f"Real Folk  (n=5)"], fontsize=11)
ax.set_ylabel(f"Mean dB in {BAND_LO//1000}–{BAND_HI//1000} kHz band", fontsize=11)
ax.set_title(
    f"Per-file mean noise-floor level in {BAND_LO//1000}–{BAND_HI//1000} kHz  |  "
    f"Variance ratio (Suno/Real) = {var_ratio:.2f}×",
    fontsize=10
)
ax.set_xlim(-0.5, 1.5)
ax.legend(fontsize=9)
ax.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(str(OUT_PNG), dpi=150, bbox_inches="tight")
print(f"\nFigure saved → {OUT_PNG}")

# ── Numeric summary ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"  VARIANCE RATIO  {BAND_LO//1000}–{BAND_HI//1000} kHz  (noise floor frames)")
print("=" * 60)
print(f"  Suno mean per-bin variance : {suno_mean_var:.4f} dB²")
print(f"  Real  mean per-bin variance : {real_mean_var:.4f} dB²")
print(f"  Ratio Suno / Real           : {var_ratio:.3f}×")
print()
print(f"  Per-file mean level in band (dB, peak-normalised):")
print(f"  {'File':<40} {'Mean dB':>8}")
print(f"  {'─'*48}")
for label, val in sorted(zip(suno_labels, suno_band_means), key=lambda x: x[1]):
    print(f"  [Suno] {label:<34} {val:>8.2f}")
for label, val in sorted(zip(real_labels, real_band_means), key=lambda x: x[1]):
    print(f"  [Real] {label:<34} {val:>8.2f}")
print()
print(f"  Suno spread (std of per-file means) : {np.std(suno_band_means):.3f} dB")
print(f"  Real  spread (std of per-file means) : {np.std(real_band_means):.3f} dB")
print("=" * 60)
