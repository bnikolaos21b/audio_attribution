"""
Acoustic fingerprint analysis: Suno AI-generated vs real folk recordings.
Extracts multiple spectral/temporal features from the middle 30s of each file.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
AUDIO_DIR     = Path(os.environ.get("SUNO_AUDIO_DIR", str(_PROJECT_ROOT / "data" / "suno")))
OUT_PNG       = _PROJECT_ROOT / "results" / "plots" / "suno_fingerprint_analysis.png"
SR          = 22050
CLIP_SEC    = 30          # seconds to analyse
HOP_LENGTH  = 512
N_FFT       = 2048
N_MELS      = 128
QUIET_THRESH_DB = -40     # frames below this are "quiet"
PHASE_BINS  = 5           # frequency bins sampled for phase variance

# ── File lists ────────────────────────────────────────────────────────────────
all_files = sorted(AUDIO_DIR.glob("*.mp3"))
real_files = [f for f in all_files if f.name.startswith("real_")]
suno_files = [f for f in all_files if not f.name.startswith("real_")]
print(f"Suno files : {len(suno_files)}")
print(f"Real files : {len(real_files)}")

# ── Feature extraction ────────────────────────────────────────────────────────

def load_middle_clip(path, sr=SR, duration=CLIP_SEC):
    """Load middle `duration` seconds of an audio file."""
    total = librosa.get_duration(path=str(path))
    offset = max(0.0, (total - duration) / 2.0)
    y, _ = librosa.load(str(path), sr=sr, offset=offset, duration=duration, mono=True)
    return y

def extract_features(path):
    """Return a dict of scalar/array features for one file."""
    print(f"  {path.name}")
    y = load_middle_clip(path)

    # STFT
    D        = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mag      = np.abs(D)
    mag_db   = librosa.amplitude_to_db(mag, ref=np.max)

    # ── 1. Spectral flatness (per frame → mean, std) ─────────────────────────
    flat     = librosa.feature.spectral_flatness(y=y, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    flat_db  = librosa.amplitude_to_db(flat[None, :], ref=1.0)[0]   # to dB for readability

    # ── 2. Noise floor: mean energy of "quiet" frames ────────────────────────
    frame_rms   = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
    frame_db    = librosa.amplitude_to_db(frame_rms, ref=np.max(frame_rms))
    quiet_mask  = frame_db < QUIET_THRESH_DB
    noise_floor = float(np.mean(frame_db[quiet_mask])) if quiet_mask.any() else float(np.min(frame_db))
    noise_floor_frac = float(np.mean(quiet_mask))   # fraction of quiet frames

    # ── 3. Phase variance at frame boundaries ────────────────────────────────
    phase    = np.angle(D)                          # (freq, time)
    # Phase difference between consecutive frames
    d_phase  = np.diff(phase, axis=1)              # (freq, time-1)
    # Wrap to [-π, π]
    d_phase  = np.angle(np.exp(1j * d_phase))
    # Sample a few frequency bins spread across spectrum
    n_freqs  = D.shape[0]
    bin_idx  = np.linspace(0, n_freqs - 1, PHASE_BINS + 2, dtype=int)[1:-1]
    phase_var_per_bin = np.var(d_phase[bin_idx, :], axis=1)  # (PHASE_BINS,)
    phase_var_mean    = float(np.mean(phase_var_per_bin))
    phase_var_std     = float(np.std(phase_var_per_bin))

    # ── 4. Spectral centroid, rolloff, bandwidth ─────────────────────────────
    centroid  = librosa.feature.spectral_centroid(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    rolloff   = librosa.feature.spectral_rolloff(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
                                                  roll_percent=0.85)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]

    # ── 5. Mel spectrogram statistics ────────────────────────────────────────
    mel      = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
                                               n_mels=N_MELS)
    mel_db   = librosa.power_to_db(mel, ref=np.max)
    mel_mean = float(np.mean(mel_db))
    mel_std  = float(np.std(mel_db))
    mel_skew = float(np.mean(((mel_db - mel_mean) / (mel_std + 1e-9)) ** 3))
    # Per-band mean energy
    mel_band_mean = np.mean(mel_db, axis=1)       # (N_MELS,)

    # ── 6. Temporal flatness (RMS variance / mean) ───────────────────────────
    rms_cv   = float(np.std(frame_rms) / (np.mean(frame_rms) + 1e-9))

    # ── 7. High-frequency energy ratio ───────────────────────────────────────
    # Ratio of energy above 8 kHz to total
    freq_bins = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    hf_mask   = freq_bins >= 8000
    hf_energy = float(np.mean(mag[hf_mask, :] ** 2))
    tot_energy= float(np.mean(mag ** 2))
    hf_ratio  = hf_energy / (tot_energy + 1e-12)

    # ── 8. Spectral contrast ─────────────────────────────────────────────────
    contrast  = librosa.feature.spectral_contrast(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH)
    contrast_mean = np.mean(contrast, axis=1)     # (n_bands+1,)

    # ── 9. Zero-crossing rate ─────────────────────────────────────────────────
    zcr       = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)[0]

    return dict(
        # scalars
        flat_mean        = float(np.mean(flat)),
        flat_std         = float(np.std(flat)),
        flat_db_mean     = float(np.mean(flat_db)),
        noise_floor      = noise_floor,
        noise_floor_frac = noise_floor_frac,
        phase_var_mean   = phase_var_mean,
        phase_var_std    = phase_var_std,
        centroid_mean    = float(np.mean(centroid)),
        centroid_std     = float(np.std(centroid)),
        rolloff_mean     = float(np.mean(rolloff)),
        rolloff_std      = float(np.std(rolloff)),
        bandwidth_mean   = float(np.mean(bandwidth)),
        bandwidth_std    = float(np.std(bandwidth)),
        mel_mean         = mel_mean,
        mel_std          = mel_std,
        mel_skew         = mel_skew,
        rms_cv           = rms_cv,
        hf_ratio         = hf_ratio,
        zcr_mean         = float(np.mean(zcr)),
        zcr_std          = float(np.std(zcr)),
        # arrays (for per-file plots)
        mel_band_mean    = mel_band_mean,
        contrast_mean    = contrast_mean,
        flat_series      = flat,
        centroid_series  = centroid,
    )

print("\n── Extracting Suno features ──")
suno_feats = [extract_features(f) for f in suno_files]
print("\n── Extracting Real features ──")
real_feats  = [extract_features(f) for f in real_files]

# ── Aggregate ─────────────────────────────────────────────────────────────────

SCALAR_KEYS = [
    "flat_mean", "flat_std", "flat_db_mean",
    "noise_floor", "noise_floor_frac",
    "phase_var_mean", "phase_var_std",
    "centroid_mean", "centroid_std",
    "rolloff_mean", "rolloff_std",
    "bandwidth_mean", "bandwidth_std",
    "mel_mean", "mel_std", "mel_skew",
    "rms_cv", "hf_ratio",
    "zcr_mean", "zcr_std",
]

def agg(feat_list, key):
    vals = np.array([f[key] for f in feat_list])
    return vals, float(np.mean(vals)), float(np.std(vals))

suno_scalars = {k: agg(suno_feats, k) for k in SCALAR_KEYS}
real_scalars  = {k: agg(real_feats,  k) for k in SCALAR_KEYS}

# Average array features
suno_mel_band = np.mean([f["mel_band_mean"] for f in suno_feats], axis=0)
real_mel_band  = np.mean([f["mel_band_mean"] for f in real_feats],  axis=0)
suno_contrast = np.mean([f["contrast_mean"] for f in suno_feats], axis=0)
real_contrast  = np.mean([f["contrast_mean"] for f in real_feats],  axis=0)

# ── Cohen's d (effect size) ────────────────────────────────────────────────────
def cohens_d(a, b):
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return (np.mean(a) - np.mean(b)) / (pooled + 1e-12)

effect_sizes = {}
for k in SCALAR_KEYS:
    a = suno_scalars[k][0]
    b = real_scalars[k][0]
    effect_sizes[k] = cohens_d(a, b)

sorted_keys = sorted(SCALAR_KEYS, key=lambda k: abs(effect_sizes[k]), reverse=True)

# ── Plotting ──────────────────────────────────────────────────────────────────
print("\n── Plotting ──")

fig = plt.figure(figsize=(22, 28))
fig.suptitle("Suno AI vs Real Folk — Acoustic Fingerprint Analysis",
             fontsize=16, fontweight="bold", y=0.995)

gs = gridspec.GridSpec(5, 4, figure=fig, hspace=0.55, wspace=0.4)

SUNO_COLOR = "#E05C5C"
REAL_COLOR = "#4C9BE8"

# Helper: bar comparison for a scalar feature
def plot_bar(ax, key, label, unit=""):
    sv, sm, ss = suno_scalars[key]
    rv, rm, rs = real_scalars[key]
    cats = ["Suno\n(AI)", "Real\n(Folk)"]
    means = [sm, rm]
    stds  = [ss, rs]
    colors = [SUNO_COLOR, REAL_COLOR]
    bars = ax.bar(cats, means, yerr=stds, color=colors, width=0.45,
                  capsize=5, error_kw=dict(elinewidth=1.2, capthick=1.2))
    ax.set_title(f"{label}\n(Cohen's d={effect_sizes[key]:.2f})", fontsize=8.5)
    ax.set_ylabel(unit, fontsize=7)
    ax.tick_params(labelsize=8)
    # Scatter individual points
    for x_offset, vals, col in zip([-0.22, 0.22], [sv, rv], colors):
        jitter = np.random.default_rng(42).uniform(-0.07, 0.07, len(vals))
        ax.scatter([x_offset + 0 + j for j in jitter], vals,
                   color=col, alpha=0.5, s=12, zorder=3)

np.random.seed(42)

# Row 0 ────────────────────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0]); plot_bar(ax, "flat_mean",      "Spectral Flatness (mean)")
ax = fig.add_subplot(gs[0, 1]); plot_bar(ax, "flat_db_mean",   "Spectral Flatness dB (mean)", "dB")
ax = fig.add_subplot(gs[0, 2]); plot_bar(ax, "noise_floor",    "Noise Floor (quiet frames)", "dB")
ax = fig.add_subplot(gs[0, 3]); plot_bar(ax, "noise_floor_frac","Fraction of Quiet Frames")

# Row 1 ────────────────────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 0]); plot_bar(ax, "phase_var_mean", "Phase Δ Variance (mean)", "rad²")
ax = fig.add_subplot(gs[1, 1]); plot_bar(ax, "phase_var_std",  "Phase Δ Variance (std)",  "rad²")
ax = fig.add_subplot(gs[1, 2]); plot_bar(ax, "centroid_mean",  "Spectral Centroid (mean)", "Hz")
ax = fig.add_subplot(gs[1, 3]); plot_bar(ax, "rolloff_mean",   "Spectral Rolloff 85% (mean)", "Hz")

# Row 2 ────────────────────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[2, 0]); plot_bar(ax, "bandwidth_mean", "Spectral Bandwidth (mean)", "Hz")
ax = fig.add_subplot(gs[2, 1]); plot_bar(ax, "mel_mean",       "Mel Spectrogram Mean",  "dB")
ax = fig.add_subplot(gs[2, 2]); plot_bar(ax, "mel_std",        "Mel Spectrogram Std",   "dB")
ax = fig.add_subplot(gs[2, 3]); plot_bar(ax, "mel_skew",       "Mel Spectrogram Skew")

# Row 3 ────────────────────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[3, 0]); plot_bar(ax, "hf_ratio",       "HF Energy Ratio (>8 kHz)")
ax = fig.add_subplot(gs[3, 1]); plot_bar(ax, "rms_cv",         "RMS Coeff. of Variation")
ax = fig.add_subplot(gs[3, 2]); plot_bar(ax, "zcr_mean",       "Zero-Crossing Rate (mean)")
ax = fig.add_subplot(gs[3, 3]); plot_bar(ax, "zcr_std",        "Zero-Crossing Rate (std)")

# Row 4 — spectrum profiles ────────────────────────────────────────────────────
mel_freqs = librosa.mel_frequencies(n_mels=N_MELS, fmin=0, fmax=SR / 2)

ax = fig.add_subplot(gs[4, 0:2])
ax.plot(mel_freqs, suno_mel_band, color=SUNO_COLOR, lw=2, label="Suno (AI)")
ax.plot(mel_freqs, real_mel_band,  color=REAL_COLOR, lw=2, label="Real Folk")
ax.fill_between(mel_freqs, suno_mel_band, real_mel_band,
                alpha=0.15, color="purple", label="Δ region")
ax.set_xscale("log")
ax.set_xlabel("Frequency (Hz)", fontsize=9)
ax.set_ylabel("Mean Mel Energy (dB)", fontsize=9)
ax.set_title("Mel Band Energy Profile (avg over all files)", fontsize=9)
ax.legend(fontsize=8)
ax.tick_params(labelsize=8)
ax.grid(True, alpha=0.3)

ax = fig.add_subplot(gs[4, 2])
ax.barh(np.arange(len(suno_contrast)), suno_contrast, color=SUNO_COLOR, alpha=0.8, label="Suno")
ax.barh(np.arange(len(real_contrast)),  real_contrast,  color=REAL_COLOR, alpha=0.8, label="Real", left=0)
ax.set_title("Spectral Contrast\nper sub-band", fontsize=9)
ax.set_xlabel("Mean contrast (dB)", fontsize=8)
ax.set_yticks(np.arange(len(suno_contrast)))
ax.set_yticklabels([f"B{i}" for i in range(len(suno_contrast))], fontsize=7)
ax.legend(fontsize=7)

ax = fig.add_subplot(gs[4, 3])
top_n = 10
top_keys = sorted_keys[:top_n]
d_vals   = [effect_sizes[k] for k in top_keys]
colors_d = [SUNO_COLOR if d > 0 else REAL_COLOR for d in d_vals]
y_pos = np.arange(top_n)
ax.barh(y_pos, d_vals, color=colors_d)
ax.axvline(0, color="black", lw=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels([k.replace("_", "\n") for k in top_keys], fontsize=6.5)
ax.set_xlabel("Cohen's d  (Suno − Real)", fontsize=8)
ax.set_title("Top 10 Discriminating\nFeatures (|Cohen's d|)", fontsize=9)
ax.tick_params(axis="x", labelsize=8)
ax.grid(True, axis="x", alpha=0.3)

plt.savefig(str(OUT_PNG), dpi=150, bbox_inches="tight")
print(f"\nFigure saved → {OUT_PNG}")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  ACOUSTIC FINGERPRINT SUMMARY  —  Suno AI vs Real Folk")
print("=" * 72)
print(f"{'Feature':<26} {'Suno mean':>11} {'Real mean':>11} {'Cohen d':>9}  Direction")
print("-" * 72)
for k in sorted_keys:
    sv, sm, ss = suno_scalars[k]
    rv, rm, rs = real_scalars[k]
    d  = effect_sizes[k]
    arrow = "Suno ↑" if d > 0 else "Real ↑"
    print(f"{k:<26} {sm:>11.4f} {rm:>11.4f} {d:>+9.2f}  {arrow}")
print("=" * 72)

print("\n── Key findings ──")
for k in sorted_keys[:5]:
    d  = effect_sizes[k]
    sv, sm, ss = suno_scalars[k]
    rv, rm, rs = real_scalars[k]
    pct = abs(sm - rm) / (abs(rm) + 1e-12) * 100
    direction = "higher" if d > 0 else "lower"
    print(f"  • {k}: Suno is {direction} by ~{pct:.1f}%  (|d|={abs(d):.2f})")
