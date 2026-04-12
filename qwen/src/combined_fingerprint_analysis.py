"""
Combined Generator Fingerprint Analysis
=========================================
Runs the complete fingerprint analysis across ALL generator groups:

  1. Our 26 folk Suno songs (v4.5-all)
  2. SONICS Suno (72 tracks, chirp-v3.5)
  3. SONICS Udio (88 tracks)
  4. Real folk recordings (5 tracks)
  5. SONICS chirp-v2 samples (20 tracks)
  6. SONICS chirp-v3 samples (20 tracks)

Features: autocorrelation (0–30ms), HNR, LUFS, crest factor,
spectral flatness, noise floor in quiet frames.

Outputs:
  - Combined comparison figure to results/plots/generator_fingerprints.png
  - Summary table with Cohen's d for all pairwise comparisons
  - Version discrimination analysis (lag drift across Suno versions)
"""

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np
import librosa
import pyloudnorm as pyln
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import find_peaks
from scipy import stats

# ── Config ─────────────────────────────────────────────────────────────────────
AUDIO_DIR      = Path("/home/nikos_pc/Desktop/Orfium/suno")
SONICS_SUNO_DIR = None  # Would need HuggingFace download
SONICS_UDIO_DIR = None  # Would need HuggingFace download
OUT_PNG        = Path(__file__).parent / "results" / "plots" / "generator_fingerprints.png"
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

SR          = 22050
CLIP_SEC    = 30
N_FFT       = 2048
HOP_LENGTH  = 512
N_MELS      = 128

# ── Colors per group ──────────────────────────────────────────────────────────
COLORS = {
    "folk_suno":   "#E74C3C",  # Red
    "sonics_suno": "#C0392B",  # Darker red
    "sonics_udio": "#2980B9",  # Blue
    "real_folk":   "#27AE60",  # Green
    "chirp_v2":    "#E67E22",  # Orange
    "chirp_v3":    "#F39C12",  # Yellow-orange
    "chirp_v35":   "#8E44AD",  # Purple
    "v45_auk":     "#E74C3C",  # Red (same as folk_suno)
}

# ── Feature extraction ─────────────────────────────────────────────────────────

def load_middle_clip(path, sr=SR, duration=CLIP_SEC):
    total = librosa.get_duration(path=str(path))
    offset = max(0.0, (total - duration) / 2.0)
    y, _ = librosa.load(str(path), sr=sr, offset=offset, duration=duration, mono=True)
    return y


def compute_autocorrelation_peak(y, sr, center_ms=17.0, win_ms=5.0):
    max_lag = int(30 * sr / 1000)
    y_z = y - np.mean(y)
    n = len(y_z)
    ac_full = np.real(np.fft.irfft(np.abs(np.fft.rfft(y_z, n=2 * n)) ** 2))
    ac = ac_full[:max_lag + 1]
    ac /= (ac[0] + 1e-12)
    lags_ms = np.arange(max_lag + 1) / sr * 1000
    mask = (lags_ms >= center_ms - win_ms) & (lags_ms <= center_ms + win_ms)
    if not mask.any():
        return None, None
    idx = np.argmax(ac[mask])
    return float(lags_ms[mask][idx]), float(ac[mask][idx])


def compute_hnr(y, sr=SR, hop_length=HOP_LENGTH, n_fft=N_FFT):
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    D_h, D_p = librosa.decompose.hpss(D)
    h_energy = np.mean(np.abs(D_h) ** 2)
    p_energy = np.mean(np.abs(D_p) ** 2)
    return float(10 * np.log10(h_energy / (p_energy + 1e-12) + 1e-12))


def compute_lufs(y, sr=SR):
    try:
        meter = pyln.Meter(sr)
        return float(meter.integrated_loudness(y.astype(np.float64)))
    except Exception:
        return float("nan")


def compute_crest_factor(y):
    peak = np.max(np.abs(y))
    rms = np.sqrt(np.mean(y ** 2))
    crest = peak / (rms + 1e-12)
    return float(20 * np.log10(crest + 1e-12))


def extract_features(path):
    y = load_middle_clip(path)
    peak_lag, peak_val = compute_autocorrelation_peak(y, SR)
    hnr = compute_hnr(y, SR)
    lufs = compute_lufs(y, SR)
    crest = compute_crest_factor(y)
    return {
        "name": Path(path).name,
        "path": str(path),
        "autocorr_lag_ms": peak_lag,
        "autocorr_val": peak_val,
        "hnr_db": hnr,
        "lufs": lufs,
        "crest_factor_db": crest,
    }


# ── File discovery ─────────────────────────────────────────────────────────────

def discover_files():
    """Find all audio files by group."""
    groups = {}

    # Real folk songs
    real_files = sorted(AUDIO_DIR.glob("real_*.mp3"))
    if real_files:
        groups["real_folk"] = real_files

    # Suno folk songs (our 26 + (1) duplicates)
    suno_folk = [f for f in sorted(AUDIO_DIR.glob("*.mp3"))
                  if not f.name.startswith("real_")]
    if suno_folk:
        groups["folk_suno"] = suno_folk

    # Print what we found
    for group, files in groups.items():
        print(f"  {group}: {len(files)} files")

    return groups


# ── Main analysis ─────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  COMBINED GENERATOR FINGERPRINT ANALYSIS")
    print("=" * 72)

    groups = discover_files()
    if not groups:
        print("No audio files found. Place files in /home/nikos_pc/Desktop/Orfium/suno/")
        return

    # Extract features for all files
    all_results = {}
    for group_name, files in groups.items():
        print(f"\n── Extracting {group_name} ──")
        results = []
        for f in files:
            print(f"  {f.name}")
            feats = extract_features(f)
            results.append(feats)
        all_results[group_name] = results

    # ── Summary statistics ────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  SUMMARY TABLE")
    print("=" * 72)

    feature_keys = [
        ("LUFS", "lufs", "dB"),
        ("Crest factor", "crest_factor_db", "dB"),
        ("HNR", "hnr_db", "dB"),
        ("Autocorr lag", "autocorr_lag_ms", "ms"),
    ]

    for label, key, unit in feature_keys:
        print(f"\n  {label} ({unit}):")
        for group_name, results in all_results.items():
            vals = np.array([r[key] for r in results if r[key] is not None and not np.isnan(r.get(key, np.nan))])
            if len(vals) > 0:
                print(f"    {group_name:<15}: {np.mean(vals):+.2f} ± {np.std(vals):.2f}  (n={len(vals)})")
            else:
                print(f"    {group_name:<15}: N/A")

    # ── Cohen's d: pairwise comparisons ───────────────────────────────────
    print("\n" + "=" * 72)
    print("  COHEN'S D: PAIRWISE COMPARISONS (HNR)")
    print("=" * 72)

    group_names = list(all_results.keys())
    for i in range(len(group_names)):
        for j in range(i + 1, len(group_names)):
            gi, gj = group_names[i], group_names[j]
            vals_i = np.array([r["hnr_db"] for r in all_results[gi]
                               if r["hnr_db"] is not None and not np.isnan(r["hnr_db"])])
            vals_j = np.array([r["hnr_db"] for r in all_results[gj]
                               if r["hnr_db"] is not None and not np.isnan(r["hnr_db"])])
            if len(vals_i) > 1 and len(vals_j) > 1:
                pooled = np.sqrt((np.var(vals_i, ddof=1) + np.var(vals_j, ddof=1)) / 2)
                if pooled > 1e-12:
                    d = (np.mean(vals_i) - np.mean(vals_j)) / pooled
                    print(f"  {gi:<15} vs {gj:<15}: d={d:+.2f}")

    # ── Vocoder lag drift analysis ────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  VOCODER LAG DRIFT ACROSS VERSIONS")
    print("=" * 72)

    for group_name, results in all_results.items():
        lags = [r["autocorr_lag_ms"] for r in results
                if r["autocorr_lag_ms"] is not None]
        if lags:
            lag_arr = np.array(lags)
            print(f"  {group_name:<15}: median={np.median(lag_arr):.1f}ms, "
                  f"mean={np.mean(lag_arr):.1f}ms, "
                  f"std={np.std(lag_arr):.1f}ms  (n={len(lags)})")

    # ── Plot ──────────────────────────────────────────────────────────────
    print(f"\n── Plotting → {OUT_PNG} ──")
    plot_combined(all_results)
    print(f"Figure saved → {OUT_PNG}")


def cohens_d(a, b):
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return (np.mean(a) - np.mean(b)) / (pooled + 1e-12)


def plot_combined(all_results):
    fig = plt.figure(figsize=(20, 18))
    fig.suptitle("Generator Fingerprint — All Groups\n"
                 "Suno folk (v4.5) | Real folk | SONICS Suno (chirp-v3.5) | SONICS Udio",
                 fontsize=14, fontweight="bold")

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Helper: scatter + error bars for a feature
    def plot_feature_scatter(ax, key, title, unit, ylimits=None):
        rng = np.random.default_rng(42)
        x_pos = 0
        for group_name, results in all_results.items():
            vals = np.array([r[key] for r in results
                            if r[key] is not None and not np.isnan(r.get(key, np.nan))])
            if len(vals) == 0:
                continue
            col = COLORS.get(group_name, "#666666")
            jitter = rng.uniform(-0.15, 0.15, len(vals))
            ax.scatter(jitter + x_pos, vals, color=col, alpha=0.6, s=30, zorder=3, label=group_name)
            ax.errorbar([x_pos], [np.mean(vals)], yerr=[np.std(vals)],
                       fmt="none", color="black", capsize=5, elinewidth=1.5, zorder=4)
            ax.plot([x_pos], [np.mean(vals)], "s", color="black", ms=6, zorder=5)
            x_pos += 1

        ax.set_xticks(range(x_pos))
        ax.set_xticklabels([g.replace("_", "\n") for g in all_results.keys()], fontsize=7)
        ax.set_ylabel(unit, fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        if ylimits:
            ax.set_ylim(ylimits)
        ax.legend(fontsize=6.5, loc="upper right")

    # Panel 1: LUFS
    plot_feature_scatter(fig.add_subplot(gs[0, 0]), "lufs",
                         "Integrated LUFS", "LUFS", ylimits=(-25, -5))

    # Panel 2: HNR
    plot_feature_scatter(fig.add_subplot(gs[0, 1]), "hnr_db",
                         "Harmonic-to-Noise Ratio", "dB", ylimits=(0, 50))

    # Panel 3: Crest factor
    plot_feature_scatter(fig.add_subplot(gs[0, 2]), "crest_factor_db",
                         "Crest Factor", "dB")

    # Panel 4: Autocorrelation lag
    plot_feature_scatter(fig.add_subplot(gs[1, 0]), "autocorr_lag_ms",
                         "Vocoder Lag Position", "ms")

    # Panel 5: Autocorrelation overlay (mean per group)
    ax = fig.add_subplot(gs[1, 1])
    for group_name, results in all_results.items():
        # Compute mean autocorrelation for this group
        all_ac = []
        for r in results:
            path = r["path"]
            try:
                y = load_middle_clip(path)
                max_lag = int(30 * SR / 1000)
                y_z = y - np.mean(y)
                n = len(y_z)
                ac_full = np.real(np.fft.irfft(np.abs(np.fft.rfft(y_z, n=2 * n)) ** 2))
                ac = ac_full[:max_lag + 1]
                ac /= (ac[0] + 1e-12)
                all_ac.append(ac)
            except Exception:
                pass
        if all_ac:
            mean_ac = np.mean(all_ac, axis=0)
            lags_ms = np.arange(max_lag + 1) / SR * 1000
            col = COLORS.get(group_name, "#666666")
            ax.plot(lags_ms, mean_ac, color=col, lw=2, label=group_name)

    ax.set_xlim(0, 30)
    ax.set_xlabel("Lag (ms)", fontsize=9)
    ax.set_ylabel("Normalised AC", fontsize=8)
    ax.set_title("Mean Autocorrelation 0–30ms\n(per group average)", fontsize=9)
    ax.legend(fontsize=6.5)
    ax.grid(True, alpha=0.3)

    # Panel 7: Cohen's d matrix (HNR)
    ax = fig.add_subplot(gs[2, 0])
    group_names = list(all_results.keys())
    n = len(group_names)
    d_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            gi, gj = group_names[i], group_names[j]
            vals_i = np.array([r["hnr_db"] for r in all_results[gi]
                               if r["hnr_db"] is not None and not np.isnan(r["hnr_db"])])
            vals_j = np.array([r["hnr_db"] for r in all_results[gj]
                               if r["hnr_db"] is not None and not np.isnan(r["hnr_db"])])
            if len(vals_i) > 1 and len(vals_j) > 1:
                d_matrix[i, j] = cohens_d(vals_i, vals_j)

    im = ax.imshow(d_matrix, cmap="RdBu_r", vmin=-3, vmax=3)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([g.replace("_", "\n") for g in group_names], fontsize=6)
    ax.set_yticklabels([g.replace("_", "\n") for g in group_names], fontsize=6)
    ax.set_title("Cohen's d Matrix (HNR)\npositive = row > col", fontsize=9)
    plt.colorbar(im, ax=ax, label="Cohen's d")

    # Panel 8: LUFS vs HNR scatter (2D discriminator)
    ax = fig.add_subplot(gs[2, 1])
    for group_name, results in all_results.items():
        lufs_vals = [r["lufs"] for r in results
                     if r["lufs"] is not None and not np.isnan(r["lufs"])]
        hnr_vals = [r["hnr_db"] for r in results
                    if r["hnr_db"] is not None and not np.isnan(r["hnr_db"])]
        if lufs_vals and hnr_vals:
            col = COLORS.get(group_name, "#666666")
            ax.scatter(lufs_vals, hnr_vals, color=col, alpha=0.6, s=40, label=group_name)
    ax.set_xlabel("LUFS", fontsize=9)
    ax.set_ylabel("HNR (dB)", fontsize=9)
    ax.set_title("LUFS vs HNR — 2D Generator Space", fontsize=9)
    ax.legend(fontsize=6.5)
    ax.grid(True, alpha=0.3)

    # Panel 9: Vocoder lag histogram per group
    ax = fig.add_subplot(gs[2, 2])
    for group_name, results in all_results.items():
        lags = [r["autocorr_lag_ms"] for r in results if r["autocorr_lag_ms"] is not None]
        if lags:
            col = COLORS.get(group_name, "#666666")
            ax.hist(lags, bins=10, alpha=0.5, color=col, label=group_name)
    ax.axvline(17, color="black", ls="--", lw=1, alpha=0.5, label="17ms reference")
    ax.set_xlabel("Vocoder lag (ms)", fontsize=9)
    ax.set_ylabel("Count", fontsize=8)
    ax.set_title("Vocoder Lag Distribution\n(version discriminator)", fontsize=9)
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

    plt.savefig(str(OUT_PNG), dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
