"""
Suno model-version acoustic fingerprint comparison.
Groups: chirp-v2-xxl-alpha (n=20), chirp-v3 (n=20), chirp-v3.5 (n=20).
Features: autocorrelation 10–25 ms, HNR, LUFS,
          spectral flatness (BW-limited), crest factor, noise floor.
Audio streamed from SONICS HuggingFace via remotezip (no full zip download).
"""

import io
import os
import csv
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import librosa
import librosa.effects
import pyloudnorm as pyln
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.signal import find_peaks
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_PNG       = _PROJECT_ROOT / "results" / "plots" / "suno_versions_fingerprint.png"

# ── Audio / feature constants ─────────────────────────────────────────────────
SR       = 22050
CLIP_SEC = 30
HOP      = 512
N_FFT    = 2048
N_MELS   = 128
RANDOM_SEED = 2025
N_PER_GROUP = 20

SONICS_CSV_URL = ("https://huggingface.co/datasets/awsaf49/sonics"
                  "/resolve/main/fake_songs.csv")
SONICS_ZIP_BASE = ("https://huggingface.co/datasets/awsaf49/sonics"
                   "/resolve/main/fake_songs/")

# ── Feature extraction (same as generator_fingerprints.py) ───────────────────

def _load_middle_mono(y_full, sr_full):
    if sr_full != SR:
        y_full = librosa.resample(y_full, orig_sr=sr_full, target_sr=SR)
    if y_full.ndim > 1:
        y_full = np.mean(y_full, axis=0)
    n    = len(y_full)
    want = int(CLIP_SEC * SR)
    start = max(0, (n - want) // 2)
    return y_full[start:start + want].astype(np.float32)


def _load_middle_stereo(y_full, sr_full):
    if sr_full != SR:
        if y_full.ndim == 1:
            y_full = librosa.resample(y_full, orig_sr=sr_full, target_sr=SR)
        else:
            y_full = np.stack([librosa.resample(c, orig_sr=sr_full, target_sr=SR)
                               for c in y_full])
    if y_full.ndim == 1:
        y_full = np.stack([y_full, y_full])
    n    = y_full.shape[1]
    want = int(CLIP_SEC * SR)
    start = max(0, (n - want) // 2)
    return y_full[:, start:start + want].astype(np.float32)


def extract_features(y_full, sr_full):
    is_mono_input = (y_full.ndim == 1)
    effective_sr  = min(int(sr_full), SR)

    y  = _load_middle_mono(y_full, sr_full)
    ys = _load_middle_stereo(y_full, sr_full)
    feats = {}

    # 1. Waveform autocorrelation — peak in 10–25 ms
    max_lag = int(30 * SR / 1000)
    y_z     = y - np.mean(y)
    n       = len(y_z)
    ac_full = np.real(np.fft.irfft(np.abs(np.fft.rfft(y_z, n=2 * n)) ** 2))
    ac      = ac_full[:max_lag + 1] / (ac_full[0] + 1e-12)
    lags_ms = np.arange(max_lag + 1) / SR * 1000
    window  = (lags_ms >= 10) & (lags_ms <= 25)
    ac_win  = ac[window]
    lg_win  = lags_ms[window]
    peaks, _ = find_peaks(ac_win, height=0.005, prominence=0.003)
    if len(peaks):
        best = peaks[np.argmax(ac_win[peaks])]
        feats["autocorr_peak_ms"]  = float(lg_win[best])
        feats["autocorr_peak_val"] = float(ac_win[best])
    else:
        feats["autocorr_peak_ms"]  = float("nan")
        feats["autocorr_peak_val"] = 0.0

    # 2. HNR
    y_harm, y_perc = librosa.effects.hpss(y, margin=3.0)
    rms_tot  = librosa.feature.rms(y=y,      frame_length=N_FFT, hop_length=HOP)[0]
    rms_harm = librosa.feature.rms(y=y_harm, frame_length=N_FFT, hop_length=HOP)[0]
    rms_perc = librosa.feature.rms(y=y_perc, frame_length=N_FFT, hop_length=HOP)[0]
    voiced   = rms_tot > np.percentile(rms_tot, 10)
    feats["hnr"] = float(np.mean(rms_harm[voiced] / (rms_perc[voiced] + 1e-12)))

    # 3. Spectral flatness — BW-limited to effective Nyquist
    stft_mag  = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    fft_freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    max_bin   = int(np.searchsorted(fft_freqs, effective_sr / 2))
    stft_bw   = stft_mag[:max_bin] + 1e-10
    geo_mean  = np.exp(np.mean(np.log(stft_bw), axis=0))
    arith_mean = np.mean(stft_bw, axis=0) + 1e-12
    feats["spectral_flatness"] = float(np.mean(geo_mean / arith_mean))

    # 5. Mid/side ratio — NaN for mono
    if is_mono_input:
        feats["ms_ratio"] = float("nan")
    else:
        L, R = ys[0].astype(np.float64), ys[1].astype(np.float64)
        mid  = (L + R) / 2.0
        side = (L - R) / 2.0
        feats["ms_ratio"] = float(np.mean(mid ** 2) / (np.mean(side ** 2) + 1e-12))

    # 6. Crest factor (dB)
    sig   = np.mean(ys, axis=0)
    peak  = float(np.max(np.abs(sig)))
    rms_v = float(np.sqrt(np.mean(sig ** 2)))
    feats["crest_db"] = 20 * np.log10(peak / (rms_v + 1e-12) + 1e-12)

    # 7. LUFS
    meter = pyln.Meter(SR)
    try:
        feats["lufs"] = float(meter.integrated_loudness(ys.T.astype(np.float64)))
    except Exception:
        feats["lufs"] = float("nan")

    # 8. Noise floor
    frame_rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    frame_db  = librosa.amplitude_to_db(frame_rms, ref=np.max(frame_rms) + 1e-12)
    thresh    = np.percentile(frame_db, 10)
    quiet     = frame_db <= thresh
    feats["noise_floor_db"] = float(np.mean(frame_db[quiet]))

    return feats


# ── SONICS audio loader ───────────────────────────────────────────────────────

def fetch_from_sonics(file_to_part: dict) -> dict:
    """
    Download files from SONICS zip parts via remotezip.
    file_to_part: {filename.mp3: part_str}
    Returns: {filename.mp3: (y_full, sr_full)}
    """
    from remotezip import RemoteZip

    # Group by part
    by_part = defaultdict(list)
    for fname, part_str in file_to_part.items():
        by_part[part_str].append(fname)

    results = {}
    for part_str, fnames in sorted(by_part.items()):
        url = SONICS_ZIP_BASE + f"{part_str}.zip"
        print(f"  Opening {part_str}.zip ({len(fnames)} files)…")
        with RemoteZip(url) as z:
            zip_map = {n.split("/")[-1]: n for n in z.namelist()}
            for fname in fnames:
                if fname not in zip_map:
                    print(f"    SKIP {fname}: not in zip")
                    continue
                data = z.read(zip_map[fname])
                y, sr = librosa.load(io.BytesIO(data), sr=None, mono=False)
                results[fname] = (y, sr)
                print(f"    OK  {fname}  shape={y.shape}  sr={sr}")
    return results


# ── Sample filenames from fake_songs.csv ─────────────────────────────────────

def sample_filenames(csv_url: str, seed: int = RANDOM_SEED, n: int = N_PER_GROUP):
    import requests
    print(f"Fetching {csv_url} …")
    r = requests.get(csv_url, timeout=180)
    reader = csv.DictReader(r.text.splitlines())
    by_algo = defaultdict(list)
    for row in reader:
        by_algo[row["algorithm"]].append(row["filename"])
    rng = random.Random(seed)
    samples = {}
    for algo in ["chirp-v2-xxl-alpha", "chirp-v3", "chirp-v3.5"]:
        pool = by_algo[algo]
        picked = rng.sample(pool, min(n, len(pool)))
        samples[algo] = [f + ".mp3" for f in picked]
        print(f"  {algo!r:25s}: sampled {len(picked)} / {len(pool)}")
    return samples


# ── Scan zip parts to locate files ───────────────────────────────────────────

def locate_files(target_fnames: set) -> dict:
    """Scan central directory of each zip part (one range request each)."""
    from remotezip import RemoteZip
    file_to_part = {}
    remaining = set(target_fnames)
    for part in range(1, 11):
        if not remaining:
            break
        part_str = f"part_{part:02d}"
        url = SONICS_ZIP_BASE + f"{part_str}.zip"
        try:
            with RemoteZip(url) as z:
                for n in z.namelist():
                    base = n.split("/")[-1]
                    if base in remaining:
                        file_to_part[base] = part_str
                        remaining.discard(base)
            print(f"  {part_str}: found {len(file_to_part)}/{len(target_fnames)} "
                  f"(remaining: {len(remaining)})")
        except Exception as e:
            print(f"  {part_str}: ERROR {e}")
    if remaining:
        print(f"  WARNING: {len(remaining)} files not found: {sorted(remaining)[:5]}")
    return file_to_part


# ── Aggregation helpers ───────────────────────────────────────────────────────

def col(records, key):
    vals = np.array([r[key] for r in records], dtype=float)
    return vals[np.isfinite(vals)]

def cohens_d(a, b):
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return (np.mean(a) - np.mean(b)) / (pooled + 1e-12)


# ── Plot ──────────────────────────────────────────────────────────────────────

FEAT_KEYS = [
    "autocorr_peak_val",
    "autocorr_peak_ms",
    "hnr",
    "spectral_flatness",
    "crest_db",
    "lufs",
    "noise_floor_db",
]

FEAT_LABELS = {
    "autocorr_peak_val":  "Autocorr peak\n(10–25 ms)",
    "autocorr_peak_ms":   "Autocorr peak lag\n(ms)",
    "hnr":                "HNR\n(harm/perc RMS)",
    "spectral_flatness":  "Spectral flatness\n(BW-limited to src Nyquist)",
    "crest_db":           "Crest factor\n(dB)",
    "lufs":               "LUFS\n(integrated)",
    "noise_floor_db":     "Noise floor\n(quiet 10%, dB)",
}

C_V2   = "#8E44AD"   # purple  — chirp-v2-xxl-alpha
C_V3   = "#E67E22"   # orange  — chirp-v3
C_V35  = "#2980B9"   # blue    — chirp-v3.5

def plot_versions(v2, v3, v35):
    groups = [
        (v2,  C_V2,  f"chirp-v2-xxl-alpha\n(n={len(v2)})"),
        (v3,  C_V3,  f"chirp-v3\n(n={len(v3)})"),
        (v35, C_V35, f"chirp-v3.5\n(n={len(v35)})"),
    ]
    ncols = 4
    nrows = (len(FEAT_KEYS) + ncols - 1) // ncols

    fig = plt.figure(figsize=(20, nrows * 5 + 2))
    fig.suptitle(
        "Suno Model-Version Acoustic Fingerprint Comparison\n"
        "chirp-v2-xxl-alpha vs chirp-v3 vs chirp-v3.5  |  "
        "n=20 per group  |  SONICS dataset (mono 16 kHz MP3)",
        fontsize=12, fontweight="bold", y=0.998,
    )
    gs = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.65, wspace=0.40)
    rng = np.random.default_rng(42)

    for idx, key in enumerate(FEAT_KEYS):
        row, c = divmod(idx, ncols)
        ax = fig.add_subplot(gs[row, c])
        all_vals = [col(g, key) for g, _, _ in groups]

        for pos, (records, color, label), vals in zip([1, 2, 3], groups, all_vals):
            if len(vals) < 2:
                continue
            parts = ax.violinplot([vals], positions=[pos], widths=0.55,
                                  showmedians=True, showextrema=True)
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.40)
            for part in ["cmedians", "cmaxes", "cmins", "cbars"]:
                parts[part].set_color(color)
                parts[part].set_linewidth(1.3)
            jitter = rng.uniform(-0.12, 0.12, len(vals))
            ax.scatter(pos + jitter, vals, color=color, alpha=0.65, s=18, zorder=3)

        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["v2-xxl\nalpha", "v3", "v3.5"], fontsize=8)
        ax.set_title(FEAT_LABELS[key], fontsize=9)
        ax.grid(True, axis="y", alpha=0.25)

        # Cohen's d annotations: v2 vs v3, v3 vs v3.5
        d_v2v3  = cohens_d(all_vals[0], all_vals[1])
        d_v3v35 = cohens_d(all_vals[1], all_vals[2])
        ann = []
        if np.isfinite(d_v2v3):
            ann.append(f"v2→v3: d={d_v2v3:+.2f}")
        if np.isfinite(d_v3v35):
            ann.append(f"v3→v3.5: d={d_v3v35:+.2f}")
        if ann:
            ax.text(0.97, 0.97, "\n".join(ann), transform=ax.transAxes,
                    fontsize=7, ha="right", va="top", color="gray", style="italic")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=C_V2,  label="chirp-v2-xxl-alpha"),
        Patch(facecolor=C_V3,  label="chirp-v3"),
        Patch(facecolor=C_V35, label="chirp-v3.5"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               fontsize=10, frameon=True, bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(OUT_PNG), dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {OUT_PNG}")


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(v2, v3, v35):
    groups = [
        ("chirp-v2-xxl-alpha", v2),
        ("chirp-v3",           v3),
        ("chirp-v3.5",         v35),
    ]
    print("\n" + "=" * 110)
    print("  SUNO VERSION FINGERPRINT SUMMARY")
    print("=" * 110)
    print(f"{'Feature':<24}" +
          "".join(f"  {name:>24}" for name, _ in groups) +
          "  d(v2→v3)  d(v3→v3.5)")
    print("-" * 110)
    for key in FEAT_KEYS:
        vals = [col(g, key) for _, g in groups]
        row  = f"{key:<24}"
        for v in vals:
            fin = v[np.isfinite(v)]
            if len(fin):
                row += f"  {np.mean(fin):>10.4f} ±{np.std(fin):>9.4f}"
            else:
                row += f"  {'N/A':>24}"
        d1 = cohens_d(vals[0], vals[1])
        d2 = cohens_d(vals[1], vals[2])
        row += f"  {d1:>+9.2f}  {d2:>+10.2f}"
        print(row)
    print("=" * 110)
    print("d = Cohen's d (positive = first group higher). |d|>0.8 = large effect.")
    print()

    # Autocorrelation prevalence
    print("── Autocorrelation peak prevalence (10–25 ms, peak_val > 0.03) ──")
    for name, g in groups:
        peaks  = col(g, "autocorr_peak_val")
        frac   = np.mean(peaks > 0.03)
        median_lag = np.nanmedian(col(g, "autocorr_peak_ms"))
        print(f"  {name:<24}  {frac*100:.0f}% have peak>0.03   median lag={median_lag:.1f} ms")
    print()

    # LUFS
    print("── LUFS summary ──")
    for name, g in groups:
        lufs = col(g, "lufs")
        lufs = lufs[np.isfinite(lufs)]
        print(f"  {name:<24}  mean={np.mean(lufs):.1f} ± {np.std(lufs):.1f} LUFS   "
              f"min={np.min(lufs):.1f}  max={np.max(lufs):.1f}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  Suno Version Fingerprint Analysis")
    print("=" * 65)

    # 1. Sample filenames from fake_songs.csv
    print("\n── Step 1: Sample filenames from fake_songs.csv ──")
    samples = sample_filenames(SONICS_CSV_URL)
    all_targets = set()
    for fnames in samples.values():
        all_targets.update(fnames)

    # 2. Locate files in zip parts
    print("\n── Step 2: Locate files in SONICS zip parts ──")
    file_to_part = locate_files(all_targets)

    # 3. Download and extract features
    print("\n── Step 3: Download audio and extract features ──")
    audio_cache = fetch_from_sonics(file_to_part)

    features = {algo: [] for algo in samples}
    for algo, fnames in samples.items():
        print(f"\n  [{algo}]")
        for fname in fnames:
            if fname not in audio_cache:
                print(f"    SKIP {fname}: not downloaded")
                continue
            y, sr = audio_cache[fname]
            feats = extract_features(y, sr)
            feats["name"] = fname
            features[algo].append(feats)
            ac_v = feats["autocorr_peak_val"]
            print(f"    {fname:40s}  HNR={feats['hnr']:.2f}  "
                  f"LUFS={feats['lufs']:.1f}  autocorr={ac_v:.4f}")

    v2  = features["chirp-v2-xxl-alpha"]
    v3  = features["chirp-v3"]
    v35 = features["chirp-v3.5"]
    print(f"\nCollected: v2={len(v2)}, v3={len(v3)}, v3.5={len(v35)}")

    # 4. Plot
    print("\n── Step 4: Plot ──")
    plot_versions(v2, v3, v35)

    # 5. Summary table
    print_summary(v2, v3, v35)
