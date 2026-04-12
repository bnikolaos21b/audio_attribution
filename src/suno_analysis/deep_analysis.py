"""
Deep acoustic analysis: LUFS/crest, clipping, self-similarity, beat regularity,
waveform autocorrelation — Suno AI-generated vs real folk recordings.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import librosa
import pyloudnorm as pyln
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.signal import fftconvolve

# ── Config ─────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
AUDIO_DIR     = Path(os.environ.get("SUNO_AUDIO_DIR", str(_PROJECT_ROOT / "data" / "suno")))
OUT_PNG       = _PROJECT_ROOT / "results" / "plots" / "suno_deep_analysis.png"
SR          = 22050
CLIP_SEC    = 30
HOP_LENGTH  = 512
N_FFT       = 2048
N_MELS      = 64
SEG_SEC     = 1          # self-similarity window size
MAX_LAG_MS  = 30         # autocorrelation lag range
CLIP_THRESH = 1.0 / (2 ** 15)   # 1 LSB at 16-bit equivalent

SUNO_C = "#E05C5C"
REAL_C = "#4C9BE8"

# ── File lists ──────────────────────────────────────────────────────────────────
all_files  = sorted(AUDIO_DIR.glob("*.mp3"))
real_files = [f for f in all_files if f.name.startswith("real_")]
suno_files = [f for f in all_files if not f.name.startswith("real_")]
print(f"Suno: {len(suno_files)}   Real: {len(real_files)}")

# ── Loaders ────────────────────────────────────────────────────────────────────

def load_mono(path):
    total  = librosa.get_duration(path=str(path))
    offset = max(0.0, (total - CLIP_SEC) / 2.0)
    y, _   = librosa.load(str(path), sr=SR, offset=offset,
                           duration=CLIP_SEC, mono=True)
    return y

def load_stereo(path):
    """Return (2, samples) float32 array."""
    total  = librosa.get_duration(path=str(path))
    offset = max(0.0, (total - CLIP_SEC) / 2.0)
    y, _   = librosa.load(str(path), sr=SR, offset=offset,
                           duration=CLIP_SEC, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])
    return y

# ══════════════════════════════════════════════════════════════════════════════
# 1. LUFS + Crest Factor
# ══════════════════════════════════════════════════════════════════════════════

def lufs_crest(path):
    y_st = load_stereo(path)
    # pyloudnorm expects (samples, channels) float64
    data = y_st.T.astype(np.float64)
    meter = pyln.Meter(SR)
    try:
        loudness = meter.integrated_loudness(data)
    except Exception:
        loudness = float("nan")
    # Crest factor on mono mix
    y_mono = np.mean(y_st, axis=0)
    peak   = np.max(np.abs(y_mono))
    rms    = np.sqrt(np.mean(y_mono ** 2))
    crest  = peak / (rms + 1e-12)
    crest_db = 20 * np.log10(crest + 1e-12)
    return loudness, crest_db

# ══════════════════════════════════════════════════════════════════════════════
# 2. Clipping detection
# ══════════════════════════════════════════════════════════════════════════════

def count_clips(path):
    y_st = load_stereo(path)
    # "near 0 dBFS": |sample| >= 1 - CLIP_THRESH
    n_clip  = int(np.sum(np.abs(y_st) >= 1.0 - CLIP_THRESH))
    n_total = y_st.size
    return n_clip, n_clip / n_total

# ══════════════════════════════════════════════════════════════════════════════
# 3. Self-similarity matrix
# ══════════════════════════════════════════════════════════════════════════════

def self_sim_matrix(path, seg_sec=SEG_SEC):
    y     = load_mono(path)
    mel   = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT,
                                            hop_length=HOP_LENGTH, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)   # (mels, frames)

    frames_per_seg = int(seg_sec * SR / HOP_LENGTH)
    n_segs = mel_db.shape[1] // frames_per_seg
    if n_segs < 2:
        return None

    # Segment-level mean mel vectors
    segs = np.array([
        np.mean(mel_db[:, i * frames_per_seg:(i + 1) * frames_per_seg], axis=1)
        for i in range(n_segs)
    ])  # (n_segs, N_MELS)

    # L2-normalise, then cosine similarity = dot product
    norms = np.linalg.norm(segs, axis=1, keepdims=True) + 1e-12
    segs_n = segs / norms
    sim = segs_n @ segs_n.T   # (n_segs, n_segs)
    return sim

# ══════════════════════════════════════════════════════════════════════════════
# 4. Beat regularity
# ══════════════════════════════════════════════════════════════════════════════

def beat_regularity(path):
    y = load_mono(path)
    tempo, beats = librosa.beat.beat_track(y=y, sr=SR, hop_length=HOP_LENGTH)
    if len(beats) < 3:
        return float("nan"), float("nan"), float(np.squeeze(tempo))
    beat_times  = librosa.frames_to_time(beats, sr=SR, hop_length=HOP_LENGTH)
    ibis        = np.diff(beat_times) * 1000   # ms
    ibi_mean    = float(np.mean(ibis))
    ibi_var     = float(np.var(ibis))
    ibi_cv      = float(np.std(ibis) / (ibi_mean + 1e-9))
    return ibi_var, ibi_cv, float(np.squeeze(tempo))

# ══════════════════════════════════════════════════════════════════════════════
# 5. Waveform autocorrelation (0–30 ms)
# ══════════════════════════════════════════════════════════════════════════════

def waveform_autocorr(path):
    y        = load_mono(path)
    max_lag  = int(MAX_LAG_MS * SR / 1000)
    # Normalised autocorrelation via FFT
    y_z      = y - np.mean(y)
    n        = len(y_z)
    ac_full  = np.real(np.fft.irfft(np.abs(np.fft.rfft(y_z, n=2 * n)) ** 2))
    ac       = ac_full[:max_lag + 1]
    ac      /= ac[0] + 1e-12          # normalise to 1 at lag 0
    lags_ms  = np.arange(max_lag + 1) / SR * 1000
    return lags_ms, ac

# ══════════════════════════════════════════════════════════════════════════════
# Extraction loop
# ══════════════════════════════════════════════════════════════════════════════

def process(files, label):
    results = []
    for f in files:
        print(f"  [{label}] {f.name}")
        loudness, crest_db = lufs_crest(f)
        n_clip, clip_frac  = count_clips(f)
        sim                = self_sim_matrix(f)
        ibi_var, ibi_cv, tempo = beat_regularity(f)
        lags_ms, ac        = waveform_autocorr(f)
        results.append(dict(
            name       = f.name,
            loudness   = loudness,
            crest_db   = crest_db,
            n_clip     = n_clip,
            clip_frac  = clip_frac,
            sim        = sim,
            ibi_var    = ibi_var,
            ibi_cv     = ibi_cv,
            tempo      = tempo,
            ac         = ac,
            lags_ms    = lags_ms,
        ))
    return results

print("\n── Suno ──")
suno_res = process(suno_files, "suno")
print("\n── Real ──")
real_res  = process(real_files,  "real")

# ── Aggregate helpers ─────────────────────────────────────────────────────────

def col(results, key):
    return np.array([r[key] for r in results], dtype=float)

def ms(arr):
    a = arr[np.isfinite(arr)]
    return float(np.mean(a)), float(np.std(a))

# Pad / crop sim matrices to a common size then average
def avg_sim(results, target_size=20):
    mats = []
    for r in results:
        m = r["sim"]
        if m is None:
            continue
        n = min(m.shape[0], target_size)
        mats.append(m[:n, :n])
    # Crop all to minimum found size
    min_n = min(m.shape[0] for m in mats)
    return np.mean([m[:min_n, :min_n] for m in mats], axis=0), min_n

suno_sim_avg, sim_n_s = avg_sim(suno_res)
real_sim_avg,  sim_n_r = avg_sim(real_res)

# Autocorrelation average
lags_ms = suno_res[0]["lags_ms"]
suno_ac_avg = np.mean([r["ac"] for r in suno_res], axis=0)
real_ac_avg  = np.mean([r["ac"] for r in real_res],  axis=0)
suno_ac_std = np.std([r["ac"]  for r in suno_res], axis=0)
real_ac_std  = np.std([r["ac"]  for r in real_res],  axis=0)

# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 26))
fig.suptitle("Deep Acoustic Analysis — Suno AI vs Real Folk",
             fontsize=15, fontweight="bold", y=0.997)

gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.55, wspace=0.38)

def scatter_compare(ax, suno_vals, real_vals, ylabel, title, unit=""):
    sv, ss = ms(suno_vals)
    rv, rs = ms(real_vals)
    rng    = np.random.default_rng(0)
    js     = rng.uniform(-0.12, 0.12, len(suno_vals))
    jr     = rng.uniform(-0.12, 0.12, len(real_vals))
    ax.scatter(js,      suno_vals, color=SUNO_C, alpha=0.65, s=30, zorder=3)
    ax.scatter(1 + jr,  real_vals, color=REAL_C,  alpha=0.65, s=55,
               marker="D", zorder=3)
    ax.errorbar([0], [sv], yerr=[ss], fmt="none", color="#8B0000",
                capsize=6, elinewidth=2, capthick=2, zorder=4)
    ax.errorbar([1], [rv], yerr=[rs], fmt="none", color="#00008B",
                capsize=6, elinewidth=2, capthick=2, zorder=4)
    ax.plot([0], [sv], "s", color="#8B0000", ms=8, zorder=5)
    ax.plot([1], [rv], "s", color="#00008B", ms=8, zorder=5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Suno\n(AI)", "Real\nFolk"], fontsize=9)
    ax.set_ylabel(unit, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    return sv, ss, rv, rs

# Row 0 — LUFS & Crest
ax = fig.add_subplot(gs[0, 0])
sv, ss, rv, rs = scatter_compare(ax, col(suno_res, "loudness"), col(real_res, "loudness"),
                                 "", "Integrated LUFS", "LUFS")
ax.axhline(-14, color="green", ls="--", lw=1, alpha=0.6, label="-14 LUFS (streaming)")
ax.axhline(-16, color="orange", ls="--", lw=1, alpha=0.6, label="-16 LUFS")
ax.legend(fontsize=6.5)

ax = fig.add_subplot(gs[0, 1])
scatter_compare(ax, col(suno_res, "crest_db"), col(real_res, "crest_db"),
                "", "Crest Factor", "dB")

# Row 0 — Clipping
ax = fig.add_subplot(gs[0, 2])
scatter_compare(ax, col(suno_res, "n_clip"), col(real_res, "n_clip"),
                "", "Near-clip Sample Count", "# samples")

ax = fig.add_subplot(gs[0, 3])
scatter_compare(ax, col(suno_res, "clip_frac") * 1e6,
                col(real_res,  "clip_frac") * 1e6,
                "", "Clip Fraction (×10⁻⁶)", "ppm")

# Row 1 — Beat regularity
ax = fig.add_subplot(gs[1, 0])
scatter_compare(ax, col(suno_res, "ibi_var"), col(real_res, "ibi_var"),
                "", "Inter-Beat Interval Variance", "ms²")

ax = fig.add_subplot(gs[1, 1])
scatter_compare(ax, col(suno_res, "ibi_cv"), col(real_res, "ibi_cv"),
                "", "IBI Coeff. of Variation", "")

ax = fig.add_subplot(gs[1, 2])
scatter_compare(ax, col(suno_res, "tempo"), col(real_res, "tempo"),
                "", "Estimated Tempo", "BPM")

# Row 1 col 3 — LUFS histogram
ax = fig.add_subplot(gs[1, 3])
bins = np.linspace(-30, -5, 26)
ax.hist(col(suno_res, "loudness"), bins=bins, color=SUNO_C, alpha=0.7,
        label=f"Suno  μ={ms(col(suno_res,'loudness'))[0]:.1f}")
ax.hist(col(real_res, "loudness"), bins=bins, color=REAL_C,  alpha=0.7,
        label=f"Real   μ={ms(col(real_res,'loudness'))[0]:.1f}")
ax.axvline(-14, color="green",  ls="--", lw=1.2, alpha=0.7, label="-14 LUFS")
ax.axvline(-16, color="orange", ls="--", lw=1.2, alpha=0.7, label="-16 LUFS")
ax.set_xlabel("LUFS", fontsize=9)
ax.set_ylabel("Count", fontsize=9)
ax.set_title("LUFS Distribution", fontsize=9)
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# Row 2 — Self-similarity matrices
min_n = min(sim_n_s, sim_n_r)
suno_sim_plot = suno_sim_avg[:min_n, :min_n]
real_sim_plot = real_sim_avg[:min_n,  :min_n]

vmin = min(suno_sim_plot.min(), real_sim_plot.min())
vmax = 1.0

ax = fig.add_subplot(gs[2, 0:2])
im = ax.imshow(suno_sim_plot, aspect="auto", origin="lower",
               cmap="hot", vmin=vmin, vmax=vmax)
ax.set_title(f"Mean Self-Similarity Matrix — Suno  ({min_n}×{min_n} 1-s windows)",
             fontsize=9)
ax.set_xlabel("Window index", fontsize=8)
ax.set_ylabel("Window index", fontsize=8)
plt.colorbar(im, ax=ax, label="Cosine similarity")

ax = fig.add_subplot(gs[2, 2:4])
im = ax.imshow(real_sim_plot, aspect="auto", origin="lower",
               cmap="hot", vmin=vmin, vmax=vmax)
ax.set_title(f"Mean Self-Similarity Matrix — Real  ({min_n}×{min_n} 1-s windows)",
             fontsize=9)
ax.set_xlabel("Window index", fontsize=8)
ax.set_ylabel("Window index", fontsize=8)
plt.colorbar(im, ax=ax, label="Cosine similarity")

# Row 3 — Autocorrelation
ax = fig.add_subplot(gs[3, 0:3])
ax.plot(lags_ms, suno_ac_avg, color=SUNO_C, lw=2.2,
        label=f"Suno AI mean (n={len(suno_res)})")
ax.fill_between(lags_ms, suno_ac_avg - suno_ac_std, suno_ac_avg + suno_ac_std,
                color=SUNO_C, alpha=0.15)
ax.plot(lags_ms, real_ac_avg, color=REAL_C, lw=2.2,
        label=f"Real Folk mean (n={len(real_res)})")
ax.fill_between(lags_ms, real_ac_avg - real_ac_std, real_ac_avg + real_ac_std,
                color=REAL_C, alpha=0.20)
# Mark vocoder-typical frame boundaries
for lag in [10, 11.6, 20]:
    ax.axvline(lag, color="gray", ls=":", lw=1.0, alpha=0.6)
    ax.text(lag + 0.2, 0.03, f"{lag}ms", fontsize=7, color="gray")
ax.set_xlabel("Lag (ms)", fontsize=10)
ax.set_ylabel("Normalised autocorrelation", fontsize=10)
ax.set_title("Waveform Autocorrelation 0–30 ms (shaded = ±1 std)\n"
             "Dotted lines: typical vocoder frame boundaries (10, 11.6, 20 ms)",
             fontsize=9)
ax.legend(fontsize=9)
ax.set_xlim(0, MAX_LAG_MS)
ax.grid(True, alpha=0.3)

# Row 3 col 3 — Difference autocorrelation
ax = fig.add_subplot(gs[3, 3])
diff_ac = suno_ac_avg - real_ac_avg
ax.plot(lags_ms[1:], diff_ac[1:], color="black", lw=1.8)
ax.fill_between(lags_ms[1:], diff_ac[1:], 0,
                where=(diff_ac[1:] > 0), color=SUNO_C, alpha=0.35,
                label="Suno > Real")
ax.fill_between(lags_ms[1:], diff_ac[1:], 0,
                where=(diff_ac[1:] < 0), color=REAL_C, alpha=0.35,
                label="Real > Suno")
ax.axhline(0, color="gray", lw=0.8)
for lag in [10, 11.6, 20]:
    ax.axvline(lag, color="gray", ls=":", lw=0.9, alpha=0.6)
ax.set_xlabel("Lag (ms)", fontsize=9)
ax.set_ylabel("Δ autocorrelation", fontsize=9)
ax.set_title("Autocorr difference\n(Suno − Real)", fontsize=9)
ax.legend(fontsize=8)
ax.set_xlim(1, MAX_LAG_MS)
ax.grid(True, alpha=0.3)

plt.savefig(str(OUT_PNG), dpi=150, bbox_inches="tight")
print(f"\nFigure saved → {OUT_PNG}")

# ══════════════════════════════════════════════════════════════════════════════
# Summary table
# ══════════════════════════════════════════════════════════════════════════════

def cohens_d(a, b):
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return (np.mean(a) - np.mean(b)) / (pooled + 1e-12)

metrics = [
    ("Integrated LUFS",         "loudness",  "LUFS"),
    ("Crest factor",            "crest_db",  "dB"),
    ("Near-clip samples",       "n_clip",    "#"),
    ("Clip fraction",           "clip_frac", "ratio"),
    ("IBI variance",            "ibi_var",   "ms²"),
    ("IBI coeff. of variation", "ibi_cv",    ""),
    ("Tempo",                   "tempo",     "BPM"),
]

print("\n" + "=" * 80)
print("  DEEP ANALYSIS SUMMARY — Suno AI vs Real Folk")
print("=" * 80)
print(f"{'Metric':<28} {'Suno mean±std':>22} {'Real mean±std':>22} {'Cohen d':>9}")
print("-" * 80)
for label, key, unit in metrics:
    sa = col(suno_res, key)
    ra = col(real_res, key)
    sm, ss = ms(sa)
    rm, rs = ms(ra)
    d = cohens_d(sa, ra)
    u = f" {unit}" if unit else ""
    print(f"{label:<28} {sm:>10.3f} ±{ss:>9.3f}{u:>3}   "
          f"{rm:>10.3f} ±{rs:>9.3f}{u:>3}  {d:>+8.2f}")

print("\n── Self-similarity (off-diagonal mean cosine similarity) ──")
def offdiag_mean(m):
    mask = ~np.eye(m.shape[0], dtype=bool)
    return float(np.mean(m[mask]))
print(f"  Suno : {offdiag_mean(suno_sim_avg):.4f}")
print(f"  Real : {offdiag_mean(real_sim_avg):.4f}")
print(f"  Δ    : {offdiag_mean(suno_sim_avg) - offdiag_mean(real_sim_avg):+.4f}")

print("\n── Autocorrelation local maxima in 8–25 ms (vocoder test) ──")
from scipy.signal import find_peaks
window = (lags_ms >= 8) & (lags_ms <= 25)
for label, ac_avg in [("Suno", suno_ac_avg), ("Real", real_ac_avg)]:
    peaks, props = find_peaks(ac_avg[window], height=0.01, prominence=0.005)
    peak_lags = lags_ms[window][peaks]
    peak_vals = ac_avg[window][peaks]
    if len(peaks):
        pstr = ", ".join(f"{l:.1f}ms ({v:.4f})" for l, v in zip(peak_lags, peak_vals))
    else:
        pstr = "none detected"
    print(f"  {label}: {pstr}")

print("\n── Binary / near-binary flags ──")

# LUFS clustering
lufs_s = col(suno_res, "loudness")
lufs_s = lufs_s[np.isfinite(lufs_s)]
in_14  = np.sum((lufs_s >= -15) & (lufs_s <= -13))
in_16  = np.sum((lufs_s >= -17) & (lufs_s <= -15))
print(f"  [LUFS] Suno files near -14 LUFS (±1): {in_14}/{len(lufs_s)}")
print(f"  [LUFS] Suno files near -16 LUFS (±1): {in_16}/{len(lufs_s)}")
lufs_r = col(real_res, "loudness")
lufs_r = lufs_r[np.isfinite(lufs_r)]
print(f"  [LUFS] Real files near -14 LUFS (±1): {np.sum((lufs_r>=-15)&(lufs_r<=-13))}/{len(lufs_r)}")

# Clipping
clip_s = col(suno_res, "n_clip")
clip_r = col(real_res, "n_clip")
print(f"  [CLIP] Suno files with any near-clip samples: {int(np.sum(clip_s>0))}/{len(suno_res)}")
print(f"  [CLIP] Real files with any near-clip samples: {int(np.sum(clip_r>0))}/{len(real_res)}")

# Beat CV
ibi_cv_s = col(suno_res, "ibi_cv")
ibi_cv_r = col(real_res, "ibi_cv")
thr = 0.05
print(f"  [BEAT] Suno files with IBI CV < {thr} (rigid tempo): "
      f"{int(np.sum(ibi_cv_s[np.isfinite(ibi_cv_s)] < thr))}/{len(suno_res)}")
print(f"  [BEAT] Real files with IBI CV < {thr} (rigid tempo): "
      f"{int(np.sum(ibi_cv_r[np.isfinite(ibi_cv_r)] < thr))}/{len(real_res)}")

print("=" * 80)
