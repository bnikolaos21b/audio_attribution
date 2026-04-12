"""
Autocorrelation persistence test across re-encoding chain.
Checks whether the Suno 17ms artifact survives lossy transcoding.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
SR          = 22050
CLIP_SEC    = 30
MAX_LAG_MS  = 30
PEAK_TARGET = 17.0     # ms — the Suno artifact we're tracking
PEAK_WIN_MS = 2.0      # ±2 ms search window around target

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_AUDIO_DIR    = Path(os.environ.get("SUNO_AUDIO_DIR", str(_PROJECT_ROOT / "data" / "suno")))
OUT_PNG       = _PROJECT_ROOT / "results" / "plots" / "suno_reencode_autocorr.png"

VERSIONS = [
    ("Original\n(Suno ~188 kbps)",
     str(_AUDIO_DIR / "Candle Wax Strings (1).mp3"),
     "#2C3E50"),
    ("Re-encode → 128 kbps",
     "/tmp/suno_128.mp3",
     "#E74C3C"),
    ("Re-encode → 320 kbps\n(from 128)",
     "/tmp/suno_320.mp3",
     "#27AE60"),
    ("Re-encode → 192 kbps\n(from 320)",
     "/tmp/suno_192.mp3",
     "#8E44AD"),
]

# ── Autocorrelation ───────────────────────────────────────────────────────────

def compute_autocorr(path):
    total  = librosa.get_duration(path=str(path))
    offset = max(0.0, (total - CLIP_SEC) / 2.0)
    y, _   = librosa.load(str(path), sr=SR, offset=offset,
                           duration=CLIP_SEC, mono=True)
    max_lag = int(MAX_LAG_MS * SR / 1000)
    y_z     = y - np.mean(y)
    n       = len(y_z)
    # FFT-based normalised autocorrelation
    ac_full = np.real(np.fft.irfft(np.abs(np.fft.rfft(y_z, n=2 * n)) ** 2))
    ac      = ac_full[:max_lag + 1]
    ac     /= ac[0] + 1e-12
    lags_ms = np.arange(max_lag + 1) / SR * 1000
    return lags_ms, ac

print("Computing autocorrelations…\n")
results = []
for label, path, color in VERSIONS:
    short = label.replace("\n", " ")
    print(f"  {short}")
    lags_ms, ac = compute_autocorr(path)
    results.append((label, path, color, lags_ms, ac))

# ── Zoom mask ─────────────────────────────────────────────────────────────────
lags_ms = results[0][3]
zoom_mask = (lags_ms >= 10) & (lags_ms <= 25)
fz = lags_ms[zoom_mask]

# ── Peak tracking ─────────────────────────────────────────────────────────────

def find_peak_near(lags, ac, target_ms, win_ms):
    """Return (lag_ms, ac_value) of the highest peak within target±win, or None."""
    mask = (lags >= target_ms - win_ms) & (lags <= target_ms + win_ms)
    if not mask.any():
        return None, None
    local_ac = ac[mask]
    local_lg = lags[mask]
    idx      = np.argmax(local_ac)
    return float(local_lg[idx]), float(local_ac[idx])

print("\n── Peak tracking at 17 ms ──────────────────────────────────────────")
orig_peak_val = None
rows = []
for label, path, color, lags, ac in results:
    pk_lag, pk_val = find_peak_near(lags, ac, PEAK_TARGET, PEAK_WIN_MS)
    short = label.replace("\n", " ")
    if pk_lag is not None:
        if orig_peak_val is None:
            orig_peak_val = pk_val
        attn = (orig_peak_val - pk_val) / (orig_peak_val + 1e-12) * 100
        rows.append((short, pk_lag, pk_val, attn))
    else:
        rows.append((short, None, None, None))

# Also find the dominant peak in 10–25 ms for each version
print()
print(f"{'Version':<36} {'Peak lag':>9} {'AC value':>10} {'Attn vs orig':>14}  Survives?")
print("─" * 76)
for short, pk_lag, pk_val, attn in rows:
    if pk_lag is None:
        print(f"{short:<36}  {'—':>9}  {'—':>10}  {'—':>14}  —")
    else:
        survives = "YES ✓" if pk_val > 0.03 else "NO  ✗"
        print(f"{short:<36}  {pk_lag:>8.1f}ms  {pk_val:>10.5f}  {attn:>+13.1f}%  {survives}")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(13, 11), gridspec_kw={"height_ratios": [3, 2]})
fig.suptitle(
    "Re-encoding Autocorrelation Test — Candle Wax Strings (1).mp3\n"
    "Does the 17 ms Suno artifact survive lossy transcoding?",
    fontsize=13, fontweight="bold"
)

# ── Panel 1: overlaid autocorrelations, zoomed 10–25 ms ──────────────────────
ax = axes[0]
for i, (label, path, color, lags, ac) in enumerate(results):
    lw  = 2.8 if i == 0 else 1.8
    ls  = "-"  if i == 0 else ["--", "-.", ":"][i - 1]
    ax.plot(fz, ac[zoom_mask], color=color, lw=lw, ls=ls, label=label, zorder=4 - i)

ax.axvline(PEAK_TARGET, color="black", ls="--", lw=1.4, alpha=0.7,
           label="17 ms marker")
ax.axvspan(PEAK_TARGET - PEAK_WIN_MS, PEAK_TARGET + PEAK_WIN_MS,
           color="gold", alpha=0.15, label="±2 ms search window")

# Annotate each peak value
for label, path, color, lags, ac in results:
    pk_lag, pk_val = find_peak_near(lags, ac, PEAK_TARGET, PEAK_WIN_MS)
    if pk_lag is not None and pk_val > 0.01:
        ax.annotate(f"{pk_val:.4f}",
                    xy=(pk_lag, pk_val),
                    xytext=(pk_lag + 0.4, pk_val + 0.008),
                    fontsize=7.5, color=color,
                    arrowprops=dict(arrowstyle="->", color=color, lw=0.9))

ax.set_xlabel("Lag (ms)", fontsize=11)
ax.set_ylabel("Normalised autocorrelation", fontsize=11)
ax.set_title("Zoomed 10–25 ms: autocorrelation across encoding generations", fontsize=10)
ax.set_xlim(10, 25)
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, alpha=0.3)

# ── Panel 2: peak value bar chart per version ─────────────────────────────────
ax = axes[1]
labels_short = []
peak_vals    = []
colors_bar   = []
for label, path, color, lags, ac in results:
    pk_lag, pk_val = find_peak_near(lags, ac, PEAK_TARGET, PEAK_WIN_MS)
    labels_short.append(label.replace("\n", "\n"))
    peak_vals.append(pk_val if pk_val is not None else 0.0)
    colors_bar.append(color)

x = np.arange(len(labels_short))
bars = ax.bar(x, peak_vals, color=colors_bar, width=0.55, alpha=0.85, zorder=3)
ax.bar_label(bars, fmt="%.5f", padding=3, fontsize=9)

# Reference line: 0.03 "survives" threshold
ax.axhline(0.03, color="red", ls="--", lw=1.2, alpha=0.7, label="Survival threshold (AC=0.03)")
ax.axhline(orig_peak_val, color="#2C3E50", ls=":", lw=1.2, alpha=0.5,
           label=f"Original peak ({orig_peak_val:.5f})")

ax.set_xticks(x)
ax.set_xticklabels(labels_short, fontsize=9)
ax.set_ylabel("Peak AC value at 17 ms", fontsize=10)
ax.set_title("Peak amplitude at 17 ms per encoding generation", fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, axis="y", alpha=0.3)
ax.set_ylim(0, max(peak_vals) * 1.25)

plt.tight_layout()
plt.savefig(str(OUT_PNG), dpi=150, bbox_inches="tight")
print(f"\nFigure saved → {OUT_PNG}")
