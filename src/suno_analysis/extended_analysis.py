#!/usr/bin/env python3
"""
Extended analysis of Suno AI-generated vs real folk recordings.
Analyses:
  1. Stereo field: L-R correlation and mid/side energy ratio
  2. Onset sharpness: mean onset strength and attack rise time (ms)
  3. Harmonic-to-Noise Ratio (HNR): harmonic RMS / percussive RMS in voiced frames
  4. True silence detection: frames below -90 dB
"""

import os
import glob
import warnings
from pathlib import Path
import numpy as np
import librosa
import librosa.effects
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')

# ── Parameters ────────────────────────────────────────────────────────────────
SR         = 22050
HOP_LENGTH = 512
N_FFT      = 2048
SEGMENT_S  = 30          # middle 30 seconds
SILENCE_DB = -90.0       # threshold for "true silence"

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BASE_DIR      = os.environ.get("SUNO_AUDIO_DIR", str(_PROJECT_ROOT / "data" / "suno"))
_OUT_PNG      = str(_PROJECT_ROOT / "results" / "plots" / "suno_extended_analysis.png")

# ── File discovery ─────────────────────────────────────────────────────────────
all_mp3s = sorted(glob.glob(os.path.join(BASE_DIR, "*.mp3")))
real_files = [f for f in all_mp3s if os.path.basename(f).startswith("real_")]
suno_files = [f for f in all_mp3s if not os.path.basename(f).startswith("real_")]

print(f"Found {len(suno_files)} Suno files and {len(real_files)} real files.\n")

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_middle_mono(path):
    """Load middle 30 s of a file as mono at SR=22050."""
    duration = librosa.get_duration(path=path)
    start = max(0.0, (duration - SEGMENT_S) / 2.0)
    y, _ = librosa.load(path, sr=SR, mono=True,
                         offset=start, duration=SEGMENT_S)
    return y

def load_middle_stereo(path):
    """Load middle 30 s as stereo (2, N) at SR=22050. Falls back to mono duplication."""
    duration = librosa.get_duration(path=path)
    start = max(0.0, (duration - SEGMENT_S) / 2.0)
    y, _ = librosa.load(path, sr=SR, mono=False,
                         offset=start, duration=SEGMENT_S)
    if y.ndim == 1:
        y = np.stack([y, y])
    return y  # shape (channels, samples)

# ── Analysis 1: Stereo field ───────────────────────────────────────────────────

def stereo_metrics(path):
    """Returns (lr_corr, ms_ratio) where ms_ratio = mid_energy / side_energy."""
    y = load_middle_stereo(path)
    L = y[0].astype(np.float64)
    R = y[1].astype(np.float64) if y.shape[0] > 1 else y[0].astype(np.float64)

    # L-R correlation
    if np.std(L) < 1e-9 or np.std(R) < 1e-9:
        lr_corr = 1.0
    else:
        lr_corr = float(np.corrcoef(L, R)[0, 1])

    # Mid = (L+R)/2, Side = (L-R)/2
    mid  = (L + R) / 2.0
    side = (L - R) / 2.0
    mid_energy  = float(np.mean(mid**2))
    side_energy = float(np.mean(side**2))
    ms_ratio = mid_energy / (side_energy + 1e-12)

    return lr_corr, ms_ratio

# ── Analysis 2: Onset sharpness ───────────────────────────────────────────────

def onset_metrics(path):
    """Returns (mean_onset_strength, mean_rise_time_ms)."""
    y = load_middle_mono(path)

    # Onset envelope and strength
    onset_env = librosa.onset.onset_strength(y=y, sr=SR,
                                              hop_length=HOP_LENGTH,
                                              n_fft=N_FFT)
    mean_strength = float(np.mean(onset_env))

    # Detected onset frames
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env,
                                               sr=SR,
                                               hop_length=HOP_LENGTH,
                                               backtrack=True)

    # Rise time: from onset frame to peak within next 100 ms
    window_samples = int(0.100 * SR)          # 100 ms in samples
    hop            = HOP_LENGTH
    rise_times_ms  = []

    for of in onset_frames:
        onset_sample = of * hop
        end_sample   = min(onset_sample + window_samples, len(y))
        window        = np.abs(y[onset_sample:end_sample])
        if len(window) == 0:
            continue
        peak_offset = int(np.argmax(window))
        rise_ms     = peak_offset / SR * 1000.0
        rise_times_ms.append(rise_ms)

    mean_rise = float(np.mean(rise_times_ms)) if rise_times_ms else np.nan
    return mean_strength, mean_rise

# ── Analysis 3: Harmonic-to-Noise Ratio ───────────────────────────────────────

def hnr_metric(path):
    """Returns mean HNR (harmonic RMS / percussive RMS) over voiced frames."""
    y = load_middle_mono(path)

    # HPSS
    y_harm, y_perc = librosa.effects.hpss(y, margin=3.0)

    # Frame RMS for total, harmonic, percussive
    rms_total = librosa.feature.rms(y=y,      frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
    rms_harm  = librosa.feature.rms(y=y_harm, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
    rms_perc  = librosa.feature.rms(y=y_perc, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]

    # Voiced frames: total RMS > 10th percentile
    threshold = np.percentile(rms_total, 10)
    voiced    = rms_total > threshold

    harm_voiced = rms_harm[voiced]
    perc_voiced = rms_perc[voiced]

    if len(harm_voiced) == 0:
        return np.nan

    hnr = float(np.mean(harm_voiced / (perc_voiced + 1e-12)))
    return hnr

# ── Analysis 4: True silence detection ────────────────────────────────────────

def silence_metrics(path):
    """
    Returns (silence_frame_count, silence_fraction) for stereo mix.
    A frame is 'true silence' if both channels have RMS below -90 dB.
    """
    y_stereo = load_middle_stereo(path)
    # Mix to mono for frame-level check across both channels
    y_L = y_stereo[0]
    y_R = y_stereo[1] if y_stereo.shape[0] > 1 else y_stereo[0]

    def frame_db(channel):
        rms = librosa.feature.rms(y=channel, frame_length=N_FFT,
                                   hop_length=HOP_LENGTH)[0]
        with np.errstate(divide='ignore'):
            db = 20.0 * np.log10(rms + 1e-12)
        return db

    db_L = frame_db(y_L)
    db_R = frame_db(y_R)

    # Both channels silent
    silent_frames = np.sum((db_L < SILENCE_DB) & (db_R < SILENCE_DB))
    total_frames  = len(db_L)
    fraction      = float(silent_frames) / total_frames

    return int(silent_frames), float(fraction)

# ── Run all analyses ───────────────────────────────────────────────────────────

def analyse_all(file_list, label):
    results = []
    for path in file_list:
        name = os.path.basename(path)
        print(f"  [{label}] {name}")
        try:
            lr_corr, ms_ratio   = stereo_metrics(path)
            mean_str, rise_ms   = onset_metrics(path)
            hnr                 = hnr_metric(path)
            sil_frames, sil_frac = silence_metrics(path)
        except Exception as e:
            print(f"    ERROR: {e}")
            continue
        results.append({
            'name':       name,
            'label':      label,
            'lr_corr':    lr_corr,
            'ms_ratio':   ms_ratio,
            'onset_str':  mean_str,
            'rise_ms':    rise_ms,
            'hnr':        hnr,
            'sil_frames': sil_frames,
            'sil_frac':   sil_frac,
        })
    return results

print("=== Running analyses ===\n")
suno_results = analyse_all(suno_files, 'Suno')
real_results = analyse_all(real_files, 'Real')
all_results  = suno_results + real_results

# ── Summary statistics ─────────────────────────────────────────────────────────

def stats(results, key):
    vals = [r[key] for r in results if not np.isnan(r[key])]
    if not vals:
        return np.nan, np.nan
    return float(np.mean(vals)), float(np.std(vals))

metrics = [
    ('lr_corr',   'L-R Correlation',      '.4f'),
    ('ms_ratio',  'Mid/Side Energy Ratio', '.2f'),
    ('onset_str', 'Mean Onset Strength',   '.4f'),
    ('rise_ms',   'Mean Rise Time (ms)',   '.2f'),
    ('hnr',       'HNR (harm/perc RMS)',   '.3f'),
    ('sil_frames','Silence Frames (count)','.1f'),
    ('sil_frac',  'Silence Fraction',      '.5f'),
]

print("\n" + "="*72)
print(f"{'Metric':<28} {'Suno mean±std':>22}  {'Real mean±std':>22}")
print("="*72)

for key, label, fmt in metrics:
    sm, ss = stats(suno_results, key)
    rm, rs = stats(real_results, key)
    fmt_s = f"{sm:{fmt}} ± {ss:{fmt}}" if not np.isnan(sm) else "N/A"
    fmt_r = f"{rm:{fmt}} ± {rs:{fmt}}" if not np.isnan(rm) else "N/A"
    print(f"{label:<28} {fmt_s:>22}  {fmt_r:>22}")

print("="*72)

# ── Binary findings ────────────────────────────────────────────────────────────

print("\n=== Binary Findings ===\n")

# 1. Fake stereo: L-R correlation > 0.98
suno_high_corr = [r for r in suno_results if r['lr_corr'] > 0.98]
real_high_corr = [r for r in real_results if r['lr_corr'] > 0.98]
print(f"[Stereo] Files with L-R correlation > 0.98 (likely fake stereo):")
print(f"  Suno: {len(suno_high_corr)}/{len(suno_results)}")
for r in suno_high_corr:
    print(f"    {r['name']}  lr_corr={r['lr_corr']:.4f}")
print(f"  Real: {len(real_high_corr)}/{len(real_results)}")
for r in real_high_corr:
    print(f"    {r['name']}  lr_corr={r['lr_corr']:.4f}")

# 2. Onset softness: rise_ms > 15 ms considered "soft"
suno_soft = [r for r in suno_results if not np.isnan(r['rise_ms']) and r['rise_ms'] > 15]
real_soft  = [r for r in real_results  if not np.isnan(r['rise_ms']) and r['rise_ms'] > 15]
print(f"\n[Onset] Files with mean rise time > 15 ms (soft onsets):")
print(f"  Suno: {len(suno_soft)}/{len(suno_results)}")
print(f"  Real: {len(real_soft)}/{len(real_results)}")

# 3. Unnaturally high HNR: > 2x real mean
rm_hnr, _ = stats(real_results, 'hnr')
if not np.isnan(rm_hnr):
    threshold_hnr = rm_hnr * 2.0
    suno_high_hnr = [r for r in suno_results if not np.isnan(r['hnr']) and r['hnr'] > threshold_hnr]
    print(f"\n[HNR] Files with HNR > 2× real mean ({threshold_hnr:.3f}):")
    print(f"  Suno: {len(suno_high_hnr)}/{len(suno_results)}")
    for r in suno_high_hnr:
        print(f"    {r['name']}  HNR={r['hnr']:.3f}")
else:
    print("\n[HNR] Could not compute real mean HNR.")

# 4. True silence
suno_silent = [r for r in suno_results if r['sil_frames'] > 0]
real_silent  = [r for r in real_results  if r['sil_frames'] > 0]
print(f"\n[Silence] Files with any true-silence frames (< {SILENCE_DB} dB):")
print(f"  Suno: {len(suno_silent)}/{len(suno_results)}")
for r in suno_silent:
    print(f"    {r['name']}  frames={r['sil_frames']}  frac={r['sil_frac']:.5f}")
print(f"  Real: {len(real_silent)}/{len(real_results)}")
for r in real_silent:
    print(f"    {r['name']}  frames={r['sil_frames']}  frac={r['sil_frac']:.5f}")

# ── Plotting ──────────────────────────────────────────────────────────────────

def extract(results, key):
    return np.array([r[key] for r in results])

suno_data = {k: extract(suno_results, k) for k, *_ in metrics}
real_data  = {k: extract(real_results,  k) for k, *_ in metrics}

fig = plt.figure(figsize=(18, 14))
fig.suptitle("Extended Forensic Analysis: Suno AI vs Real Folk Recordings",
             fontsize=15, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.38)

suno_color = '#e05c5c'
real_color = '#4a90d9'
alpha_violin = 0.75

plot_specs = [
    # (row, col, key, ylabel, title)
    (0, 0, 'lr_corr',   'Pearson r',         '1. L-R Channel Correlation'),
    (0, 1, 'ms_ratio',  'Mid / Side energy', '1. Mid/Side Energy Ratio'),
    (0, 2, 'onset_str', 'Strength (a.u.)',    '2. Mean Onset Strength'),
    (0, 3, 'rise_ms',   'Rise time (ms)',     '2. Mean Attack Rise Time'),
    (1, 0, 'hnr',       'Harm / Perc RMS',   '3. Harmonic-to-Noise Ratio'),
    (1, 1, 'sil_frames','Frame count',        '4. True-Silence Frames'),
    (1, 2, 'sil_frac',  'Fraction',           '4. True-Silence Fraction'),
]

for row, col, key, ylabel, title in plot_specs:
    ax = fig.add_subplot(gs[row, col])
    sv = suno_data[key]
    rv = real_data[key]

    # violin plot
    parts = ax.violinplot([sv, rv], positions=[1, 2],
                           showmeans=True, showmedians=True, showextrema=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(suno_color if i == 0 else real_color)
        pc.set_alpha(alpha_violin)
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
        if partname in parts:
            parts[partname].set_color('black')
            parts[partname].set_linewidth(1.2)

    # scatter jitter
    jitter_s = np.random.default_rng(42).uniform(-0.12, 0.12, len(sv))
    jitter_r = np.random.default_rng(99).uniform(-0.12, 0.12, len(rv))
    ax.scatter(1 + jitter_s, sv, color=suno_color, edgecolors='k',
               linewidths=0.5, s=28, zorder=5, alpha=0.9)
    ax.scatter(2 + jitter_r, rv, color=real_color, edgecolors='k',
               linewidths=0.5, s=28, zorder=5, alpha=0.9)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Suno', 'Real'], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.4)

# Legend panel in the remaining slot (row=1, col=3)
ax_leg = fig.add_subplot(gs[1, 3])
ax_leg.axis('off')
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=suno_color, edgecolor='k', label=f'Suno (n={len(suno_results)})'),
                   Patch(facecolor=real_color,  edgecolor='k', label=f'Real  (n={len(real_results)})')]
ax_leg.legend(handles=legend_elements, loc='center', fontsize=11,
              title='Source', title_fontsize=12, frameon=True, framealpha=0.9)

# Annotation text
note = (
    "Key thresholds:\n"
    "  • L-R corr > 0.98 → fake stereo flag\n"
    "  • Rise time > 15 ms → soft onset\n"
    "  • HNR > 2× real mean → unnaturally clean\n"
    "  • Any silence frames → digital artifact"
)
ax_leg.text(0.05, 0.2, note, transform=ax_leg.transAxes,
            fontsize=7.5, verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8))

plt.savefig(_OUT_PNG, dpi=150, bbox_inches='tight')
print(f"\nFigure saved to: {_OUT_PNG}")
