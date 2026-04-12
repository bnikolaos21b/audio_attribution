"""
Cross-generator acoustic fingerprint comparison.
Groups: Folk Suno (26), SONICS Suno (72, chirp-v3.5), SONICS Udio (88, udio-120s).

Features per track (middle 30 s):
  autocorr_peak_ms    — lag of dominant peak in 10–25 ms window (vocoder frame rate)
  autocorr_peak_val   — amplitude of that peak
  hnr                 — harmonic/percussive RMS (HPSS)
  spectral_flatness   — mean Wiener entropy (0=tonal, 1=white noise)
  ms_ratio            — mid/side energy ratio (stereo width proxy)
  crest_db            — peak-to-RMS ratio in dB
  lufs                — integrated loudness (pyloudnorm)
  noise_floor_db      — mean dB of quietest-10% frames
"""

import os
import io
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
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# Folk Suno audio is in Desktop/Orfium/suno/ (26 AI-generated folk tracks)
FOLK_AUDIO_DIR = Path(os.environ.get("SUNO_AUDIO_DIR",
                                     str(Path(__file__).resolve().parent.parent.parent.parent / "suno")))
META_CSV  = _PROJECT_ROOT / "data" / "feature_inventory" / "sonics_baseaware_sample_files.csv"
OUT_PNG   = _PROJECT_ROOT / "results" / "plots" / "generator_fingerprints.png"

SR        = 22050
CLIP_SEC  = 30
HOP       = 512
N_FFT     = 2048
N_MELS    = 128

# ── Feature extraction ─────────────────────────────────────────────────────────

def _load_middle_mono(y_full, sr_full):
    """Resample if needed, take middle CLIP_SEC seconds, return mono float32."""
    if sr_full != SR:
        y_full = librosa.resample(y_full, orig_sr=sr_full, target_sr=SR)
    if y_full.ndim > 1:
        y_full = np.mean(y_full, axis=0)
    n = len(y_full)
    want = int(CLIP_SEC * SR)
    start = max(0, (n - want) // 2)
    y = y_full[start:start + want].astype(np.float32)
    return y

def _load_middle_stereo(y_full, sr_full):
    """Return (2, samples) stereo clip resampled to SR."""
    if sr_full != SR:
        if y_full.ndim == 1:
            y_full = librosa.resample(y_full, orig_sr=sr_full, target_sr=SR)
        else:
            y_full = np.stack([librosa.resample(c, orig_sr=sr_full, target_sr=SR)
                               for c in y_full])
    if y_full.ndim == 1:
        y_full = np.stack([y_full, y_full])
    n = y_full.shape[1]
    want = int(CLIP_SEC * SR)
    start = max(0, (n - want) // 2)
    return y_full[:, start:start + want].astype(np.float32)

def extract_features(y_full, sr_full):
    is_mono_input = (y_full.ndim == 1)          # True for SONICS (16kHz mono)
    effective_sr  = min(int(sr_full), SR)        # actual bandwidth ceiling

    y  = _load_middle_mono(y_full, sr_full)
    ys = _load_middle_stereo(y_full, sr_full)

    feats = {}

    # 1. Waveform autocorrelation — peak in 10–25 ms (vocoder artifact)
    max_lag = int(30 * SR / 1000)
    y_z = y - np.mean(y)
    n   = len(y_z)
    ac_full = np.real(np.fft.irfft(np.abs(np.fft.rfft(y_z, n=2*n))**2))
    ac  = ac_full[:max_lag + 1] / (ac_full[0] + 1e-12)
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

    # 2. HNR — harmonic RMS / percussive RMS (voiced frames)
    y_harm, y_perc = librosa.effects.hpss(y, margin=3.0)
    rms_tot  = librosa.feature.rms(y=y,      frame_length=N_FFT, hop_length=HOP)[0]
    rms_harm = librosa.feature.rms(y=y_harm, frame_length=N_FFT, hop_length=HOP)[0]
    rms_perc = librosa.feature.rms(y=y_perc, frame_length=N_FFT, hop_length=HOP)[0]
    voiced   = rms_tot > np.percentile(rms_tot, 10)
    feats["hnr"] = float(np.mean(rms_harm[voiced] / (rms_perc[voiced] + 1e-12)))

    # 3. Spectral flatness — computed only up to effective Nyquist so that
    #    zero-padded bins introduced when resampling 16 kHz → 22050 Hz don't
    #    collapse the geometric mean toward zero (SONICS tracks are 16 kHz).
    stft_mag  = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    fft_freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    max_bin   = int(np.searchsorted(fft_freqs, effective_sr / 2))
    stft_bw   = stft_mag[:max_bin] + 1e-10
    geo_mean  = np.exp(np.mean(np.log(stft_bw), axis=0))
    arith_mean = np.mean(stft_bw, axis=0) + 1e-12
    feats["spectral_flatness"] = float(np.mean(geo_mean / arith_mean))

    # 5. Mid/side energy ratio — undefined for mono inputs (side=0 identically);
    #    return NaN so the panel is skipped cleanly for SONICS tracks.
    if is_mono_input:
        feats["ms_ratio"] = float("nan")
    else:
        L, R = ys[0].astype(np.float64), ys[1].astype(np.float64)
        mid  = (L + R) / 2.0
        side = (L - R) / 2.0
        feats["ms_ratio"] = float(np.mean(mid**2) / (np.mean(side**2) + 1e-12))

    # 6. Crest factor (dB)
    peak  = float(np.max(np.abs(np.mean(ys, axis=0))))
    rms_v = float(np.sqrt(np.mean(np.mean(ys, axis=0)**2)))
    feats["crest_db"] = 20 * np.log10(peak / (rms_v + 1e-12) + 1e-12)

    # 7. LUFS
    meter = pyln.Meter(SR)
    try:
        feats["lufs"] = float(meter.integrated_loudness(ys.T.astype(np.float64)))
    except Exception:
        feats["lufs"] = float("nan")

    # 8. Noise floor — mean dB of quietest 10% frames
    frame_rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    frame_db  = librosa.amplitude_to_db(frame_rms, ref=np.max(frame_rms) + 1e-12)
    thresh    = np.percentile(frame_db, 10)
    quiet     = frame_db <= thresh
    feats["noise_floor_db"] = float(np.mean(frame_db[quiet]))

    return feats


# ── Load folk Suno tracks ──────────────────────────────────────────────────────

def load_folk_suno(audio_dir: Path):
    files = sorted(audio_dir.glob("*.mp3"))
    folk_suno = [f for f in files if not f.name.startswith("real_")]
    print(f"Folk Suno: {len(folk_suno)} files from {audio_dir}")
    results = []
    for f in folk_suno:
        try:
            y, sr = librosa.load(str(f), sr=None, mono=False)
            feats = extract_features(y, sr)
            feats["name"] = f.stem
            results.append(feats)
            print(f"  {f.name}")
        except Exception as e:
            print(f"  SKIP {f.name}: {e}")
    return results


# ── Stream SONICS audio from HuggingFace ──────────────────────────────────────

SONICS_ZIP_URL = ("https://huggingface.co/datasets/awsaf49/sonics"
                  "/resolve/main/fake_songs/part_01.zip")

def load_sonics_groups(meta_csv: Path):
    """Extract 160 target tracks from SONICS part_01.zip using remotezip
    (HTTP range requests — no full download required)."""
    from remotezip import RemoteZip

    meta = pd.read_csv(meta_csv)
    # filename column has .mp3 extension; zip stores as fake_songs/<name>.mp3
    target_suno = set(meta[meta.engine == "suno"]["filename"])
    target_udio = set(meta[meta.engine == "udio"]["filename"])
    all_targets  = target_suno | target_udio
    print(f"\nSONICS targets: {len(target_suno)} suno, {len(target_udio)} udio")
    print(f"Opening {SONICS_ZIP_URL} (range requests, no full download)…")

    suno_results, udio_results = [], []

    with RemoteZip(SONICS_ZIP_URL) as z:
        names_in_zip = z.namelist()
        # build lookup: basename -> full zip path
        zip_map = {n.split("/")[-1]: n for n in names_in_zip}
        missing = all_targets - set(zip_map)
        if missing:
            print(f"  WARNING: {len(missing)} targets not found in zip")

        targets_found = all_targets & set(zip_map)
        print(f"  {len(targets_found)} targets located in zip")

        for i, fname in enumerate(sorted(targets_found), 1):
            zip_path = zip_map[fname]
            engine   = "suno" if fname in target_suno else "udio"
            try:
                data = z.read(zip_path)
                y_full, sr_full = librosa.load(io.BytesIO(data), sr=None, mono=False)
                feats = extract_features(y_full, sr_full)
                feats["name"]   = fname.replace(".mp3", "")
                feats["engine"] = engine
                print(f"  [{i:3d}/160] [{engine}] {fname}")
                if engine == "suno":
                    suno_results.append(feats)
                else:
                    udio_results.append(feats)
            except Exception as e:
                print(f"  SKIP {fname}: {e}")

    print(f"SONICS collected: {len(suno_results)} suno, {len(udio_results)} udio")
    return suno_results, udio_results


# ── Aggregate helpers ──────────────────────────────────────────────────────────

FEAT_KEYS = [
    "autocorr_peak_val",
    "autocorr_peak_ms",
    "hnr",
    "spectral_flatness",
    "ms_ratio",
    "crest_db",
    "lufs",
    "noise_floor_db",
]

FEAT_LABELS = {
    "autocorr_peak_val":   "Autocorr peak\n(10–25 ms)",
    "autocorr_peak_ms":    "Autocorr peak lag\n(ms)",
    "hnr":                 "HNR\n(harm/perc RMS)",
    "spectral_flatness":   "Spectral flatness\n(BW-limited to src Nyquist)",
    "ms_ratio":            "Mid/Side ratio\n(NaN for mono — SONICS only)",
    "crest_db":            "Crest factor\n(dB)",
    "lufs":                "LUFS\n(integrated)",
    "noise_floor_db":      "Noise floor\n(quiet 10%, dB)",
}

def col(records, key):
    vals = np.array([r[key] for r in records], dtype=float)
    return vals[np.isfinite(vals)]

def ms(arr):
    a = arr[np.isfinite(arr)]
    return float(np.mean(a)), float(np.std(a))

def cohens_d(a, b):
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return (np.mean(a) - np.mean(b)) / (pooled + 1e-12)


# ── Plotting ───────────────────────────────────────────────────────────────────

C_FOLK  = "#E05C5C"   # folk Suno — red
C_SUNO  = "#F39C12"   # SONICS Suno — orange
C_UDIO  = "#2980B9"   # SONICS Udio — blue

def plot_comparison(folk, s_suno, s_udio):
    n_feats = len(FEAT_KEYS)
    ncols = 3
    nrows = (n_feats + ncols - 1) // ncols  # ceil

    fig = plt.figure(figsize=(20, nrows * 5 + 3))
    fig.suptitle(
        "Acoustic Fingerprint Comparison: Folk Suno vs SONICS Suno vs SONICS Udio\n"
        "Folk Suno: 26 tracks (chirp, stereo 44.1 kHz) | "
        "SONICS Suno: 72 tracks (chirp-v3.5, mono 16 kHz) | "
        "SONICS Udio: 88 tracks (udio-120s, mono 16 kHz)\n"
        "Note: spectral_flatness BW-limited to source Nyquist; ms_ratio undefined for mono (shown as NaN)",
        fontsize=11, fontweight="bold", y=0.998,
    )

    gs = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.60, wspace=0.38)

    groups = [
        (folk,   C_FOLK,  f"Folk\nSuno\n(n={len(folk)})"),
        (s_suno, C_SUNO,  f"SONICS\nSuno\n(n={len(s_suno)})"),
        (s_udio, C_UDIO,  f"SONICS\nUdio\n(n={len(s_udio)})"),
    ]

    rng = np.random.default_rng(42)

    for idx, key in enumerate(FEAT_KEYS):
        row, c = divmod(idx, ncols)
        ax = fig.add_subplot(gs[row, c])

        all_vals = [col(g, key) for g, _, _ in groups]

        # violin + strip
        positions = [1, 2, 3]
        for pos, (records, color, label), vals in zip(positions, groups, all_vals):
            if len(vals) < 2:
                continue
            parts = ax.violinplot([vals], positions=[pos], widths=0.55,
                                  showmedians=True, showextrema=True)
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.40)
            for part in ["cmedians", "cmaxes", "cmins", "cbars"]:
                parts[part].set_color(color)
                parts[part].set_linewidth(1.2)
            jitter = rng.uniform(-0.10, 0.10, len(vals))
            ax.scatter(pos + jitter, vals, color=color, alpha=0.55, s=12, zorder=3)

        ax.set_xticks(positions)
        ax.set_xticklabels(["Folk\nSuno", "SONICS\nSuno", "SONICS\nUdio"], fontsize=8)
        ax.set_title(FEAT_LABELS[key], fontsize=9)
        ax.grid(True, axis="y", alpha=0.25)

        # Annotate Cohen's d (SONICS Suno vs Udio) in corner
        d = cohens_d(all_vals[1], all_vals[2])
        if np.isfinite(d):
            ax.text(0.97, 0.97, f"d={d:+.2f}", transform=ax.transAxes,
                    fontsize=7.5, ha="right", va="top",
                    color="gray", style="italic")

    # Legend panel in last unused slot (if any)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=C_FOLK,  label="Folk Suno (26 folk tracks, Suno AI)"),
        Patch(facecolor=C_SUNO,  label="SONICS Suno (72 tracks, chirp-v3.5)"),
        Patch(facecolor=C_UDIO,  label="SONICS Udio (88 tracks, udio-120s)"),
    ]
    # Place legend in a remaining subplot or as figlegend
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               fontsize=10, frameon=True, bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(OUT_PNG), dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {OUT_PNG}")


# ── Summary table ──────────────────────────────────────────────────────────────

def print_summary(folk, s_suno, s_udio):
    groups = [
        ("Folk Suno",   folk),
        ("SONICS Suno", s_suno),
        ("SONICS Udio", s_udio),
    ]

    print("\n" + "=" * 100)
    print("  GENERATOR FINGERPRINT SUMMARY")
    print("=" * 100)
    hdr = f"{'Feature':<24}" + "".join(f"  {'mean±std':>22}" for _, _ in groups)
    print(f"{'Feature':<24}" +
          "".join(f"  {name:>22}" for name, _ in groups) +
          "  d(Suno↔Udio)  d(Folk↔Suno)")
    print("-" * 100)

    for key in FEAT_KEYS:
        row = f"{key:<24}"
        vals = [col(g, key) for _, g in groups]
        for v in vals:
            mn, sd = ms(v)
            row += f"  {mn:>9.4f} ±{sd:>9.4f}"
        d_su = cohens_d(vals[1], vals[2])  # SONICS Suno vs Udio
        d_fs = cohens_d(vals[0], vals[1])  # Folk vs SONICS Suno
        row += f"  {d_su:>+12.2f}  {d_fs:>+12.2f}"
        print(row)

    print("=" * 100)
    print("d = Cohen's d (Suno − Udio) and (Folk − SONICS Suno). |d|>0.8 = large effect.")
    print()

    # Autocorr detail
    print("── Autocorrelation peak prevalence (10–25 ms window, peak_val > 0.03) ──")
    for name, g in groups:
        peaks = col(g, "autocorr_peak_val")
        frac  = np.mean(peaks > 0.03)
        median_lag = np.nanmedian(col(g, "autocorr_peak_ms"))
        print(f"  {name:<16}  {frac*100:.0f}% have peak>0.03   median lag={median_lag:.1f} ms")
    print()

    # LUFS detail
    print("── LUFS clustering ──")
    for name, g in groups:
        lufs = col(g, "lufs")
        lufs = lufs[np.isfinite(lufs)]
        near_14 = np.mean((lufs >= -15) & (lufs <= -13)) * 100
        print(f"  {name:<16}  mean={np.mean(lufs):.1f}±{np.std(lufs):.1f} LUFS   "
              f"{near_14:.0f}% near −14 LUFS (±1)")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Generator Fingerprint Analysis")
    print("=" * 60)

    # 1. Folk Suno
    print("\n── Step 1: Folk Suno tracks ──")
    folk_results = load_folk_suno(FOLK_AUDIO_DIR)

    # 2. SONICS Suno + Udio via HF streaming
    print("\n── Step 2: SONICS tracks (streaming from HuggingFace) ──")
    sonics_suno, sonics_udio = load_sonics_groups(META_CSV)

    # 3. Plot
    print("\n── Step 3: Plotting ──")
    plot_comparison(folk_results, sonics_suno, sonics_udio)

    # 4. Summary
    print_summary(folk_results, sonics_suno, sonics_udio)
