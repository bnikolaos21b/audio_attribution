"""
Stage 1: AI Generation Pre-Filter
==================================
Detects whether an audio track was AI-generated using acoustic fingerprints
established in the Orfium Suno analysis:

  1. Autocorrelation peak in 10–25ms (neural vocoder frame artifact)
  2. Harmonic-to-Noise Ratio (HNR) — AI generators produce unnaturally clean harmonics
  3. Mid/Side power ratio — AI stereo is often artificially collapsed or wide

These features are generator-agnostic: they detect neural synthesis, not a specific model.

HNR formula (validated at margin=2.0, 3.0, 4.0 — all p<0.004 on 26 folk + 5 real):
  librosa.effects.hpss(y, margin=3.0) → RMS frames → voiced-frame mean ratio

Autocorrelation window 10–25ms captures vocoder frame rates (40–100 Hz).
Going wider picks up musical periodicity (e.g. 27ms = ~37Hz rhythmic strumming).
Going narrower risks clipping version drift (v2=15.7ms through v3.5=19.0ms).

Validated on 26 folk Suno + 5 real folk (p=0.003, d=1.34) and 160 SONICS tracks.
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np
import librosa


# ── Thresholds (validated on 26 folk + 5 real + 160 SONICS) ───────────────────

AUTOCORR_PEAK_MIN     = 0.03    # Normalised AC value threshold
AUTOCORR_LO_MS        = 10.0    # Search window low bound (ms)
AUTOCORR_HI_MS        = 25.0    # Search window high bound (ms)

HNR_MARGIN            = 3.0     # HPSS margin parameter
HNR_THRESHOLD         = 15.0    # Linear ratio (harmonic RMS / percussive RMS)
                                # Above this → likely AI. Separation confirmed
                                # at margin=2,3,4 (all p<0.004).

MID_SIDE_RATIO_THRESH = 0.95    # L-R Pearson correlation > 0.95 → fake stereo


@dataclass
class AIDetectionResult:
    is_likely_ai: bool
    confidence: float          # [0, 1] — aggregate confidence
    autocorr_peak_value: Optional[float]
    autocorr_peak_lag_ms: Optional[float]
    hnr_ratio: Optional[float]   # Linear ratio (not dB)
    mid_side_correlation: Optional[float]
    evidence_count: int        # How many of 3 indicators fired
    notes: str


def _load_stereo(path: str, sr: int = 22050, clip_sec: int = 30):
    """Load middle clip_sec seconds as stereo (2, N) array."""
    total = librosa.get_duration(path=str(path))
    offset = max(0.0, (total - clip_sec) / 2.0)
    y, _ = librosa.load(str(path), sr=sr, offset=offset, duration=clip_sec, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])
    return y


def _load_mono(path: str, sr: int = 22050, clip_sec: int = 30):
    """Load middle clip_sec seconds as mono."""
    total = librosa.get_duration(path=str(path))
    offset = max(0.0, (total - clip_sec) / 2.0)
    y, _ = librosa.load(str(path), sr=sr, offset=offset, duration=clip_sec, mono=True)
    return y


def compute_autocorrelation_peak(y: np.ndarray, sr: int,
                                  lo_ms: float = AUTOCORR_LO_MS,
                                  hi_ms: float = AUTOCORR_HI_MS) -> tuple:
    """
    Compute normalised waveform autocorrelation and find peak in [lo, hi] ms.
    Returns (peak_lag_ms, peak_value).

    The peak comes from the neural vocoder's frame-rate periodicity.
    Window 10–25ms captures vocoder frame rates (40–100 Hz).
    Going wider picks up musical periodicity (e.g. 27ms = ~37Hz strumming).
    Found in ~96–100% of both Suno and Udio tracks at scale.
    """
    max_lag = int(30 * sr / 1000)
    y_z = y - np.mean(y)
    n = len(y_z)
    ac_full = np.real(np.fft.irfft(np.abs(np.fft.rfft(y_z, n=2 * n)) ** 2))
    ac = ac_full[:max_lag + 1]
    ac /= (ac[0] + 1e-12)
    lags_ms = np.arange(max_lag + 1) / sr * 1000

    mask = (lags_ms >= lo_ms) & (lags_ms <= hi_ms)
    if not mask.any():
        return None, None

    idx = np.argmax(ac[mask])
    return float(lags_ms[mask][idx]), float(ac[mask][idx])


def compute_hnr(y: np.ndarray, sr: int = 22050,
                margin: float = HNR_MARGIN,
                frame_length: int = 2048, hop_length: int = 512) -> float:
    """
    Harmonic-to-Noise Ratio via HPSS with voiced-frame filtering.

    This is the formula from the Orfium analysis (extended_analysis.py):
      1. librosa.effects.hpss(y, margin=margin) — separate harmonic/percussive
      2. Compute RMS per frame for each component
      3. Select "voiced" frames (harmonic RMS > median)
      4. Return mean(harmonic_RMS / percussive_RMS) over voiced frames

    AI generators produce unnaturally clean harmonic content because the
    neural vocoder generates near-periodic waveforms without the
    micro-perturbations of real instruments.

    Validated on 26 folk + 5 real:
      Real:  mean 16.44, range [10.82–24.30]   (margin=3.0)
      Suno:  mean 31.54, range [15.33–83.80]   (margin=3.0)
      Mann-Whitney p=0.003, Cohen's d=1.34

    Robust across margin=2.0 (p=0.0038), 3.0 (p=0.0030), 4.0 (p=0.0023).
    """
    y_harm, y_perc = librosa.effects.hpss(y, margin=margin)
    rms_harm = librosa.feature.rms(y=y_harm, frame_length=frame_length, hop_length=hop_length)[0]
    rms_perc = librosa.feature.rms(y=y_perc, frame_length=frame_length, hop_length=hop_length)[0]
    # Voiced frames: harmonic RMS above median
    voiced = rms_harm > np.median(rms_harm)
    if not voiced.any():
        return float("nan")
    return float(np.mean(rms_harm[voiced] / (rms_perc[voiced] + 1e-12)))


def compute_mid_side_correlation(y_stereo: np.ndarray) -> float:
    """
    Pearson correlation between left and right channels.

    Real stereo recordings have natural L-R differences from microphone placement
    and room acoustics. AI generators often produce near-identical channels
    (collapsed mono) or artificially widened stereo.

    PRELIMINARY — validated on 26 stereo Suno folk tracks only.
    SONICS re-encoded all audio to mono 128 kbps, so this feature cannot be
    verified at scale.  Do not treat as a validated generator fingerprint
    for mono or unknown-format inputs.
    """
    if y_stereo.shape[0] < 2:
        return float("nan")
    left = y_stereo[0].astype(np.float64)
    right = y_stereo[1].astype(np.float64)

    max_samples = 22050 * 30
    if len(left) > max_samples:
        step = len(left) // max_samples
        left = left[::step]
        right = right[::step]

    left_c = left - np.mean(left)
    right_c = right - np.mean(right)
    numerator = np.sum(left_c * right_c)
    denominator = np.sqrt(np.sum(left_c**2) * np.sum(right_c**2)) + 1e-12
    return float(numerator / denominator)


def detect_ai(path: str, sr: int = 22050, clip_sec: int = 30) -> AIDetectionResult:
    """
    Run all three AI detection features on a single audio file.

    Returns AIDetectionResult with individual feature values and aggregate confidence.
    """
    path = str(path)

    # ── Feature 1: Autocorrelation peak ───────────────────────────────────
    y_mono = _load_mono(path, sr, clip_sec)
    peak_lag, peak_val = compute_autocorrelation_peak(y_mono, sr)
    has_autocorr = peak_val is not None and peak_val > AUTOCORR_PEAK_MIN

    # ── Feature 2: HNR (linear ratio, not dB) ────────────────────────────
    hnr = compute_hnr(y_mono, sr)
    has_hnr = not np.isnan(hnr) and hnr > HNR_THRESHOLD

    # ── Feature 3: Mid/Side correlation ───────────────────────────────────
    y_stereo = _load_stereo(path, sr, clip_sec)
    ms_corr = compute_mid_side_correlation(y_stereo)
    has_fake_stereo = not np.isnan(ms_corr) and ms_corr > MID_SIDE_RATIO_THRESH

    # ── Aggregate ─────────────────────────────────────────────────────────
    evidence = sum([has_autocorr, has_hnr, has_fake_stereo])

    # Confidence: per-feature scoring → combined
    scores = []

    if peak_val is not None:
        # Autocorrelation confidence: 0.03 → 0.5+ maps to 0 → 1
        autocorr_conf = min(1.0, max(0.0, (peak_val - 0.01) / 0.30))
        scores.append(autocorr_conf)

    if hnr is not None and not np.isnan(hnr):
        # HNR confidence: 10 → 60+ maps to 0 → 1
        # (validated ranges: real 10–24, suno 15–84)
        hnr_conf = min(1.0, max(0.0, (hnr - 10) / 50))
        scores.append(hnr_conf)

    if not np.isnan(ms_corr):
        # MS correlation confidence: 0.90 → 1.0 maps to 0 → 1
        ms_conf = min(1.0, max(0.0, (ms_corr - 0.85) / 0.15))
        scores.append(ms_conf)

    confidence = float(np.mean(scores)) if scores else 0.0

    # Boost confidence if multiple indicators agree
    if evidence >= 2:
        confidence = min(1.0, confidence * 1.3)
    if evidence == 3:
        confidence = min(1.0, confidence * 1.5)

    is_ai = evidence >= 2 or confidence > 0.7

    notes_parts = []
    if has_autocorr and peak_lag is not None:
        notes_parts.append(f"vocoder peak at {peak_lag:.1f}ms (AC={peak_val:.4f})")
    if has_hnr:
        notes_parts.append(f"HNR elevated ({hnr:.1f})")
    if has_fake_stereo:
        notes_parts.append(f"L-R correlation too high ({ms_corr:.3f})")
    if not notes_parts:
        notes_parts.append("No strong AI indicators detected")

    return AIDetectionResult(
        is_likely_ai=is_ai,
        confidence=round(confidence, 4),
        autocorr_peak_value=round(peak_val, 5) if peak_val is not None else None,
        autocorr_peak_lag_ms=round(peak_lag, 2) if peak_lag is not None else None,
        hnr_ratio=round(hnr, 2) if hnr is not None and not np.isnan(hnr) else None,
        mid_side_correlation=round(ms_corr, 4) if not np.isnan(ms_corr) else None,
        evidence_count=evidence,
        notes="; ".join(notes_parts),
    )
