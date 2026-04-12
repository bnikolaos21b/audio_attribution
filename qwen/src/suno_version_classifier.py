"""
Multi-Version Suno Classifier
==============================
Identifies which Suno model version generated a track using:
  - Autocorrelation lag (10–25ms window, vocoder artifact)
  - LUFS (integrated loudness)
  - Crest factor
  - HNR (HPSS margin=3.0, voiced-frame RMS ratio)

Version profiles measured from SONICS data:
  chirp-v2:   15.7ms lag, -11.6 LUFS
  chirp-v3:   17.6ms lag, -17.7 LUFS
  chirp-v3.5: 19.0ms lag, -15.7 LUFS
  v4.5-auk:   15.3ms lag, ~-15.5 LUFS  (new architecture — chirp-auk)

IMPORTANT: The lag drift (15.7→17.6→19.0ms) is a population-level trend.
Within-pair variance is ~3ms std, so per-track version ID has low confidence.
Use on a corpus, not for individual verdicts.
"""

import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
from typing import Optional

import numpy as np
import librosa
import pyloudnorm as pyln

from .ai_detector import _load_mono, compute_autocorrelation_peak


def _compute_hnr(y: np.ndarray, sr: int = 22050, margin: float = 3.0) -> float:
    y_harm, y_perc = librosa.effects.hpss(y, margin=margin)
    rms_harm = librosa.feature.rms(y=y_harm, frame_length=2048, hop_length=512)[0]
    rms_perc = librosa.feature.rms(y=y_perc, frame_length=2048, hop_length=512)[0]
    voiced = rms_harm > np.median(rms_harm)
    if not voiced.any():
        return float("nan")
    return float(np.mean(rms_harm[voiced] / (rms_perc[voiced] + 1e-12)))


def _compute_mel_skewness(y: np.ndarray, sr: int, n_mels: int = 128,
                           n_fft: int = 2048, hop_length: int = 512) -> float:
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                          hop_length=hop_length, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mean = float(np.mean(mel_db))
    std = float(np.std(mel_db))
    if std < 1e-9:
        return 0.0
    return float(np.mean(((mel_db - mean) / std) ** 3))


def _compute_crest_factor(y: np.ndarray) -> float:
    peak = np.max(np.abs(y))
    rms = np.sqrt(np.mean(y ** 2))
    crest = peak / (rms + 1e-12)
    return float(20 * np.log10(crest + 1e-12))


def _compute_lufs(y: np.ndarray, sr: int) -> float:
    try:
        meter = pyln.Meter(sr)
        return float(meter.integrated_loudness(y.astype(np.float64)))
    except Exception:
        return float("nan")


# ── Version profiles ──────────────────────────────────────────────────────────
# Lag std widened to 5.0 to reflect within-pair variance (~3ms) plus
# measurement noise. Per-track version ID is LOW CONFIDENCE.

VERSION_PROFILES = {
    "chirp-v2": {
        "lag_mean": 15.7, "lag_std": 5.0,
        "lufs_mean": -11.6, "lufs_std": 1.7,
        "crest_mean": 14.0, "crest_std": 1.5,
        "hnr_mean": 22.0, "hnr_std": 10.0,
    },
    "chirp-v3": {
        "lag_mean": 17.6, "lag_std": 5.0,
        "lufs_mean": -17.7, "lufs_std": 1.3,
        "crest_mean": 14.0, "crest_std": 1.5,
        "hnr_mean": 28.0, "hnr_std": 15.0,
    },
    "chirp-v3.5": {
        "lag_mean": 19.0, "lag_std": 5.0,
        "lufs_mean": -15.7, "lufs_std": 0.7,
        "crest_mean": 12.0, "crest_std": 1.0,
        "hnr_mean": 20.0, "hnr_std": 10.0,
    },
    "v4.5-auk": {
        "lag_mean": 15.3, "lag_std": 5.0,
        "lufs_mean": -15.5, "lufs_std": 2.0,
        "crest_mean": 13.0, "crest_std": 1.5,
        "hnr_mean": 31.0, "hnr_std": 15.0,
    },
}


@dataclass
class SunoVersionResult:
    predicted_version: str
    confidence: float
    per_version_scores: dict
    lag_ms: Optional[float]
    lufs: Optional[float]
    crest_factor_db: Optional[float]
    hnr_ratio: Optional[float]
    notes: str


def _gaussian_log_likelihood(value: float, mean: float, std: float) -> float:
    if std < 1e-12:
        std = 1e-12
    return -0.5 * ((value - mean) / std) ** 2 - np.log(std * np.sqrt(2 * np.pi))


def classify_suno_version(path: str, sr: int = 22050, clip_sec: int = 30) -> SunoVersionResult:
    y_mono = _load_mono(path, sr, clip_sec)

    lag_ms, lag_val = compute_autocorrelation_peak(y_mono, sr)
    lufs = _compute_lufs(y_mono, sr)
    crest_db = _compute_crest_factor(y_mono)
    hnr = _compute_hnr(y_mono, sr)

    scores = {}
    for ver_name, profile in VERSION_PROFILES.items():
        log_likelihood = 0.0

        # Lag — primary discriminator, but with wide std
        if lag_ms is not None:
            log_likelihood += 2.0 * _gaussian_log_likelihood(
                lag_ms, profile["lag_mean"], profile["lag_std"])

        # LUFS — strong (v2 is 6dB louder than v3)
        if not np.isnan(lufs):
            log_likelihood += _gaussian_log_likelihood(lufs, profile["lufs_mean"], profile["lufs_std"])

        # Crest factor (v3.5 more compressed)
        if not np.isnan(crest_db):
            log_likelihood += _gaussian_log_likelihood(crest_db, profile["crest_mean"], profile["crest_std"])

        # HNR — high variance, lower weight
        if hnr is not None and not np.isnan(hnr):
            log_likelihood += 0.3 * _gaussian_log_likelihood(hnr, profile["hnr_mean"], profile["hnr_std"])

        scores[ver_name] = log_likelihood

    # Softmax
    log_vals = np.array(list(scores.values()))
    log_vals -= np.max(log_vals)
    exp_vals = np.exp(log_vals)
    probs = exp_vals / (np.sum(exp_vals) + 1e-12)

    version_names = list(scores.keys())
    version_probs = dict(zip(version_names, probs))

    predicted_version = max(version_probs, key=version_probs.get)
    confidence = version_probs[predicted_version]

    notes_parts = []
    if lag_ms is not None:
        notes_parts.append(f"vocoder lag at {lag_ms:.1f}ms")
    if not np.isnan(lufs):
        notes_parts.append(f"LUFS={lufs:.1f}")
    if hnr is not None and not np.isnan(hnr):
        notes_parts.append(f"HNR={hnr:.1f}")

    if confidence < 0.4:
        notes_parts.append("LOW CONFIDENCE — version ambiguous (lag is population-level trend)")

    return SunoVersionResult(
        predicted_version=predicted_version,
        confidence=round(confidence, 4),
        per_version_scores={k: round(v, 4) for k, v in version_probs.items()},
        lag_ms=round(lag_ms, 2) if lag_ms is not None else None,
        lufs=round(lufs, 2) if not np.isnan(lufs) else None,
        crest_factor_db=round(crest_db, 2) if not np.isnan(crest_db) else None,
        hnr_ratio=round(hnr, 2) if hnr is not None and not np.isnan(hnr) else None,
        notes="; ".join(notes_parts),
    )
