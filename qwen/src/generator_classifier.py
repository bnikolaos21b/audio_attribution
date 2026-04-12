"""
Stage 2: Generator Identification
==================================
Classifies an audio track as: Real, Suno, or Udio.
Optionally identifies Suno model version (chirp-v2/v3/v3.5).

Uses features established from 26 folk Suno songs + 160 SONICS tracks:
  - LUFS (integrated loudness) — Udio 5.4dB louder than Suno (d=3.01)
  - HNR (harmonic/percussive RMS, margin=3.0, voiced frames) — Suno cleaner than Udio (d=1.66)
  - Crest factor — v3.5 more compressed than v2/v3 (d=1.31)
  - Vocoder autocorrelation lag — population-level trend across Suno versions
    (median 15.7→17.6→19.0ms for v2→v3→v3.5); intra-version spread ~3ms std,
    so this cannot identify version from a single track.

All parameters derived from empirical measurements.
"""

import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
from typing import Optional

import numpy as np
import librosa
import pyloudnorm as pyln

from .ai_detector import _load_mono, _load_stereo, compute_autocorrelation_peak


# ── HNR computation (same formula as ai_detector, for generator profiling) ─────

def _compute_hnr(y: np.ndarray, sr: int = 22050, margin: float = 3.0) -> float:
    y_harm, y_perc = librosa.effects.hpss(y, margin=margin)
    rms_harm = librosa.feature.rms(y=y_harm, frame_length=2048, hop_length=512)[0]
    rms_perc = librosa.feature.rms(y=y_perc, frame_length=2048, hop_length=512)[0]
    voiced = rms_harm > np.median(rms_harm)
    if not voiced.any():
        return float("nan")
    return float(np.mean(rms_harm[voiced] / (rms_perc[voiced] + 1e-12)))


# ── Generator profiles (empirically measured) ─────────────────────────────────
# Source: 26 folk Suno + 72 SONICS Suno + 88 SONICS Udio + 5 real folk
#
# HNR values use the HPSS margin=3.0, voiced-frame formula.
# Real vs Suno on 26 folk + 5 real: mean 16.44 vs 31.54 (d=1.34, p=0.003)
# Suno vs Udio on 160 SONICS: d=+1.66

GENERATOR_PROFILES = {
    "real": {
        "lufs_mean": -18.5, "lufs_std": 3.0,
        "hnr_mean": 16.0,   "hnr_std": 6.0,      # Real: ~10–24 on folk MP3s
        "crest_mean": 16.0, "crest_std": 2.0,
        "autocorr_lag_mean": None,  # No consistent vocoder peak
    },
    "suno_chirp_v2": {
        "lufs_mean": -11.6, "lufs_std": 1.7,
        "hnr_mean": 22.0,   "hnr_std": 10.0,
        "crest_mean": 14.0, "crest_std": 1.5,
        "autocorr_lag_mean": 15.7,
    },
    "suno_chirp_v3": {
        "lufs_mean": -17.7, "lufs_std": 1.3,
        "hnr_mean": 28.0,   "hnr_std": 15.0,
        "crest_mean": 14.0, "crest_std": 1.5,
        "autocorr_lag_mean": 17.6,
    },
    "suno_chirp_v35": {
        "lufs_mean": -15.7, "lufs_std": 0.7,
        "hnr_mean": 20.0,   "hnr_std": 10.0,
        "crest_mean": 12.0, "crest_std": 1.0,
        "autocorr_lag_mean": 19.0,
    },
    "suno_v45_auk": {
        # Estimated from 26 folk songs. v4.5 uses chirp-auk internally.
        "lufs_mean": -15.5, "lufs_std": 2.0,
        "hnr_mean": 31.0,   "hnr_std": 15.0,     # 26 folk mean ~31.5
        "crest_mean": 13.0, "crest_std": 1.5,
        "autocorr_lag_mean": 15.3,  # Measured from 26 folk songs
    },
    "udio": {
        "lufs_mean": -10.1, "lufs_std": 2.5,
        "hnr_mean": 14.0,   "hnr_std": 8.0,      # Udio has more diffusion noise
        "crest_mean": 13.0, "crest_std": 1.5,
        "autocorr_lag_mean": 17.0,
    },
}

PRIMARY_GENERATORS = ["real", "suno", "udio"]


@dataclass
class GeneratorResult:
    predicted_generator: str
    predicted_version: Optional[str]
    confidence: float
    lufs: Optional[float]
    crest_factor_db: Optional[float]
    hnr_ratio: Optional[float]       # Linear ratio (not dB)
    autocorr_lag_ms: Optional[float]
    per_generator_scores: dict


def _gaussian_log_likelihood(value: float, mean: float, std: float) -> float:
    if std < 1e-12:
        std = 1e-12
    return -0.5 * ((value - mean) / std) ** 2 - np.log(std * np.sqrt(2 * np.pi))


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


def classify_generator(path: str, sr: int = 22050, clip_sec: int = 30,
                       detect_version: bool = True) -> GeneratorResult:
    y_mono = _load_mono(path, sr, clip_sec)
    y_stereo = _load_stereo(path, sr, clip_sec)
    y_mono_mix = np.mean(y_stereo, axis=0) if y_stereo.ndim > 1 else y_mono

    lufs = _compute_lufs(y_mono_mix, sr)
    crest_db = _compute_crest_factor(y_mono_mix)
    hnr = _compute_hnr(y_mono, sr)
    peak_lag, peak_val = compute_autocorrelation_peak(y_mono, sr)

    # Score each generator
    scores = {}
    for gen_name, profile in GENERATOR_PROFILES.items():
        log_likelihood = 0.0

        # LUFS (strong discriminator, d=3.01 Suno vs Udio)
        if not np.isnan(lufs):
            log_likelihood += _gaussian_log_likelihood(lufs, profile["lufs_mean"], profile["lufs_std"])

        # HNR (d=1.66 Suno vs Udio)
        if hnr is not None and not np.isnan(hnr):
            log_likelihood += _gaussian_log_likelihood(hnr, profile["hnr_mean"], profile["hnr_std"])

        # Crest factor (d=1.31 v3 vs v3.5)
        if not np.isnan(crest_db):
            log_likelihood += _gaussian_log_likelihood(crest_db, profile["crest_mean"], profile["crest_std"])

        # Autocorrelation lag (version discriminator)
        # Population-level trend only — not reliable per-track.
        if peak_lag is not None and profile["autocorr_lag_mean"] is not None:
            if gen_name == "real":
                log_likelihood -= 3.0  # Penalty for real if vocoder peak exists
            else:
                lag_std = 5.0  # Wide — within-pair variance can be ~3ms
                log_likelihood += _gaussian_log_likelihood(peak_lag, profile["autocorr_lag_mean"], lag_std)
        elif peak_lag is not None and gen_name == "real":
            log_likelihood -= 2.0

        scores[gen_name] = log_likelihood

    # Softmax
    log_vals = np.array(list(scores.values()))
    log_vals -= np.max(log_vals)
    exp_vals = np.exp(log_vals)
    probs = exp_vals / (np.sum(exp_vals) + 1e-12)

    gen_names = list(scores.keys())
    gen_probs = dict(zip(gen_names, probs))

    # Aggregate to primary generator level
    primary_probs = {"real": 0.0, "suno": 0.0, "udio": 0.0}
    primary_probs["real"] = gen_probs.get("real", 0.0)
    primary_probs["udio"] = gen_probs.get("udio", 0.0)
    suno_keys = [k for k in gen_names if k.startswith("suno")]
    primary_probs["suno"] = sum(gen_probs.get(k, 0.0) for k in suno_keys)

    total_primary = sum(primary_probs.values())
    if total_primary > 1e-12:
        primary_probs = {k: v / total_primary for k, v in primary_probs.items()}

    predicted_primary = max(primary_probs, key=primary_probs.get)
    primary_confidence = primary_probs[predicted_primary]

    # Version identification
    predicted_version = None
    if predicted_primary == "suno" and detect_version:
        suno_probs = {k: v for k, v in gen_probs.items() if k.startswith("suno")}
        if suno_probs:
            best_version = max(suno_probs, key=suno_probs.get)
            version_map = {
                "suno_chirp_v2": "chirp-v2",
                "suno_chirp_v3": "chirp-v3",
                "suno_chirp_v35": "chirp-v3.5",
                "suno_v45_auk": "v4.5-auk",
            }
            predicted_version = version_map.get(best_version, best_version)

    return GeneratorResult(
        predicted_generator=predicted_primary,
        predicted_version=predicted_version,
        confidence=round(primary_confidence, 4),
        lufs=round(lufs, 2) if not np.isnan(lufs) else None,
        crest_factor_db=round(crest_db, 2) if not np.isnan(crest_db) else None,
        hnr_ratio=round(hnr, 2) if hnr is not None and not np.isnan(hnr) else None,
        autocorr_lag_ms=round(peak_lag, 2) if peak_lag is not None else None,
        per_generator_scores={k: round(v, 4) for k, v in gen_probs.items()},
    )
