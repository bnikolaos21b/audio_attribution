#!/usr/bin/env python3
"""
General-Purpose Audio Attribution System
==========================================
5-stage pipeline that handles any track-pair type and reports its own limits.

Design principle: a system that admits uncertainty is more valuable than one
that confidently gives wrong answers.

Calibration status by pair type
────────────────────────────────
  real/real       MIPPIA min-max  — validated on 39 pos + 117 neg pairs ✅
  AI/AI same-eng  SONICS GMM      — validated on 40 pos + 40 neg pairs  ✅
  real/AI         Lyrics Jaccard  — validated on 128 pos + 192 neg pairs ✅
  cross-engine    No derivative data — flagged as unvalidated              ⚠️
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import os
import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import librosa

# ── Module root ────────────────────────────────────────────────────────────────
_SRC_DIR     = Path(__file__).parent          # src/
_PROJECT_DIR = _SRC_DIR.parent               # project root
sys.path.insert(0, str(_PROJECT_DIR))


# ══════════════════════════════════════════════════════════════════════════════
# Calibration constants  (all values come from result files in results/)
# ══════════════════════════════════════════════════════════════════════════════

# SONICS GMM  (same-engine AI/AI) ─ fitted on 40 pos + 40 neg SONICS pairs
SONICS_GMM = {
    "mu_pos": 0.935767, "sigma_pos": 0.018309,
    "mu_neg": 0.924318, "sigma_neg": 0.017772,
    "pi_pos": 0.5,
    "source": "SONICS GMM (40 pos + 40 neg, chirp-v3.5 + udio-120s)",
    "validated": True,
}

# MIPPIA min-max  (real/real compositional plagiarism)
MIPPIA_MINMAX = {
    "lo": 0.88, "hi": 0.99,
    "source": "MIPPIA min-max (real/real, compositional plagiarism)",
    "validated": True,
}

# Cross-engine: only unrelated pairs exist — no positive calibration
CROSS_ENGINE_CAL = {
    "unrelated_mean": 0.924, "unrelated_std": 0.019, "n_unrelated": 21,
    "source": "Cross-engine SONICS (21 unrelated pairs, 0 derivative pairs)",
    "validated": False,
    "warning": (
        "No cross-engine derivative calibration exists. "
        "System cannot distinguish same-prompt cross-engine from unrelated."
    ),
}

# Same-prompt sibling confounder (from mert_geometry_investigation)
SIBLING_DATA = {
    "mean": 0.94096, "std": 0.02667, "n": 13,
    "note": (
        "Same-prompt siblings score ~0.941 ± 0.027. "
        "System cannot distinguish siblings from derivatives."
    ),
}

# Lyrics calibration for real/AI  (Jaccard AUC = 0.8856)
LYRICS_CALIBRATION = {
    "jaccard": {
        "mu_pos": 0.2767, "sigma_pos": 0.1179,
        "mu_neg": 0.1288, "sigma_neg": 0.0402,
        "roc_auc": 0.8856,
        "source": "Lyrics Jaccard (128 pos + 192 neg real→AI pairs)",
    },
    "sentence_transformer": {
        "mu_pos": 0.6158, "sigma_pos": 0.1487,
        "mu_neg": 0.3918, "sigma_neg": 0.118,
        "roc_auc": 0.8749,
        "source": "Sentence-Transformer cosine (128 pos + 192 neg real→AI pairs)",
    },
    "mert_note": (
        "MERT Top-K for real→AI: positive_mean=0.871, negative_mean=0.885, "
        "gap=-0.014, AUC=0.5725. Not used as primary signal for this pair type."
    ),
}

# Acoustic profiles for Stage-1 track classification
TRACK_PROFILES = {
    "real": {
        "hnr_mean":    16.44, "hnr_std":    5.2,
        "lufs_mean":  -16.5,  "lufs_std":   2.0,
        "autocorr_mean": 0.0, "autocorr_std": 0.1,
    },
    "suno": {
        "hnr_mean":    31.5,  "hnr_std":   15.0,
        "lufs_mean":  -15.5,  "lufs_std":   1.5,
        "autocorr_mean": 0.15, "autocorr_std": 0.10,
    },
    "udio": {
        "hnr_mean":    14.0,  "hnr_std":    8.0,
        "lufs_mean":  -10.1,  "lufs_std":   2.5,
        "autocorr_mean": 0.12, "autocorr_std": 0.08,
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 — Track classifier
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrackResult:
    predicted_type: str       # "real" | "suno" | "udio" | "unknown"
    confidence: float
    hnr: Optional[float]
    lufs: Optional[float]
    autocorr_peak: Optional[float]
    autocorr_lag_ms: Optional[float]
    log_likelihoods: dict


def _compute_hnr(y: np.ndarray, sr: int = 22050, margin: float = 3.0) -> float:
    y_harm, y_perc = librosa.effects.hpss(y, margin=margin)
    rms_h = librosa.feature.rms(y=y_harm, frame_length=2048, hop_length=512)[0]
    rms_p = librosa.feature.rms(y=y_perc, frame_length=2048, hop_length=512)[0]
    voiced = rms_h > np.median(rms_h)
    if not voiced.any():
        return float("nan")
    return float(np.mean(rms_h[voiced] / (rms_p[voiced] + 1e-12)))


def _compute_lufs(y: np.ndarray, sr: int) -> float:
    try:
        import pyloudnorm as pyln
        meter = pyln.Meter(sr)
        return float(meter.integrated_loudness(y.astype(np.float64)))
    except Exception:
        return float("nan")


def _compute_autocorr_peak(
    y: np.ndarray, sr: int, lo_ms: float = 10.0, hi_ms: float = 25.0
):
    max_lag = int(30 * sr / 1000)
    y_z = y - np.mean(y)
    n = len(y_z)
    ac_full = np.real(np.fft.irfft(np.abs(np.fft.rfft(y_z, n=2 * n)) ** 2))
    ac = ac_full[: max_lag + 1] / (ac_full[0] + 1e-12)
    lags_ms = np.arange(max_lag + 1) / sr * 1000
    mask = (lags_ms >= lo_ms) & (lags_ms <= hi_ms)
    if not mask.any():
        return None, None
    idx = np.argmax(ac[mask])
    return float(lags_ms[mask][idx]), float(ac[mask][idx])


def classify_track(path: str, sr: int = 22050, clip_sec: int = 30) -> TrackResult:
    """
    Classify one track as real, suno, or udio using acoustic fingerprints.

    Returns TrackResult with predicted_type="unknown" when path is a pre-extracted
    .npy feature file (audio is unavailable for fingerprinting).
    """
    p = Path(path)
    if p.suffix.lower() == ".npy":
        return TrackResult(
            predicted_type="unknown", confidence=0.0,
            hnr=None, lufs=None, autocorr_peak=None, autocorr_lag_ms=None,
            log_likelihoods={"real": 0.0, "suno": 0.0, "udio": 0.0},
        )

    total  = librosa.get_duration(path=str(path))
    offset = max(0.0, (total - clip_sec) / 2.0)
    y, _   = librosa.load(str(path), sr=sr, mono=True, offset=offset, duration=clip_sec)

    hnr             = _compute_hnr(y, sr)
    lufs            = _compute_lufs(y, sr)
    lag_ms, peak_v  = _compute_autocorr_peak(y, sr)

    log_likes = {}
    for name, prof in TRACK_PROFILES.items():
        ll = 0.0
        if hnr is not None and not np.isnan(hnr):
            ll += norm.logpdf(hnr,    prof["hnr_mean"],      prof["hnr_std"])
        if lufs is not None and not np.isnan(lufs):
            ll += norm.logpdf(lufs,   prof["lufs_mean"],     prof["lufs_std"])
        if peak_v is not None:
            ll += norm.logpdf(peak_v, prof["autocorr_mean"], prof["autocorr_std"])
        log_likes[name] = ll

    vals = np.array(list(log_likes.values()))
    vals -= np.max(vals)
    probs = np.exp(vals) / (np.sum(np.exp(vals)) + 1e-12)
    types = list(log_likes.keys())
    predicted  = types[int(np.argmax(probs))]
    confidence = float(np.max(probs))

    return TrackResult(
        predicted_type=predicted,
        confidence=round(confidence, 4),
        hnr=round(hnr, 2) if hnr is not None and not np.isnan(hnr) else None,
        lufs=round(lufs, 2) if lufs is not None and not np.isnan(lufs) else None,
        autocorr_peak=round(peak_v, 5) if peak_v is not None else None,
        autocorr_lag_ms=round(lag_ms, 2) if lag_ms is not None else None,
        log_likelihoods={k: round(v, 2) for k, v in log_likes.items()},
    )


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 — Pair-type router
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PairTypeInfo:
    pair_type: str
    calibration_name: str
    calibration_params: dict
    warning: Optional[str] = None
    reliability: str = "MODERATE"   # HIGH | MODERATE | LOW | INSUFFICIENT


def classify_pair(track_a: TrackResult, track_b: TrackResult) -> PairTypeInfo:
    """Route a track pair to the appropriate calibration based on track types."""
    ta = track_a.predicted_type
    tb = track_b.predicted_type

    if ta == "unknown" or tb == "unknown":
        return PairTypeInfo(
            pair_type="unknown/unclassified",
            calibration_name="SONICS GMM (fallback — track types unknown)",
            calibration_params=SONICS_GMM,
            warning=(
                "Track types could not be determined "
                "(pre-extracted features lack audio). "
                "Falling back to SONICS GMM calibration."
            ),
            reliability="MODERATE",
        )

    if ta == "real" and tb == "real":
        return PairTypeInfo(
            pair_type="real/real",
            calibration_name="MIPPIA min-max",
            calibration_params=MIPPIA_MINMAX,
            reliability="HIGH",
        )

    if ta in ("suno", "udio") and tb in ("suno", "udio"):
        if ta == tb:
            return PairTypeInfo(
                pair_type="same-engine-ai",
                calibration_name="SONICS GMM",
                calibration_params=SONICS_GMM,
                warning=(
                    f"Both tracks classified as {ta}. Same-prompt siblings "
                    f"score {SIBLING_DATA['mean']:.3f} ± {SIBLING_DATA['std']:.3f} "
                    f"— indistinguishable from derivatives in MERT space."
                ),
                reliability="HIGH",
            )
        return PairTypeInfo(
            pair_type="cross-engine-ai",
            calibration_name="Cross-engine (unrelated-only reference)",
            calibration_params=CROSS_ENGINE_CAL,
            warning=CROSS_ENGINE_CAL["warning"],
            reliability="LOW",
        )

    # One real, one AI → lyrics-based route
    return PairTypeInfo(
        pair_type="real/ai",
        calibration_name="Lyrics Jaccard (AUC=0.8856, 128 pos + 192 neg pairs)",
        calibration_params=LYRICS_CALIBRATION["jaccard"],
        reliability="HIGH",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3 — MERT Top-K similarity  (k = 10)
# ══════════════════════════════════════════════════════════════════════════════

def _get_mert_segments(path: str, cache_dir: str, device: str = "cpu") -> np.ndarray:
    """Return (N_segments × 768) MERT embeddings, loading from cache when available."""
    import torch
    from transformers import AutoProcessor, AutoModel

    p          = Path(path)
    cache_dir  = Path(cache_dir)
    cache_path = cache_dir / f"{p.stem}_mert.npy"

    if p.suffix.lower() == ".npy":
        if cache_path.exists():
            return np.load(cache_path)
        arr = np.load(str(p), mmap_mode="r")
        return arr[:, :768].astype(np.float32)

    if cache_path.exists():
        return np.load(cache_path)

    # ── Extract from raw audio ────────────────────────────────────────────────
    processor = AutoProcessor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
    model = (
        AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
        .to(device)
        .eval()
    )

    sr  = 24000
    win = int(10 * sr)
    hop = int(5  * sr)
    min_len = int(3 * sr)

    y, _ = librosa.load(str(path), sr=sr, mono=True)
    segs, start = [], 0
    while start < len(y):
        chunk = y[start : start + win]
        if len(chunk) < min_len:
            break
        if len(chunk) < win:
            chunk = np.pad(chunk, (0, win - len(chunk)))
        segs.append(chunk.astype(np.float32))
        start += hop

    import torch
    seg_embs = []
    with torch.no_grad():
        for seg in segs:
            inputs = processor(seg, sampling_rate=sr, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            emb = model(**inputs).last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
            seg_embs.append(emb.astype(np.float32))

    result = np.array(seg_embs)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(cache_path), result)
    return result


def mert_topk_similarity(
    seg_a: np.ndarray, seg_b: np.ndarray, k: int = 10
) -> tuple[float, list]:
    """
    Compute Top-K cosine similarity between two MERT segment arrays.

    For every segment in A, finds its most similar segment in B (regardless
    of position). Returns the mean of the k highest per-segment maxima.

    IMPORTANT: Applies (cos + 1) / 2 rescaling to match the SONICS pipeline
    where GMM calibration parameters were fitted. Without this rescaling the
    GMM would be systematically mis-calibrated (Issue 4/8D fix).
    """
    a_n = seg_a / (np.linalg.norm(seg_a, axis=1, keepdims=True) + 1e-12)
    b_n = seg_b / (np.linalg.norm(seg_b, axis=1, keepdims=True) + 1e-12)
    sim = a_n @ b_n.T

    # Rescale to [0, 1] — matches the SONICS pipeline calibration space
    sim = (sim + 1.0) / 2.0

    top1  = np.max(sim, axis=1)
    topk  = np.sort(top1)[-k:]
    raw   = float(np.mean(topk))

    matches = []
    for i in range(len(seg_a)):
        j = int(np.argmax(sim[i]))
        matches.append({
            "seg_a_idx": i, "seg_b_idx": j,
            "similarity": round(float(sim[i, j]), 5),
            "seg_a_time_sec": i * 5,
            "seg_b_time_sec": j * 5,
        })
    matches.sort(key=lambda x: x["similarity"], reverse=True)
    return raw, matches[:20]


# ══════════════════════════════════════════════════════════════════════════════
# Stage 4 — Calibrated score
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CalibratedResult:
    raw_score: float
    calibrated_score: float
    calibration_source: str
    pair_type: str
    reliability: str
    warnings: list
    top_matches: list


def _gmm_calibrate(
    raw: float,
    mu_pos: float, sigma_pos: float,
    mu_neg: float, sigma_neg: float,
    pi_pos: float = 0.5,
) -> float:
    from scipy.stats import norm  # lazy import — only needed for GMM calibration
    like_pos = norm.pdf(raw, mu_pos, sigma_pos)
    like_neg = norm.pdf(raw, mu_neg, sigma_neg)
    num = pi_pos * like_pos
    den = num + (1 - pi_pos) * like_neg
    return float(num / den) if den >= 1e-12 else 0.5


def calibrate_score(
    raw: float, pair_info: PairTypeInfo, k: int = 10
) -> CalibratedResult:
    """Apply the pair-type-appropriate calibration to a raw MERT or Jaccard score."""
    params   = pair_info.calibration_params
    warnings = []

    if pair_info.pair_type == "real/real":
        lo, hi = params["lo"], params["hi"]
        calibrated = float(np.clip((raw - lo) / (hi - lo + 1e-12), 0.0, 1.0))
        source = params["source"]

    elif pair_info.pair_type == "same-engine-ai":
        calibrated = _gmm_calibrate(
            raw, params["mu_pos"], params["sigma_pos"],
            params["mu_neg"], params["sigma_neg"], params.get("pi_pos", 0.5)
        )
        source = params["source"]
        if raw > SIBLING_DATA["mean"] - 2 * SIBLING_DATA["std"]:
            warnings.append(
                f"Raw score {raw:.4f} falls within the same-prompt sibling range "
                f"({SIBLING_DATA['mean']:.3f} ± {SIBLING_DATA['std']:.3f}). "
                "Derivative and independent generations are indistinguishable here."
            )

    elif pair_info.pair_type == "cross-engine-ai":
        z = (raw - params["unrelated_mean"]) / (params["unrelated_std"] + 1e-12)
        calibrated = float(1.0 / (1.0 + np.exp(-z)))
        source = params["source"]
        warnings.append(params["warning"])
        warnings.append(
            f"Cross-engine score is speculative. "
            f"Raw {raw:.4f} vs unrelated mean {params['unrelated_mean']:.4f} (z={z:.2f})."
        )

    elif pair_info.pair_type == "real/ai":
        # Primary signal: Jaccard similarity (passed in as `raw`)
        calibrated = _gmm_calibrate(
            raw, params["mu_pos"], params["sigma_pos"],
            params["mu_neg"], params["sigma_neg"]
        )
        source = params.get("source", "Lyrics Jaccard calibration")

    elif pair_info.pair_type == "unknown/unclassified":
        calibrated = _gmm_calibrate(
            raw, params["mu_pos"], params["sigma_pos"],
            params["mu_neg"], params["sigma_neg"], params.get("pi_pos", 0.5)
        )
        source = params["source"]
        warnings.append("Track types unknown — using SONICS GMM as default.")

    else:
        calibrated = raw
        source = "No calibration available"
        warnings.append("Unrecognised pair type — raw score returned uncalibrated.")

    return CalibratedResult(
        raw_score=round(raw, 5),
        calibrated_score=round(calibrated, 4),
        calibration_source=source,
        pair_type=pair_info.pair_type,
        reliability=pair_info.reliability,
        warnings=warnings,
        top_matches=[],
    )


# ══════════════════════════════════════════════════════════════════════════════
# Lyrics analysis  (for real/AI pairs)
# ══════════════════════════════════════════════════════════════════════════════

def compute_lyrics_similarity(path_a: str, path_b: str) -> dict:
    """
    Transcribe both tracks with Whisper-tiny and return Jaccard + sentence-
    transformer similarities. This is the primary signal for real/AI pairs.
    """
    import whisper
    from sentence_transformers import SentenceTransformer

    wmodel = whisper.load_model("tiny")

    def transcribe(p):
        return wmodel.transcribe(str(p), language="en")["text"].strip()

    text_a = transcribe(path_a)
    text_b = transcribe(path_b)

    wa = set(text_a.lower().split())
    wb = set(text_b.lower().split())
    jaccard = len(wa & wb) / len(wa | wb) if (wa and wb) else 0.0

    smodel = SentenceTransformer("all-MiniLM-L6-v2")
    ea = smodel.encode([text_a])[0]
    eb = smodel.encode([text_b])[0]
    sent_sim = float(
        np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb) + 1e-12)
    )

    return {
        "text_a": text_a[:300],
        "text_b": text_b[:300],
        "jaccard": round(jaccard, 4),
        "sentence_similarity": round(sent_sim, 4),
        "combined": round(0.7 * jaccard + 0.3 * sent_sim, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Stage 5 — Explanation layer
# ══════════════════════════════════════════════════════════════════════════════

def _verdict(score: float, reliability: str) -> str:
    if reliability == "INSUFFICIENT":
        return "INSUFFICIENT SIGNAL"
    if score >= 0.75:
        return "LIKELY DERIVATIVE"
    if score >= 0.55:
        return "POSSIBLY RELATED"
    if score >= 0.35:
        return "WEAK SIMILARITY"
    return "UNRELATED"


def _fmt_time(sec: int) -> str:
    m, s = divmod(sec, 60)
    return f"{m:02d}:{s:02d}"


def generate_explanation(
    track_a: TrackResult,
    track_b: TrackResult,
    cal: CalibratedResult,
    pair_info: PairTypeInfo,
    lyrics_info: Optional[dict] = None,
) -> str:
    score    = cal.calibrated_score
    verdict  = _verdict(score, cal.reliability)
    bar      = "█" * max(0, min(24, int(round(score * 24)))) + "░" * max(0, 24 - int(round(score * 24)))
    lines    = []

    lines += [
        "=" * 72,
        "  AUDIO ATTRIBUTION ANALYSIS",
        "=" * 72,
        f"  Score:       {score:.4f}  [{bar}]  {score*100:.1f}%",
        f"  Verdict:     {verdict}",
        f"  Reliability: {cal.reliability}",
        f"  Raw:         {cal.raw_score:.4f}",
        f"  Pair type:   {pair_info.pair_type}",
        f"  Calibration: {cal.calibration_source}",
        "",
    ]

    for label, tr in [("A", track_a), ("B", track_b)]:
        lines.append(f"  ── Track {label} ({tr.predicted_type}, conf={tr.confidence:.2f}) ──")
        if tr.hnr is not None:
            lines.append(
                f"    HNR: {tr.hnr}  |  LUFS: {tr.lufs}  |  "
                f"Autocorr: {tr.autocorr_peak} at {tr.autocorr_lag_ms}ms"
            )
        else:
            lines.append("    (pre-extracted features — no audio fingerprint available)")
        lines.append("")

    if cal.top_matches:
        lines.append("  ── Strongest Segment Matches ──")
        for m in cal.top_matches[:5]:
            lines.append(
                f"    Track A {_fmt_time(m['seg_a_time_sec'])} ↔ "
                f"Track B {_fmt_time(m['seg_b_time_sec'])}  sim={m['similarity']:.5f}"
            )
        lines.append("")

    if pair_info.pair_type == "real/ai" and lyrics_info:
        lines.append("  ── Lyrics-Based Attribution (primary signal) ──")
        if "error" not in lyrics_info:
            lines += [
                f"    Jaccard similarity:   {lyrics_info['jaccard']:.4f}  "
                f"(threshold μ_pos=0.277, μ_neg=0.129)",
                f"    Sentence similarity:  {lyrics_info['sentence_similarity']:.4f}",
                f"    Combined (70/30):     {lyrics_info['combined']:.4f}",
                f"",
                f"    MERT AUC (real→AI):  0.5725 — not used as primary signal",
                f"    Jaccard AUC:         0.8856 — primary signal for this pair type",
            ]
        else:
            lines.append(
                f"    Lyrics analysis failed: {lyrics_info.get('error', 'unknown')}. "
                "Falling back to MERT score."
            )
        lines.append("")

    all_warnings = ([pair_info.warning] if pair_info.warning else []) + cal.warnings
    if all_warnings:
        lines.append("  ── Warnings ──")
        for w in all_warnings:
            lines.append(f"    ⚠  {w}")
        lines.append("")

    lines.append("=" * 72)
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AttributionResult:
    score: float
    verdict: str
    raw_score: float
    pair_type: str
    reliability: str
    track_a: TrackResult
    track_b: TrackResult
    calibration_source: str
    warnings: list
    top_matches: list
    explanation: str

    def summary(self) -> str:
        return self.explanation


def attribute_tracks(
    path_a: str,
    path_b: str,
    cache_dir: str = "features",
    device: str = "cpu",
    k: int = 10,
) -> AttributionResult:
    """
    Compare two audio tracks and return an AttributionResult.

    Parameters
    ----------
    path_a, path_b : str
        Paths to audio files (.mp3, .wav, .flac) or pre-extracted .npy features.
    cache_dir : str
        Directory for MERT embedding cache.
    device : str
        "cpu" or "cuda".
    k : int
        Number of top segment similarities to average (default 10).

    Returns
    -------
    AttributionResult
        Call `.summary()` for the formatted explanation string.
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Stage 1 — classify each track
    track_a = classify_track(path_a)
    track_b = classify_track(path_b)

    # Stage 2 — route to appropriate calibration
    pair_info = classify_pair(track_a, track_b)

    # Stage 3 — MERT Top-K
    seg_a = _get_mert_segments(path_a, cache_dir, device)
    seg_b = _get_mert_segments(path_b, cache_dir, device)
    mert_raw, top_matches = mert_topk_similarity(seg_a, seg_b, k=k)

    # Stage 4a — for real/AI pairs, override raw score with Jaccard
    lyrics_info = None
    raw_score   = mert_raw
    if pair_info.pair_type == "real/ai":
        try:
            lyrics_info = compute_lyrics_similarity(path_a, path_b)
            raw_score   = lyrics_info["jaccard"]
        except Exception as e:
            lyrics_info = {"error": str(e)}
            # Fall through with MERT raw score; pair_info stays real/ai

    # Stage 4b — calibrate
    cal = calibrate_score(raw_score, pair_info, k=k)
    cal.top_matches = top_matches

    # Stage 5 — explain
    explanation = generate_explanation(track_a, track_b, cal, pair_info, lyrics_info)
    verdict     = _verdict(cal.calibrated_score, cal.reliability)

    all_warnings = ([pair_info.warning] if pair_info.warning else []) + cal.warnings

    return AttributionResult(
        score=cal.calibrated_score,
        verdict=verdict,
        raw_score=cal.raw_score,
        pair_type=pair_info.pair_type,
        reliability=cal.reliability,
        track_a=track_a,
        track_b=track_b,
        calibration_source=cal.calibration_source,
        warnings=all_warnings,
        top_matches=top_matches,
        explanation=explanation,
    )
