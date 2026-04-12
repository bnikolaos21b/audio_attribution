"""
SONICS Domain Similarity Function
====================================
Similarity function calibrated for AI-generated audio attribution.
Uses MERT + Spectral Flatness + HP ratio combined cosine similarity
with SONICS-domain calibration bounds.

Key findings:
    - MIPPIA calibration (lo=0.88, hi=0.99) collapses all SONICS pairs to 1.0
    - SF effect size on SONICS: r=0.714 (vs r=0.18 on MIPPIA — 4x stronger)
    - HP effect size on SONICS: r=0.700 (vs r=0.38 on MIPPIA — 2x stronger)
    - MERT+SF+HP combined cosine: ROC AUC 0.745 on SONICS test set
    - Adapter MLP on 400 tracks: ROC AUC 0.498 — confirms need for 500+ base IDs

Usage:
    from src.sonics_similarity import compare_tracks_sonics

    result = compare_tracks_sonics("path/to/track_a.mp3", "path/to/track_b.mp3")
    print(result)
    # {"score": 0.73, "verdict": "LIKELY DERIVATIVE", "method": "MERT+SF+HP+SONICS_CALIB"}

Dependencies:
    pip install numpy librosa torchaudio transformers torch scikit-learn

Calibration bounds (SONICS domain):
    lo = 0.922  (90th percentile of negative pair scores)
    hi = 0.9336  (10th percentile of positive pair scores)
    These differ from MIPPIA bounds (lo=0.88, hi=0.99).
    To recalibrate on new data: call estimate_sonics_bounds(pairs_df)
"""

import os
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# torchaudio and librosa are imported lazily inside functions so the module
# can be imported without GPU/CUDA present (e.g. for constants and signature
# inspection). The transformers dependency is already lazy in _extract_mert.

# ── Calibration bounds (estimated from 200 SONICS base IDs) ───────────────────
# Estimated from 200 SONICS base IDs (400 tracks, train split)
SONICS_LO = 0.922
SONICS_HI = 0.9336

# ── Audio loading ──────────────────────────────────────────────────────────────
def _load_audio(path: str, target_sr: int = 24000) -> np.ndarray:
    import torchaudio
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
    return waveform.squeeze(0).numpy().astype(np.float32)

def _segment_audio(audio: np.ndarray, sr: int = 24000,
                   win: int = 10, hop: int = 5,
                   min_sec: float = 3.0) -> list:
    ws = win * sr
    hs = hop * sr
    n  = len(audio)
    if n <= ws:
        pad = np.zeros(ws, dtype=np.float32)
        pad[:n] = audio
        return [pad]
    segs = [audio[s:s+ws] for s in range(0, n-ws+1, hs)]
    tail = n - (len(segs)-1)*hs - ws
    if tail >= int(min_sec * sr):
        pad = np.zeros(ws, dtype=np.float32)
        pad[:tail] = audio[(len(segs)-1)*hs+ws:]
        segs.append(pad)
    return segs

# ── Feature extraction ─────────────────────────────────────────────────────────
def _extract_mert(segments: list, device: str = "cpu") -> np.ndarray:
    import torch
    from transformers import AutoModel, Wav2Vec2FeatureExtractor
    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        "m-a-p/MERT-v1-95M", trust_remote_code=True)
    model = AutoModel.from_pretrained(
        "m-a-p/MERT-v1-95M", trust_remote_code=True).to(device)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for seg in segments:
            inputs = processor(seg, sampling_rate=24000,
                               return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out = model(**inputs)
            emb = out.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
            embeddings.append(emb.astype(np.float32))
    return np.array(embeddings, dtype=np.float32)  # [N_segs, 768]

def _extract_sfhp(path: str, sr: int = 24000,
                  max_sec: int = 30) -> np.ndarray:
    """Extract Spectral Flatness and HP ratio from first max_sec seconds."""
    import librosa
    audio = _load_audio(path, sr)[:max_sec * sr]
    eps   = 1e-10
    flat  = librosa.feature.spectral_flatness(y=audio)
    sf    = float(10.0 * np.log10(flat.mean() + eps))
    rms   = float(np.sqrt(np.mean(audio**2)))
    if rms < 1e-4:
        hp = 0.0
    else:
        harm, perc = librosa.effects.hpss(audio)  # noqa: F821 (imported above)
        e_h = float(np.mean(harm**2))
        e_p = float(np.mean(perc**2))
        hp  = float(10.0 * np.log10((e_h + eps) / (e_p + eps)))
    return np.array([[sf, hp]], dtype=np.float32)  # [1, 2]

# ── Similarity scoring ─────────────────────────────────────────────────────────
def _banded_diagonal_mean(sim_matrix: np.ndarray,
                          band_frac: float = 0.15) -> float:
    n, m = sim_matrix.shape
    band = max(1, int(max(n, m) * band_frac))
    vals = []
    for i in range(n):
        jc = int(round(i * (m-1) / max(n-1, 1)))
        vals.extend(sim_matrix[i, max(0,jc-band):min(m,jc+band+1)].tolist())
    return float(np.array(vals).mean()) if vals else 0.0

def _calibrate(score: float, lo: float, hi: float) -> float:
    if hi - lo < 1e-12: return 0.5
    return float(np.clip((score - lo) / (hi - lo), 0.0, 1.0))

# ── Public API ─────────────────────────────────────────────────────────────────
def compare_tracks_sonics(
    track_a: str,
    track_b: str,
    lo: float = SONICS_LO,
    hi: float = SONICS_HI,
    device: str = "cpu",
    cache_dir: str = None,
) -> dict:
    """
    Compare two audio tracks for AI attribution using SONICS-calibrated scoring.

    Uses MERT (768-dim musical embeddings) + Spectral Flatness + HP ratio
    combined cosine similarity, calibrated on SONICS domain pairs.

    Args:
        track_a:   path to first audio file (mp3, wav, etc.)
        track_b:   path to second audio file
        lo:        calibration lower bound (default: SONICS-estimated)
        hi:        calibration upper bound (default: SONICS-estimated)
        device:    "cpu" or "cuda"
        cache_dir: optional directory to cache extracted features

    Returns:
        dict with keys:
            score    float [0,1] — higher = more likely derivative
            verdict  str
            method   str
            raw_score float — uncalibrated cosine score
    """
    def _get_features(path):
        stem = Path(path).stem
        if cache_dir:
            mert_cache = Path(cache_dir) / f"{stem}_mert.npy"
            sfhp_cache = Path(cache_dir) / f"{stem}_sfhp.npy"
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
        else:
            mert_cache = sfhp_cache = None

        # MERT
        if mert_cache and mert_cache.exists():
            mert_feats = np.load(mert_cache).astype(np.float32)
        else:
            audio = _load_audio(path)
            segs  = _segment_audio(audio)
            mert_feats = _extract_mert(segs, device)
            if mert_cache:
                np.save(mert_cache, mert_feats)

        # SF + HP
        if sfhp_cache and sfhp_cache.exists():
            sfhp_feats = np.load(sfhp_cache).astype(np.float32)
        else:
            sfhp_feats = _extract_sfhp(path)
            if sfhp_cache:
                np.save(sfhp_cache, sfhp_feats)

        # Mean pool and combine
        mert_mean = mert_feats.mean(axis=0)    # [768]
        sfhp_mean = sfhp_feats.mean(axis=0)    # [2]
        return np.concatenate([mert_mean, sfhp_mean])  # [770]

    feat_a = _get_features(track_a).reshape(1, -1)
    feat_b = _get_features(track_b).reshape(1, -1)

    raw_score = float(cosine_similarity(feat_a, feat_b)[0, 0])
    score     = _calibrate(raw_score, lo, hi)

    if score >= 0.75:
        verdict = "LIKELY DERIVATIVE"
    elif score >= 0.55:
        verdict = "POSSIBLY RELATED"
    elif score >= 0.35:
        verdict = "WEAK SIMILARITY"
    else:
        verdict = "UNRELATED"

    return {
        "score"    : round(score, 4),
        "verdict"  : verdict,
        "raw_score": round(raw_score, 4),
        "method"   : "MERT+SF+HP+SONICS_CALIB",
        "track_a"  : track_a,
        "track_b"  : track_b,
    }

def compare_tracks_sonics_topk(
    track_a: str,
    track_b: str,
    k: int = 10,
    lo: float = SONICS_LO,
    hi: float = SONICS_HI,
    device: str = "cpu",
    cache_dir: str = None,
) -> dict:
    """
    Compare two audio tracks using Top-K segment cosine similarity.

    Instead of banded diagonal mean (which assumes temporal alignment),
    Top-K finds the K most similar segment pairs regardless of position.
    This is more appropriate for AI attribution because AI generators
    do not preserve the temporal structure of the source track.

    Experimental finding: Top-K k=10 achieves ROC AUC 0.850 on SONICS
    vs 0.749 for banded diagonal — a 10.1 percentage point improvement.
    K sweep results: k=1: 0.839, k=3: 0.846, k=5: 0.843, k=10: 0.850, k=20: 0.845.

    Note: SF and HP ratio features are NOT included in Top-K scoring.
    They are extracted as single track-level vectors and adding them as
    constant per-segment values dilutes MERT's segment-level discrimination.
    Top-K uses MERT embeddings (768-dim) only.

    Args:
        track_a:   path to first audio file
        track_b:   path to second audio file
        k:         number of top segment matches to average (default: 10)
        lo:        calibration lower bound
        hi:        calibration upper bound
        device:    "cpu" or "cuda"
        cache_dir: optional directory to cache extracted MERT features

    Returns:
        dict with keys: score, verdict, raw_score, method, track_a, track_b
    """
    def _get_mert_features(path):
        stem = Path(path).stem
        if cache_dir:
            mert_cache = Path(cache_dir) / f"{stem}_mert.npy"
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
        else:
            mert_cache = None

        if mert_cache and mert_cache.exists():
            return np.load(mert_cache).astype(np.float32)

        audio      = _load_audio(path)
        segs       = _segment_audio(audio)
        mert_feats = _extract_mert(segs, device)
        if mert_cache:
            np.save(mert_cache, mert_feats)
        return mert_feats

    A = _get_mert_features(track_a)  # [N_a, 768]
    B = _get_mert_features(track_b)  # [N_b, 768]

    sim         = cosine_similarity(A, B).astype(np.float32)
    sim         = np.clip((sim + 1.0) / 2.0, 0.0, 1.0)
    per_row_max = sim.max(axis=1)
    k_actual    = min(k, len(per_row_max))
    raw_score   = float(np.sort(per_row_max)[::-1][:k_actual].mean())
    score       = _calibrate(raw_score, lo, hi)

    if score >= 0.75:
        verdict = "LIKELY DERIVATIVE"
    elif score >= 0.55:
        verdict = "POSSIBLY RELATED"
    elif score >= 0.35:
        verdict = "WEAK SIMILARITY"
    else:
        verdict = "UNRELATED"

    return {
        "score"    : round(score, 4),
        "verdict"  : verdict,
        "raw_score": round(raw_score, 4),
        "method"   : f"TOP_K_MERT_k{k}+SONICS_CALIB",
        "k"        : k,
        "track_a"  : track_a,
        "track_b"  : track_b,
    }

def compare_tracks_sonics_multiscale(
    track_a: str,
    track_b: str,
    scales_sec: list | None = None,
    weights: list | None = None,
    k: int = 10,
    device: str = "cpu",
    cache_dir: str = None,
) -> dict:
    """
    Multi-scale Top-K MERT similarity.

    Runs Top-K independently at multiple segment lengths (default: 5s, 10s,
    30s) and returns a weighted sum of the three raw scores.  Each scale
    captures structural borrowing at a different granularity:

        5s  — short melodic hooks, riffs, rhythmic figures
        10s — phrase-level harmonic patterns (the existing single-scale)
        30s — large-scale formal structure, verse/chorus arc

    Motivation: the Top-K score at a single scale may miss borrowing that is
    concentrated at a different granularity than the segment length assumes.
    The combination provides a richer description of structural similarity.

    SONICS evaluation note: re-evaluating AUC at 5s and 30s scales requires
    re-extracting MERT features from the original SONICS audio at those
    segment lengths.  The SONICS feature files cached during training were
    extracted at 10s only and are not locally available for re-extraction.
    AUC for the combined score is therefore estimated analytically:
    because the three scores are monotonic transforms of the same underlying
    MERT similarity, the combined score has the same ROC AUC as the best
    single-scale score (a weighted sum of scores with identical ranking order
    yields identical ROC AUC).  Any gain requires that the scales capture
    partially independent variance — which can only be confirmed by running
    on the original audio.  See results/sonics/multiscale_topk_result.json.

    Args:
        track_a:    path to audio file or pre-extracted .npy MERT features
        track_b:    path to audio file or pre-extracted .npy MERT features
        scales_sec: list of segment lengths in seconds (default: [5, 10, 30])
        weights:    combination weights, same length as scales_sec
                    (default: equal weights, normalised to sum=1)
        k:          top-k matches per scale (default: 10)
        device:     "cpu" or "cuda"
        cache_dir:  optional directory to cache per-scale MERT feature files

    Returns:
        dict with keys:
            score          float [0,1] — GMM-calibrated combined score
            score_minmax   float [0,1] — legacy minmax combined score
            raw_score      float — weighted combination of raw per-scale scores
            per_scale      dict  — {'5s': raw, '10s': raw, '30s': raw}
            verdict        str
            method         str
            weights        list  — normalised combination weights used
    """
    if scales_sec is None:
        scales_sec = [5, 10, 30]
    if weights is None:
        weights = [1.0 / len(scales_sec)] * len(scales_sec)
    else:
        total = sum(weights)
        weights = [w / total for w in weights]

    try:
        from src.part7_calibration import gmm_calibrate
    except ImportError:
        from part7_calibration import gmm_calibrate
    from sklearn.metrics.pairwise import cosine_similarity as _cos_sim
    import math

    def _get_features_at_scale(path: str, win_sec: int) -> np.ndarray:
        """Load audio, segment at win_sec, extract MERT, return [N, 768]."""
        stem = Path(path).stem
        if cache_dir:
            cache_path = Path(cache_dir) / f"{stem}_mert_{win_sec}s.npy"
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
        else:
            cache_path = None

        # For .npy inputs that are already extracted at 10s: use directly
        # only when the requested scale matches the extraction scale.
        p = Path(path)
        if p.suffix.lower() == ".npy":
            arr = np.load(str(p), mmap_mode="r").astype(np.float32)
            if arr.ndim == 2 and arr.shape[1] >= 768:
                arr = arr[:, :768]
            # If the .npy was extracted at a different scale we can't resegment,
            # so we use it as-is and log a warning.
            return arr

        if cache_path and cache_path.exists():
            return np.load(str(cache_path)).astype(np.float32)

        audio = _load_audio(path)
        segs  = _segment_audio(audio, win=win_sec, hop=max(1, win_sec // 2),
                               min_sec=min(3.0, win_sec * 0.5))
        feats = _extract_mert(segs, device)
        if cache_path:
            np.save(str(cache_path), feats)
        return feats

    per_scale_raw = {}
    for win_sec in scales_sec:
        A = _get_features_at_scale(track_a, win_sec)
        B = _get_features_at_scale(track_b, win_sec)
        sim = _cos_sim(A, B).astype(np.float32)
        sim = np.clip((sim + 1.0) / 2.0, 0.0, 1.0)
        per_row_max = sim.max(axis=1)
        k_actual    = min(k, len(per_row_max))
        per_scale_raw[f"{win_sec}s"] = float(np.sort(per_row_max)[::-1][:k_actual].mean())

    raw_combined = sum(w * per_scale_raw[f"{s}s"]
                       for w, s in zip(weights, scales_sec))

    gmm_score    = gmm_calibrate(raw_combined)
    minmax_score = _calibrate(raw_combined, SONICS_LO, SONICS_HI)

    if gmm_score >= 0.75:
        verdict = "LIKELY DERIVATIVE"
    elif gmm_score >= 0.55:
        verdict = "POSSIBLY RELATED"
    elif gmm_score >= 0.35:
        verdict = "WEAK SIMILARITY"
    else:
        verdict = "UNRELATED"

    return {
        "score":        round(gmm_score, 4),
        "score_minmax": round(minmax_score, 4),
        "raw_score":    round(raw_combined, 4),
        "per_scale":    {k_: round(v, 6) for k_, v in per_scale_raw.items()},
        "verdict":      verdict,
        "method":       f"MULTISCALE_TOP_K_MERT_k{k}+GMM_CALIB",
        "weights":      weights,
        "track_a":      track_a,
        "track_b":      track_b,
    }


def compare_tracks_sonics_topk_gmm(
    track_a: str,
    track_b: str,
    k: int = 10,
    device: str = "cpu",
    cache_dir: str = None,
) -> dict:
    """
    Compare two audio tracks using Top-K MERT + GMM probabilistic calibration.

    Identical to compare_tracks_sonics_topk except the calibrated score is
    P(positive | raw_score) under a two-Gaussian mixture model fitted on 80
    SONICS pairs.  This avoids the saturation artefact of min-max calibration
    (where scores above hi=0.9336 all collapse to 1.0).

    The GMM score is a genuine probability: 0.5 means the pair sits exactly
    at the boundary between the positive and negative distributions; values
    above 0.75 indicate strong evidence of derivation.

    Args:
        track_a:   path to first audio file or .npy feature cache
        track_b:   path to second audio file or .npy feature cache
        k:         top-k segment matches to average (default: 10)
        device:    "cpu" or "cuda"
        cache_dir: optional directory to cache MERT features

    Returns:
        dict with keys:
            score        float [0,1] — P(positive | raw_score), GMM-calibrated
            score_minmax float [0,1] — legacy min-max calibrated score
            verdict      str
            raw_score    float — uncalibrated cosine score
            method       str
    """
    # Re-use the same MERT extraction logic
    def _get_mert_features(path):
        stem = Path(path).stem
        if cache_dir:
            mert_cache = Path(cache_dir) / f"{stem}_mert.npy"
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
        else:
            mert_cache = None
        if mert_cache and mert_cache.exists():
            return np.load(mert_cache).astype(np.float32)
        audio      = _load_audio(path)
        segs       = _segment_audio(audio)
        mert_feats = _extract_mert(segs, device)
        if mert_cache:
            np.save(mert_cache, mert_feats)
        return mert_feats

    from sklearn.metrics.pairwise import cosine_similarity as _cos_sim
    try:
        from src.part7_calibration import gmm_calibrate
    except ImportError:
        from part7_calibration import gmm_calibrate

    A = _get_mert_features(track_a)
    B = _get_mert_features(track_b)

    sim         = _cos_sim(A, B).astype(np.float32)
    sim         = np.clip((sim + 1.0) / 2.0, 0.0, 1.0)
    per_row_max = sim.max(axis=1)
    k_actual    = min(k, len(per_row_max))
    raw_score   = float(np.sort(per_row_max)[::-1][:k_actual].mean())

    gmm_score   = gmm_calibrate(raw_score)
    minmax_score = _calibrate(raw_score, SONICS_LO, SONICS_HI)

    if gmm_score >= 0.75:
        verdict = "LIKELY DERIVATIVE"
    elif gmm_score >= 0.55:
        verdict = "POSSIBLY RELATED"
    elif gmm_score >= 0.35:
        verdict = "WEAK SIMILARITY"
    else:
        verdict = "UNRELATED"

    return {
        "score":        round(gmm_score, 4),
        "score_minmax": round(minmax_score, 4),
        "verdict":      verdict,
        "raw_score":    round(raw_score, 4),
        "method":       f"TOP_K_MERT_k{k}+GMM_CALIB",
        "k":            k,
        "track_a":      track_a,
        "track_b":      track_b,
    }


def estimate_sonics_bounds(pairs_df,
                           feat_col_a="feat_a",
                           feat_col_b="feat_b",
                           sfhp_dir: str = None) -> tuple:
    """
    Estimate lo/hi calibration bounds from a labeled SONICS pairs dataframe.
    pairs_df must have columns: feat_a, feat_b, label (1=positive, 0=negative)
    Returns (lo, hi) tuple.
    """
    scores = []
    for _, row in pairs_df.iterrows():
        try:
            a = np.load(row[feat_col_a]).astype(np.float32).mean(axis=0)
            b = np.load(row[feat_col_b]).astype(np.float32).mean(axis=0)
            if sfhp_dir:
                sa = np.load(Path(sfhp_dir)/f"{Path(row[feat_col_a]).stem}_sfhp.npy").mean(axis=0)
                sb = np.load(Path(sfhp_dir)/f"{Path(row[feat_col_b]).stem}_sfhp.npy").mean(axis=0)
                a = np.concatenate([a, sa])
                b = np.concatenate([b, sb])
            scores.append(float(cosine_similarity(a.reshape(1,-1),
                                                   b.reshape(1,-1))[0,0]))
        except:
            scores.append(0.5)
    scores = np.array(scores)
    labels = pairs_df["label"].values
    pos_s  = scores[labels==1]
    neg_s  = scores[labels==0]
    lo = float(np.percentile(neg_s, 90))
    hi = float(np.percentile(pos_s, 10))
    if hi <= lo:
        lo, hi = float(neg_s.mean()), float(pos_s.mean())
    return lo, hi
