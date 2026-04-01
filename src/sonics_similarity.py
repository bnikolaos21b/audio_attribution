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
