"""
Final Similarity Function
==========================
Compares two audio tracks and returns a similarity score in [0, 1].

Pipeline:
    1. Load pre-extracted segment feature matrices ([N, 782] .npy files)
    2. Compute pairwise cosine similarity matrix [Na × Nb]
    3. Aggregate with banded_diagonal_mean (best from Part 6)
    4. Calibrate with min-max rescaling (best from Part 7)

This function does not require a loaded neural network at inference time.
The cosine + temporal alignment + calibration pipeline is robust, fast,
and fully traceable to the experiments in Parts 5–7.

Usage:
    from src.final_similarity_function import compute_similarity

    score = compute_similarity(
        "features/track_a.npy",
        "features/track_b.npy",
        lo=0.88,    # MIPPIA-domain lower bound (typical unrelated-pair score)
        hi=0.99,    # MIPPIA-domain upper bound (typical derivative-pair score)
    )
    print(f"Similarity: {score:.1%}")   # e.g. "Similarity: 73.4%"

Feature extraction:
    Use src/compare_tracks_api.py → compare_tracks(audio_path_a, audio_path_b)
    It extracts 1294-dim features via FeatureExtractor, slices to 782-dim
    (dropping CLAP), and caches [N_segments, 782] float32 .npy files.
    Do NOT call FeatureExtractor.extract_and_cache() directly — it saves
    1294-dim features which will fail the 782-dim validation in load_features().
    Segments are 10-second windows with 5-second hop.

Calibration bounds:
    lo and hi are determined from the score distribution over known pairs.
    From MIPPIA: negative pairs score ≈ 0.88–0.93, positive pairs ≈ 0.95–0.99.
    These bounds are MIPPIA-domain approximations. They do not transfer to
    AI-generated audio (SONICS/Suno/Udio) without recalibration on domain-
    specific labeled pairs.

Dependencies:
    pip install numpy scikit-learn
"""

from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def load_features(npy_path: str | Path) -> np.ndarray:
    """Load a [N_segments, 782] float32 array from a .npy cache file."""
    arr = np.load(npy_path).astype(np.float32)
    assert arr.ndim == 2, f"Expected 2-D array, got shape {arr.shape}"
    if arr.shape[-1] != 782:
        raise ValueError(
            f"compute_similarity expects 782-dim features, "
            f"got {arr.shape[-1]}. "
            f"If you used FeatureExtractor (which produces 1294-dim output), "
            f"drop the CLAP block first: "
            f"np.concatenate([features[:, :768], features[:, 1280:]], axis=1). "
            f"Do NOT use features[:, :782] — that retains CLAP dims 768-781 "
            f"instead of Chroma/HP/SF, producing wrong vectors that pass this check silently."
        )
    return arr


def banded_diagonal_mean(sim_matrix: np.ndarray, band_frac: float = 0.15) -> float:
    """
    Mean cosine similarity within a temporal band around the main diagonal.

    For each segment i in track A, looks at the proportionally corresponding
    position in track B (j_center = round(i × (m-1) / (n-1))) and averages
    all similarity values within ±band_frac of the longer track's length.

    Captures temporal correspondence without the O(n×m) cost of full DTW.
    Best method in the temporal alignment study (ROC AUC 0.742, PR AUC 0.784).
    """
    n, m = sim_matrix.shape
    band = max(1, int(max(n, m) * band_frac))
    vals = []
    for i in range(n):
        j_center = int(round(i * (m - 1) / max(n - 1, 1)))
        j0 = max(0, j_center - band)
        j1 = min(m, j_center + band + 1)
        vals.extend(sim_matrix[i, j0:j1].tolist())
    vals = np.array(vals, dtype=np.float32)
    return float(vals.mean()) if len(vals) else 0.0


def minmax_calibrate(score: float, lo: float, hi: float) -> float:
    """
    Rescale a raw score from [lo, hi] to [0, 1].

    lo: typical score for unrelated track pairs (~0.88 on MIPPIA)
    hi: typical score for confirmed derivatives (~0.99 on MIPPIA)

    Scores outside [lo, hi] are clipped. Best calibration method
    from Part 7: F1 0.740 at calibrated threshold ≈ 0.653.
    """
    if hi - lo < 1e-12:
        return 0.5
    return float(np.clip((score - lo) / (hi - lo), 0.0, 1.0))


def compute_similarity(
    track_a_npy: str | Path,
    track_b_npy: str | Path,
    lo: float,
    hi: float,
) -> float:
    """
    Return a calibrated similarity score in [0, 1] for two audio tracks.

    Args:
        track_a_npy: path to .npy feature file for track A
        track_b_npy: path to .npy feature file for track B
        lo:          calibration lower bound (unrelated-pair score level)
        hi:          calibration upper bound (derivative-pair score level)

    Returns:
        float in [0, 1] — higher means more similar / more likely derivative
    """
    A = load_features(track_a_npy)   # [Na, 782]
    B = load_features(track_b_npy)   # [Nb, 782]

    # Cosine similarity matrix, rescaled from [-1, 1] to [0, 1]
    sim = cosine_similarity(A, B).astype(np.float32)
    sim = np.clip((sim + 1.0) / 2.0, 0.0, 1.0)

    raw        = banded_diagonal_mean(sim)          # temporally-aligned mean
    calibrated = minmax_calibrate(raw, lo, hi)      # [lo, hi] → [0, 1]
    return float(calibrated)
