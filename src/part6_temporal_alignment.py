"""
Part 6 — Temporal Alignment Scoring Study
==========================================
Test whether accounting for temporal structure in the [Na × Nb] segment
similarity matrix improves track-pair scores.

Six methods compared:
    - siamese_mean_global   (Part 5 best, no temporal alignment)
    - cosine_mean_global    (raw cosine, no alignment)
    - dtw_cosine_mean_path  (Dynamic Time Warping, mean along aligned path)
    - dtw_cosine_max_path   (DTW, max along aligned path)
    - banded_diagonal_mean  (±15% band around main diagonal, mean)
    - banded_diagonal_max   (±15% band, max)

Key finding: banded_diagonal_mean wins (ROC 0.742, PR 0.784).
It is nearly as accurate as DTW but orders of magnitude faster.

Source: results/temporal/temporal_alignment_scoring_study_summary.csv

Dependencies:
    pip install numpy scikit-learn
"""

import numpy as np
from sklearn.preprocessing import normalize


# ── Banded diagonal mean (selected method) ────────────────────────────────────

def banded_diagonal_mean(
    sim_matrix: np.ndarray,
    band_frac: float = 0.15,
) -> float:
    """
    Mean cosine similarity along a temporal band around the main diagonal.

    For each segment i in track A, the corresponding position in track B
    is estimated as j_center = round(i * (m-1) / (n-1)). Values within
    ±band of j_center are included in the mean.

    Why this works for plagiarism detection:
        When material is borrowed, similar segments tend to appear at
        proportionally similar positions (the chorus appears in both tracks
        at roughly the same relative time). Following the diagonal captures
        this alignment without the O(n×m) cost of full DTW.

    band_frac=0.15 allows up to ±15% temporal shift between tracks.
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


def cosine_similarity_matrix(
    feats_a: np.ndarray,   # [Na, D]
    feats_b: np.ndarray,   # [Nb, D]
) -> np.ndarray:
    """Normalised cosine similarity matrix, rescaled to [0, 1]."""
    norm_a = normalize(feats_a, norm="l2")
    norm_b = normalize(feats_b, norm="l2")
    sim    = norm_a @ norm_b.T                          # [-1, 1]
    return np.clip((sim + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)


def score_pair_temporal(
    feats_a: np.ndarray,
    feats_b: np.ndarray,
    band_frac: float = 0.15,
) -> float:
    """Full pipeline: cosine matrix → banded diagonal mean."""
    sim = cosine_similarity_matrix(feats_a, feats_b)
    return banded_diagonal_mean(sim, band_frac=band_frac)


# ── Results ───────────────────────────────────────────────────────────────────
# Source: results/temporal/temporal_alignment_scoring_study_summary.csv

TEMPORAL_RESULTS = [
    # (method, roc_auc_mean, pr_auc_mean)
    ("banded_diagonal_mean",  0.7424, 0.7841),   # ← winner
    ("dtw_cosine_mean_path",  0.7382, 0.7870),
    ("siamese_mean_global",   0.7382, 0.7034),
    ("cosine_mean_global",    0.7306, 0.7624),
    ("banded_diagonal_max",   0.7236, 0.7707),
    ("dtw_cosine_max_path",   0.7167, 0.7622),
]
# Improvement over global mean cosine: +0.011 ROC AUC, +0.022 PR AUC
