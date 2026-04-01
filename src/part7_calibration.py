"""
Part 7 — Score Calibration Study
==================================
Calibrate the raw banded_diagonal_mean score so that the output is an
interpretable similarity percentage.

Problem: raw cosine scores cluster in a narrow high range (≈0.88–0.99)
regardless of whether a pair is a derivative. A score of "0.96" is not
interpretable without knowing the background level.

Three variants tested:
    raw:    use banded_diagonal_mean score directly
    logit:  logistic transform centred on 0.5
    minmax: rescale [lo, hi] → [0, 1] using empirical bounds

Key finding: all three variants have identical ROC AUC (0.708) — calibration
does not change the ranking order, only the scale. Min-max calibration is
best for usability: the output is directly interpretable as a percentage
and the calibration parameters have a clear physical meaning.

Source: results/calibration/temporal_score_calibration_study_summary.csv

Dependencies:
    pip install numpy
"""

import numpy as np


# ── Calibration functions ─────────────────────────────────────────────────────

def minmax_calibrate(score: float, lo: float, hi: float) -> float:
    """
    Rescale score from [lo, hi] → [0, 1].

    lo and hi are empirical bounds learned from the score distribution:
        lo ≈ typical score for unrelated MIPPIA pairs  (~0.88)
        hi ≈ typical score for confirmed derivatives   (~0.99)

    Scores outside [lo, hi] are clipped to [0, 1].
    This is the chosen calibration for final_similarity_function.py.
    """
    if hi - lo < 1e-12:
        return 0.5
    return float(np.clip((score - lo) / (hi - lo), 0.0, 1.0))


def logit_calibrate(score: float, centre: float = 0.5, scale: float = 10.0) -> float:
    """
    Logistic (sigmoid) transform centred on `centre`.

    Pushes scores above centre toward 1 and below toward 0.
    Produces calibrated-threshold accuracy 0.692 — same as minmax
    but less interpretable: the centre and scale parameters have no
    direct physical meaning.
    """
    return float(1.0 / (1.0 + np.exp(-scale * (score - centre))))


# ── Calibration parameter estimation ─────────────────────────────────────────

def estimate_bounds_from_pairs(
    scores: np.ndarray,
    labels: np.ndarray,
    percentile: float = 10.0,
) -> tuple[float, float]:
    """
    Estimate lo and hi from a set of scored pairs.

    lo: `percentile`-th percentile of negative scores
    hi: (100 - percentile)-th percentile of positive scores

    Robust to outliers. Use a held-out calibration set, not the test set.
    """
    neg_scores = scores[labels == 0]
    pos_scores = scores[labels == 1]
    lo = float(np.percentile(neg_scores, 100 - percentile))
    hi = float(np.percentile(pos_scores, percentile))
    return lo, hi


# ── Results ───────────────────────────────────────────────────────────────────
# Source: results/calibration/temporal_score_calibration_study_summary.csv
# Base scorer: banded_diagonal_mean on 1294-dim cosine similarity matrix

CALIBRATION_RESULTS = {
    # All variants share identical ROC/PR AUC (calibration = monotonic transform)
    "roc_auc":   0.7076,
    "pr_auc":    0.7612,
    # At default 0.5 threshold:
    "raw":    {"accuracy": 0.500,  "precision": 0.500,  "recall": 1.000, "f1": 0.667},
    "logit":  {"accuracy": 0.629,  "precision": 0.632,  "recall": 0.692, "f1": 0.654},
    "minmax": {"accuracy": 0.563,  "precision": 0.544,  "recall": 0.933, "f1": 0.684},
    # At best (calibrated) threshold ≈ 0.653 for minmax:
    "calibrated": {
        "best_threshold": 0.6527,
        "accuracy":   0.692,
        "precision":  0.690,
        "recall":     0.850,
        "f1":         0.740,
    },
}
