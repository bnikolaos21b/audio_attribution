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

GMM calibration (added post-submission):
    Models positive and negative score distributions as Gaussians and outputs
    the Bayesian posterior P(positive | score).  Avoids the saturation problem
    of minmax calibration: where minmax clips everything above hi to 1.0, GMM
    gives a continuous probability across the full score range.  ROC AUC is
    identical (monotonic transform of raw score), but the output is a genuine
    probability rather than a rescaled rank.

    Parameters fitted from 80 SONICS pairs (40 pos, 40 neg):
        positive: mean=0.9358, std=0.0183
        negative: mean=0.9243, std=0.0178
        prior: equal (0.50 / 0.50)

    The minmax calibration remains available as a fallback and for
    backward compatibility.

Source: results/calibration/temporal_score_calibration_study_summary.csv
        results/calibration/gmm_calibration_params.json

Dependencies:
    pip install numpy
"""

import math
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


# ── GMM calibration ──────────────────────────────────────────────────────────

def _gauss_pdf(x: float, mu: float, sigma: float) -> float:
    """Gaussian probability density at x."""
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))


def gmm_posterior(
    score: float,
    mu_pos: float,
    sigma_pos: float,
    mu_neg: float,
    sigma_neg: float,
    pi_pos: float = 0.5,
) -> float:
    """
    Return P(positive | score) under a two-Gaussian mixture model.

    Uses Bayes' theorem:
        P(pos | s) = P(s | pos) * pi_pos
                     ─────────────────────────────────────────
                     P(s | pos) * pi_pos + P(s | neg) * pi_neg

    Args:
        score:     raw cosine similarity score
        mu_pos:    mean of positive-pair Gaussian
        sigma_pos: std  of positive-pair Gaussian
        mu_neg:    mean of negative-pair Gaussian
        sigma_neg: std  of negative-pair Gaussian
        pi_pos:    prior probability of a positive pair (default 0.5)

    Returns:
        float in [0, 1] — probability this pair is a derivative
    """
    pi_neg = 1.0 - pi_pos
    lp = _gauss_pdf(score, mu_pos, sigma_pos) * pi_pos
    ln = _gauss_pdf(score, mu_neg, sigma_neg) * pi_neg
    denom = lp + ln
    if denom < 1e-300:
        return 0.5
    return float(lp / denom)


def fit_gmm_from_scores(
    scores: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """
    Fit a two-Gaussian mixture from labelled scores.

    Args:
        scores: 1-D array of raw similarity scores
        labels: 1-D array of integer labels (1 = positive, 0 = negative)

    Returns:
        dict with keys: mu_pos, sigma_pos, mu_neg, sigma_neg, pi_pos,
                        n_pos, n_neg
    """
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("Need at least one positive and one negative example.")
    return {
        "mu_pos":    float(np.mean(pos)),
        "sigma_pos": float(np.std(pos, ddof=1)),
        "mu_neg":    float(np.mean(neg)),
        "sigma_neg": float(np.std(neg, ddof=1)),
        "pi_pos":    float(len(pos) / (len(pos) + len(neg))),
        "n_pos":     int(len(pos)),
        "n_neg":     int(len(neg)),
    }


# Pre-fitted GMM parameters from 80 SONICS pairs (40 positive, 40 negative).
# Fit: fit_gmm_from_scores on results/sonics/sonics_all_scores.csv raw_score column.
SONICS_GMM = {
    "mu_pos":    0.935767,
    "sigma_pos": 0.018309,
    "mu_neg":    0.924318,
    "sigma_neg": 0.017772,
    "pi_pos":    0.5,
    "n_pos":     40,
    "n_neg":     40,
}


def gmm_calibrate(score: float, params: dict | None = None) -> float:
    """
    Convenience wrapper: return GMM posterior using SONICS_GMM by default.

    Args:
        score:  raw cosine similarity score
        params: GMM parameter dict (default: SONICS_GMM)

    Returns:
        float in [0, 1] — P(positive | score)
    """
    p = params if params is not None else SONICS_GMM
    return gmm_posterior(
        score,
        mu_pos=p["mu_pos"],
        sigma_pos=p["sigma_pos"],
        mu_neg=p["mu_neg"],
        sigma_neg=p["sigma_neg"],
        pi_pos=p.get("pi_pos", 0.5),
    )


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
