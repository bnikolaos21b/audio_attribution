"""
Part 4 — Segment-Level Experiments
=====================================
Two sub-experiments to diagnose the mean-pooling bottleneck:

A) Top-K cosine similarity (no trained head)
   Replace mean pooling with the mean of the top-5 pairwise cosine
   similarities across all [Na × Nb] segment pairs.
   Result: recall 0.975, precision 0.516 — fires indiscriminately.

B) Hard-negative mining
   Mine 78 MIPPIA negative pairs with the highest 1294-dim cosine
   similarity (mean similarity 0.9915), retrain with pos_weight=2.45.
   Result: ROC AUC 0.635 — worse than both baseline and weighted.

Key finding: the segment-level signal exists, but extracting it
requires a trained scoring head, not raw cosine top-K.

Dependencies:
    pip install numpy sklearn
"""

import numpy as np
from sklearn.preprocessing import normalize


# ── A. Top-K cosine segment scorer ───────────────────────────────────────────

def topk_cosine_score(
    feats_a: np.ndarray,   # [Na, 1294]
    feats_b: np.ndarray,   # [Nb, 1294]
    k: int = 5,
) -> float:
    """
    Non-learned track-pair similarity score.

    Computes all pairwise cosine similarities between segments of track A
    and segments of track B, then returns the mean of the top-k values.

    Why it fails:
        np.partition picks the k globally highest values with no learned
        context. High cosine similarity between a few segments is common
        even for completely unrelated tracks — a few spectrally similar
        moments will always exist. Without a trained decision boundary,
        the scorer flags almost everything as a match.

    Diagnostic value:
        The signal IS in the features. A trained head above this matrix
        is what is missing.
    """
    assert feats_a.ndim == 2 and feats_b.ndim == 2
    assert feats_a.shape[1] == feats_b.shape[1]

    norm_a = normalize(feats_a, norm="l2")   # [Na, D]
    norm_b = normalize(feats_b, norm="l2")   # [Nb, D]
    sim    = norm_a @ norm_b.T               # [Na, Nb] cosine similarities

    flat   = sim.ravel()
    k      = min(k, len(flat))
    top_k  = np.partition(flat, -k)[-k:]
    # Rescale from [-1,1] to [0,1] before returning
    return float(np.clip((top_k.mean() + 1.0) / 2.0, 0.0, 1.0))


# ── B. Hard-negative mining ───────────────────────────────────────────────────

def mine_hard_negatives(
    neg_pairs: list[tuple[np.ndarray, np.ndarray]],
    n_mine: int = 78,
    top_k: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Select the n_mine negative pairs with the highest top-K cosine similarity.

    These "confusable" negatives are the hardest for the model to distinguish
    from genuine positives. Training on them should sharpen the decision
    boundary near the positive/negative boundary.

    What happened:
        78 hard negatives were mined (mean topK cosine 0.9915).
        Retraining with pos_weight=2.45 achieved ROC AUC 0.635 — worse
        than both the weighted model (0.739) and the baseline (0.838).
        Hard negatives this close to positives confused the model rather
        than sharpening it.
    """
    scored = []
    for a, b in neg_pairs:
        score = topk_cosine_score(a, b, k=top_k)
        scored.append((score, a, b))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [(a, b) for _, a, b in scored[:n_mine]]


# ── Results (pre-computed) ────────────────────────────────────────────────────
# Sources:
#   results/segment/segment_topk_1294_honest_eval_topk5_summary.json
#   results/hardneg/hardneg_1294_honest_eval_summary.json

TOPK_RESULTS = {
    # k=5, non-learned, weighted 1294 features, 10-seed mean
    "roc_auc":   0.6736,
    "pr_auc":    0.6870,
    "accuracy":  0.5292,
    "precision": 0.5158,
    "recall":    0.9750,   # fires on almost everything
    "f1":        0.6745,
}

HARDNEG_RESULTS = {
    # 78 hard negatives mined (mean sim 0.9915), pos_weight=2.45, 10-seed mean
    "roc_auc":              0.6347,
    "pr_auc":               0.6483,
    "accuracy_default":     0.5875,
    "precision_default":    0.5737,
    "recall_default":       0.7250,
    "f1_default":           0.6354,
    # Calibrated at best-threshold (~0.470)
    "accuracy_calibrated":  0.6542,
    "precision_calibrated": 0.6110,
    "recall_calibrated":    0.9000,
    "f1_calibrated":        0.7235,
}
