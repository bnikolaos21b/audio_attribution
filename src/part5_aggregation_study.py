"""
Part 5 — Pairwise Score Aggregation Study
==========================================
Systematically compare 8 aggregation strategies for collapsing a
[Na × Nb] segment similarity matrix into a single track-pair score.

Two backends: Siamese model scores vs. raw cosine similarity.
Four aggregations: mean / max / top-K mean / softmax-weighted.

Key finding: Siamese + mean (ROC 0.738) and cosine + top-K (ROC 0.731)
are the best per backend. None exceed the 782-dim baseline (ROC 0.838).

Source: results/aggregation/pairwise_score_aggregation_study_summary.csv

Dependencies:
    pip install numpy torch
"""

import numpy as np
import torch
from typing import Literal


# ── Aggregation functions ─────────────────────────────────────────────────────

AggMethod = Literal["mean", "max", "topk_mean", "softmax_weighted"]


def aggregate_scores(
    sim_matrix: np.ndarray,
    method: AggMethod = "mean",
    k: int = 5,
) -> float:
    """
    Collapse a [Na, Nb] similarity matrix to a single scalar.

    Args:
        sim_matrix: pairwise scores, values in [0, 1]
        method:     aggregation strategy
        k:          top-K window (only for topk_mean)

    Returns:
        float in [0, 1]
    """
    flat = sim_matrix.ravel()

    if method == "mean":
        return float(flat.mean())

    elif method == "max":
        return float(flat.max())

    elif method == "topk_mean":
        k_eff = min(k, len(flat))
        return float(np.partition(flat, -k_eff)[-k_eff:].mean())

    elif method == "softmax_weighted":
        # Softmax-weighted mean: higher scores get more weight
        weights = np.exp(flat) / np.exp(flat).sum()
        return float((weights * flat).sum())

    else:
        raise ValueError(f"Unknown method: {method}")


# ── Siamese pairwise matrix ───────────────────────────────────────────────────

def siamese_pairwise_matrix(
    model: torch.nn.Module,
    feats_a: np.ndarray,   # [Na, 1294]
    feats_b: np.ndarray,   # [Nb, 1294]
    device: str = "cpu",
) -> np.ndarray:
    """
    Build [Na, Nb] matrix of Siamese model scores for all segment pairs.

    The model was trained on mean-pooled vectors, so segment-level
    scores are not calibrated — this is why Siamese + max and
    Siamese + top-K perform worse than Siamese + mean.
    """
    model.eval()
    scores = np.zeros((len(feats_a), len(feats_b)), dtype=np.float32)

    fa = torch.tensor(feats_a, dtype=torch.float32, device=device)
    fb = torch.tensor(feats_b, dtype=torch.float32, device=device)

    with torch.no_grad():
        for i in range(len(fa)):
            xa = fa[i].unsqueeze(0).expand(len(fb), -1)
            logits = model(xa, fb)
            scores[i] = torch.sigmoid(logits).cpu().numpy()

    return scores


# ── Results ───────────────────────────────────────────────────────────────────
# Source: results/aggregation/pairwise_score_aggregation_study_summary.csv

AGGREGATION_RESULTS = [
    # (backend, aggregation, roc_auc_mean, pr_auc_mean)
    ("siamese", "mean",             0.7382, 0.7034),
    ("siamese", "softmax_weighted", 0.7340, 0.7012),
    ("cosine",  "topk_mean",        0.7306, 0.7734),
    ("cosine",  "softmax_weighted", 0.7236, 0.7673),
    ("cosine",  "max",              0.7215, 0.7676),
    ("cosine",  "mean",             0.7160, 0.7570),
    ("siamese", "topk_mean",        0.6736, 0.6870),
    ("siamese", "max",              0.6611, 0.6750),
]
# Note: cosine backends outperform Siamese on top-K/max because the model
# was trained on mean-pooled vectors and is not calibrated at segment level.
