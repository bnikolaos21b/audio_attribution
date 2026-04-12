"""
Part 3 — Weighted Training Fix
================================
Fix the recall collapse in the 1294-dim model caused by 3:1 class imbalance.
Two-pronged correction: negative undersampling + positive class weight in loss.

Training config (models/siamese_custom_1294_weighted_train_config.json):
    epochs=60, patience=10, lr=1e-4, batch_size=32
    neg_to_pos_ratio=1.5, pos_weight=1.4842
    Epochs completed: 42, best val loss: 0.584

Key finding: recall recovered (0.800 → 0.833) but ROC AUC and PR AUC
did not improve — weighting shifts the decision boundary, it does not
improve the model's inherent discriminative ability.

Dependencies:
    pip install numpy torch
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ── Weighted data loader ──────────────────────────────────────────────────────

def build_weighted_loader(
    X: np.ndarray,
    y: np.ndarray,
    neg_to_pos_ratio: float = 1.5,
    batch_size: int = 32,
    seed: int = 42,
) -> DataLoader:
    """
    Negative undersampling + balanced DataLoader.

    Step 1 — Undersampling: reduce negatives from 3:1 to 1.5:1 ratio.
              Narrows the class imbalance before the loss sees it.
    Step 2 — pos_weight in BCEWithLogitsLoss amplifies gradient on
              false negatives (see criterion below).

    Together these shift the decision boundary toward higher recall
    without changing the model's inherent ranking ability (ROC AUC).
    """
    rng      = np.random.default_rng(seed)
    pos_idx  = np.where(y == 1)[0]   # 119 positives
    neg_idx  = np.where(y == 0)[0]   # 353 negatives
    n_keep   = int(len(pos_idx) * neg_to_pos_ratio)   # 119 × 1.5 = 178
    neg_keep = rng.choice(neg_idx, size=n_keep, replace=False)
    idx      = np.concatenate([pos_idx, neg_keep])
    rng.shuffle(idx)

    ds = TensorDataset(
        torch.tensor(X[idx], dtype=torch.float32),
        torch.tensor(y[idx], dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


def build_weighted_criterion(n_neg: int, n_pos: int) -> nn.BCEWithLogitsLoss:
    """
    BCEWithLogitsLoss with pos_weight = n_neg / n_pos.

    For the 472-pair training set: 353 / 119 ≈ 1.484.
    A false negative (missed derivative) is penalised 1.484× more
    than a false positive.
    """
    pw = torch.tensor([n_neg / n_pos], dtype=torch.float32)   # ≈ 1.484
    return nn.BCEWithLogitsLoss(pos_weight=pw)


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
) -> tuple[float, float]:
    """Run one training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for feat_a, feat_b, labels in loader:
        feat_a  = feat_a.to(device)
        feat_b  = feat_b.to(device)
        labels  = labels.to(device)
        logits  = model(feat_a, feat_b)
        loss    = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        probs   = torch.sigmoid(logits)
        preds   = (probs >= 0.5).float()
        total_loss += loss.item() * labels.size(0)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


# ── Results (pre-computed) ────────────────────────────────────────────────────
# Source: results/weighted/custom_1294_honest_eval_multi_seed.csv

WEIGHTED_RESULTS = {
    "roc_auc":   {"mean": 0.7389, "std": 0.1296},
    "pr_auc":    {"mean": 0.7012, "std": 0.1452},
    "accuracy":  {"mean": 0.7333, "std": 0.0838},
    "precision": {"mean": 0.6997, "std": 0.0820},
    "recall":    {"mean": 0.8333, "std": 0.1111},
    "f1":        {"mean": 0.7572, "std": 0.0758},
}

# Delta vs 782-dim baseline (negative = worse than baseline)
DELTA_VS_BASELINE = {
    "roc_auc":  -0.099,
    "pr_auc":   -0.165,   # most diagnostic: no threshold recovers baseline tradeoff
    "accuracy": -0.080,
    "precision":-0.126,
    "recall":   +0.033,   # only metric where weighted model wins
    "f1":       -0.051,
}
