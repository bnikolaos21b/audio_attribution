"""
Part 1 — 782-Dim Baseline Evaluation
=====================================
Evaluate the pre-existing Siamese MLP checkpoint on held-out MIPPIA pairs
using a corrected, pair-number-aware split across 10 seeds.

Key finding: ROC AUC 0.838, PR AUC 0.866, F1 0.808 (mean across 10 seeds).
This is the strongest result in the entire project.

NOTE ON REPRODUCIBILITY
-----------------------
The ROC AUC 0.838 result is pre-computed and stored in BASELINE_RESULTS /
results/baseline/baseline_multi_seed_results.csv. The full training run was
performed in Google Colab (March 28) with per-track mean-pooled 782-dim feature
cache files that are NOT included in this repo (feature_dir: "not-available" in
data/manifests/prep_manifest.json).

To use evaluate_checkpoint / run_multi_seed_eval you need:
  1. A feature directory containing one .npy file per track with shape [1, 782]
     or [782] (pre-mean-pooled). These are NOT the multi-segment [N_segments, 782]
     files produced by compare_tracks_api.py — those have N > 1 and will be
     flattened incorrectly by load_cached_vector.
  2. The pair CSV (data/pairs/…) which references MIPPIA file paths that are
     not included for copyright reasons.

Without the feature cache, evaluate_checkpoint will raise FileNotFoundError.
The BASELINE_RESULTS dict at the bottom of this file is the authoritative source.

NOTE ON SPLIT LEAKAGE
---------------------
pair_aware_split prevents pair-level leakage: both members of any given pair
land on the same side of the split. It does NOT prevent track-level leakage
across pairs — a track that appears in multiple pairs with different pair_numbers
can straddle train and test. This is a known limitation of the MIPPIA dataset
structure (57 pairs, ~127 tracks, some tracks appear in more than one pair).

Dependencies:
    pip install numpy pandas scikit-learn torch
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score,
)


# ── Model architecture ────────────────────────────────────────────────────────

class _Encoder(nn.Module):
    """
    Shared encoder branch: 782 → 512 → 256.
    Architecture reconstructed from siamese_best_v2.pt weight shapes.
    Each block is Linear + BatchNorm1d + ReLU (ReLU applied in forward).
    """
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Sequential(nn.Linear(782, 512), nn.BatchNorm1d(512)),
            nn.Sequential(nn.Linear(512, 256), nn.BatchNorm1d(256)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.net:
            x = F.relu(block(x))
        return x


class SiameseMLP(nn.Module):
    """
    Siamese MLP for binary attribution.

    Architecture reconstructed from siamese_best_v2.pt checkpoint:
      encoder (shared weights): 782 → 512 → 256  (Linear+BN+ReLU per block)
      classifier: cat([h1, h2]) 512 → 128 → 64 → 1  (Linear+BN+ReLU, then Linear)

    Forward takes two separate 782-dim tensors (NOT a single concatenated 1564-dim input).

    Usage:
        model = load_siamese_checkpoint("models/siamese_best_v2.pt")
        A = torch.tensor(track_a_features_782)  # [N, 782]
        B = torch.tensor(track_b_features_782)  # [N, 782]
        probs = torch.sigmoid(model(A, B)).squeeze()
    """
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _Encoder()
        self.classifier = nn.Sequential(
            nn.Sequential(nn.Linear(512, 128), nn.BatchNorm1d(128)),
            nn.Sequential(nn.Linear(128, 64), nn.BatchNorm1d(64)),
            nn.Linear(64, 1),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        h1 = self.encoder(x1)                          # [N, 256]
        h2 = self.encoder(x2)                          # [N, 256]
        combined = torch.cat([h1, h2], dim=1)           # [N, 512]
        for i, layer in enumerate(self.classifier):
            combined = F.relu(layer(combined)) if i < 2 else layer(combined)
        return combined                                 # [N, 1] logits


def load_siamese_checkpoint(path: str | Path) -> SiameseMLP:
    """
    Load siamese_best_v2.pt into the reconstructed SiameseMLP.
    Raises RuntimeError if the state dict does not match the architecture.
    """
    model = SiameseMLP()
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


# ── Reproducible split ────────────────────────────────────────────────────────

def pair_aware_split(
    df: pd.DataFrame,
    test_ratio: float = 0.25,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by pair_number to prevent pair-level leakage across train/test.

    The original repository split grouped by path_a (first-track file path).
    Because a track can appear as path_a in multiple pairs, this allowed
    related pairs to appear on both sides of the split.

    Note: splitting by pair_number prevents pair-level leakage (both members
    of any pair land in the same fold) but does NOT prevent track-level leakage
    across pairs — a track appearing in multiple pairs with different pair_numbers
    may straddle train and test.

    Documented in data/manifests/prep_manifest.json:
        "warning_repo_split": "train._group_aware_split groups by path_a,
                                not true pair_number"
    """
    pair_ids = df["pair_number"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(pair_ids)
    n_test   = max(1, int(len(pair_ids) * test_ratio))
    test_ids = set(pair_ids[:n_test])
    train = df[~df["pair_number"].isin(test_ids)].reset_index(drop=True)
    test  = df[ df["pair_number"].isin(test_ids)].reset_index(drop=True)
    return train, test


# ── Feature loading ───────────────────────────────────────────────────────────

def load_cached_vector(stem: str, feature_dir: Path) -> np.ndarray:
    """Load a pre-cached 782-dim feature vector for one track."""
    path = feature_dir / f"{stem}.npy"
    arr  = np.load(path).astype(np.float32)
    # Baseline cache: shape [1, 782] or [782] — always return flat [782]
    return arr.flatten()


# ── Evaluation ────────────────────────────────────────────────────────────────

def build_held_out_tensors(
    test_df: pd.DataFrame,
    feature_dir: Path,
    n_pos: int = 12,
    n_neg: int = 12,
) -> tuple[torch.Tensor, np.ndarray]:
    """Stratified held-out set: n_pos positives, n_neg negatives."""
    pos = test_df[test_df["label"] == 1].head(n_pos)
    neg = test_df[test_df["label"] == 0].head(n_neg)
    held = pd.concat([pos, neg]).reset_index(drop=True)

    pairs = []
    for _, row in held.iterrows():
        a = load_cached_vector(row["ori_stem"],  feature_dir)
        b = load_cached_vector(row["comp_stem"], feature_dir)
        pairs.append(np.concatenate([a, b]))        # simple concat for baseline model

    X = torch.tensor(np.stack(pairs), dtype=torch.float32)
    y = held["label"].values.astype(int)
    return X, y


def evaluate_checkpoint(
    model: SiameseMLP,
    X_test: torch.Tensor,
    y_test: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Return all six metrics for one seed.

    X_test must be [N, 1564]: the concatenation of two 782-dim track vectors
    as returned by build_held_out_tensors. This function splits the tensor
    and calls model(A, B) with two separate [N, 782] inputs as the SiameseMLP
    forward requires.

    IMPORTANT: requires a feature cache of per-track [1, 782] or [782] files
    (pre-mean-pooled). The multi-segment [N_segments, 782] files produced by
    compare_tracks_api.py are incompatible with load_cached_vector and will
    produce incorrect flattened inputs. See module docstring for details.
    """
    model.eval()
    with torch.no_grad():
        A, B = X_test[:, :782], X_test[:, 782:]    # split back into per-track vectors
        probs = torch.sigmoid(model(A, B)).squeeze().numpy()
    preds = (probs >= threshold).astype(int)
    return {
        "roc_auc":   roc_auc_score(y_test, probs),
        "pr_auc":    average_precision_score(y_test, probs),
        "accuracy":  accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall":    recall_score(y_test, preds, zero_division=0),
        "f1":        f1_score(y_test, preds, zero_division=0),
    }


def run_multi_seed_eval(
    model: torch.nn.Module,
    df: pd.DataFrame,
    feature_dir: Path,
    seeds: list[int] | None = None,
) -> pd.DataFrame:
    """Evaluate across 10 seeds and return per-seed results."""
    if seeds is None:
        seeds = [123, 456, 789, 1011, 2024, 3031, 4049, 5051, 6067, 7079]

    records = []
    for seed in seeds:
        _, test_df = pair_aware_split(df, seed=seed)
        X_test, y_test = build_held_out_tensors(test_df, feature_dir)
        metrics = evaluate_checkpoint(model, X_test, y_test)
        records.append({"seed": seed, **metrics})

    results = pd.DataFrame(records)
    means   = results.drop(columns="seed").mean()
    print("\n782-dim baseline — mean across 10 seeds:")
    for k, v in means.items():
        print(f"  {k:12s}: {v:.4f}")
    return results


# ── Results (pre-computed, from results/baseline/baseline_multi_seed_results.csv) ─

BASELINE_RESULTS = {
    "roc_auc":   {"mean": 0.8382, "std": 0.0627},
    "pr_auc":    {"mean": 0.8660, "std": 0.0583},
    "accuracy":  {"mean": 0.8125, "std": 0.0650},
    "precision": {"mean": 0.8261, "std": 0.0700},
    "recall":    {"mean": 0.8000, "std": 0.1120},
    "f1":        {"mean": 0.8079, "std": 0.0721},
}
