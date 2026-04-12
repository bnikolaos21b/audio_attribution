#!/usr/bin/env python3
"""
Real→AI MERT Calibration Experiment
======================================
Computes MERT Top-K similarity between real songs and their AI-generated
derivatives from the SONICS dataset, then reports calibration parameters.

Key finding: MERT AUC = 0.5725 (barely above chance) for real→AI pairs.
The gap is NEGATIVE (-0.014): derivatives score LOWER than unrelated pairs.
Reason: MERT captures AI generator fingerprint more strongly than source relationship.

Results saved to: results/real_ai/real_ai_calibration.json

Usage:
  python src/real_ai_calibration.py
"""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
_SRC_DIR     = Path(__file__).parent
_PROJECT_DIR = _SRC_DIR.parent
PAIRS_DIR    = _PROJECT_DIR / "data" / "real_ai_pairs"
RESULTS_DIR  = _PROJECT_DIR / "results" / "real_ai"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

with open(PAIRS_DIR / "real_ai_pairs_mapping.json") as f:
    pairs = json.load(f)


def cosine(a, b):
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))


def topk_cosine(segs_a, segs_b, k=10):
    """Top-K mean cosine similarity between two segment arrays."""
    a_n = segs_a / (np.linalg.norm(segs_a, axis=1, keepdims=True) + 1e-12)
    b_n = segs_b / (np.linalg.norm(segs_b, axis=1, keepdims=True) + 1e-12)
    sim  = a_n @ b_n.T
    top1 = np.max(sim, axis=1)
    topk = np.sort(top1)[-k:]
    return float(np.mean(topk))


def load_embedding(name):
    """Load MERT embedding from cache."""
    p = PAIRS_DIR / f"{name}_mert.npy"
    return np.load(p) if p.exists() else None


def run():
    print("=" * 72)
    print("  REAL→AI MERT CALIBRATION")
    print("=" * 72)

    positive_scores = []
    negative_scores = []
    pair_details    = []
    complete        = 0

    all_fids = list(pairs.keys())

    for fid, info in pairs.items():
        real_emb = load_embedding(f"real_{fid}")
        if real_emb is None:
            continue

        ai_files = [info["fake_files"][0], info["fake_files"][1]] \
            if len(info.get("fake_files", [])) >= 2 \
            else info.get("fake_files", [])

        for af in ai_files:
            ai_name = f"ai_{fid}_{af}"
            ai_emb  = load_embedding(ai_name)
            if ai_emb is None:
                continue

            # Handle shape variations
            if real_emb.ndim == 1:
                real_emb = real_emb.reshape(1, -1)
            if ai_emb.ndim == 1:
                ai_emb = ai_emb.reshape(1, -1)

            sim = topk_cosine(real_emb, ai_emb)
            positive_scores.append(sim)
            pair_details.append({
                "fid": fid,
                "title": info["real_title"],
                "artist": info["real_artist"],
                "algorithm": info.get("algorithm", "unknown"),
                "ai_file": ai_name,
                "similarity": round(sim, 5),
            })

        complete += 1

    # Build negative scores: real from one song vs AI from a different song
    np.random.seed(42)
    neg_fids = list(pairs.keys())
    for fid_a in neg_fids:
        real_emb = load_embedding(f"real_{fid_a}")
        if real_emb is None:
            continue
        neg_pool = [x for x in neg_fids if x != fid_a]
        sampled  = np.random.choice(neg_pool, min(2, len(neg_pool)), replace=False)
        for fid_b in sampled:
            info_b = pairs[fid_b]
            af = (info_b.get("fake_files") or [""])[0]
            ai_name = f"ai_{fid_b}_{af}"
            ai_emb  = load_embedding(ai_name)
            if ai_emb is None:
                continue
            if real_emb.ndim == 1: real_emb = real_emb.reshape(1, -1)
            if ai_emb.ndim == 1: ai_emb = ai_emb.reshape(1, -1)
            sim = topk_cosine(real_emb, ai_emb)
            negative_scores.append(sim)

    if len(positive_scores) < 5 or len(negative_scores) < 5:
        print("  Not enough pairs with cached MERT embeddings to compute AUC.")
        print("  Run extract_real_ai_calibration.py first to build the embedding cache.")
        return

    pos = np.array(positive_scores)
    neg = np.array(negative_scores)
    all_scores = np.concatenate([pos, neg])
    all_labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    auc = roc_auc_score(all_labels, all_scores)

    print(f"\n  Positive pairs:   {len(pos)}")
    print(f"  Negative pairs:   {len(neg)}")
    print(f"  Positive mean:    {pos.mean():.5f} ± {pos.std():.5f}")
    print(f"  Negative mean:    {neg.mean():.5f} ± {neg.std():.5f}")
    print(f"  Gap (pos-neg):    {pos.mean() - neg.mean():+.5f}")
    print(f"  ROC AUC:          {auc:.4f}")
    print(f"\n  Conclusion: MERT is NOT reliable for real→AI attribution (AUC={auc:.4f})")

    output = {
        "n_pairs":        complete,
        "n_positive":     int(len(pos)),
        "n_negative":     int(len(neg)),
        "positive_mean":  round(float(pos.mean()), 5),
        "positive_std":   round(float(pos.std()),  5),
        "negative_mean":  round(float(neg.mean()), 5),
        "negative_std":   round(float(neg.std()),  5),
        "gap":            round(float(pos.mean() - neg.mean()), 5),
        "roc_auc":        round(float(auc), 4),
        "gmm_params": {
            "mu_pos":    round(float(pos.mean()), 5),
            "sigma_pos": round(float(pos.std()),  5),
            "mu_neg":    round(float(neg.mean()), 5),
            "sigma_neg": round(float(neg.std()),  5),
            "roc_auc":   round(float(auc), 4),
            "gap":       round(float(pos.mean() - neg.mean()), 5),
            "n_positive": int(len(pos)),
            "n_negative": int(len(neg)),
        },
        "pair_details": pair_details,
    }

    out_path = RESULTS_DIR / "real_ai_calibration.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved → {out_path}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Real→AI MERT Calibration", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.hist(pos, bins=15, alpha=0.5, color="#27AE60", label=f"Derivative (n={len(pos)})", density=True)
    ax.hist(neg, bins=15, alpha=0.5, color="#E74C3C", label=f"Unrelated (n={len(neg)})", density=True)
    ax.axvline(pos.mean(), color="#27AE60", ls="--", lw=2)
    ax.axvline(neg.mean(), color="#E74C3C", ls="--", lw=2)
    ax.set_xlabel("MERT Top-K Similarity")
    ax.set_ylabel("Density")
    ax.set_title(f"ROC AUC = {auc:.4f}  (gap = {pos.mean()-neg.mean():+.4f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.text(0.05, 0.95,
            "⚠ Negative gap: AI fingerprint\ndominates source relationship",
            transform=ax.transAxes, va="top", fontsize=9, color="#8B0000",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7))

    ax = axes[1]
    sorted_sims = sorted(pair_details, key=lambda x: x["similarity"])
    names = [f"{d['artist'][:15]}" for d in sorted_sims]
    sims  = [d["similarity"] for d in sorted_sims]
    colors = ["#E74C3C" if s < neg.mean() else "#27AE60" for s in sims]
    ax.barh(range(len(names)), sims, color=colors, alpha=0.7)
    ax.axvline(pos.mean(), color="#27AE60", ls="--", lw=1.5, label=f"Derivative mean ({pos.mean():.3f})")
    ax.axvline(neg.mean(), color="#E74C3C", ls="--", lw=1.5, label=f"Unrelated mean ({neg.mean():.3f})")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("MERT Top-K Similarity")
    ax.set_title("Per-pair similarities (positive pairs)")
    ax.legend(fontsize=8); ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(RESULTS_DIR.parent / "plots" / "real_ai_calibration.png"),
                dpi=150, bbox_inches="tight")
    print(f"  Plot saved → results/plots/real_ai_calibration.png")


if __name__ == "__main__":
    run()
