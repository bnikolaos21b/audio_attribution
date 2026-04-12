#!/usr/bin/env python3
"""
Quick calibration from available MERT embeddings (whatever is cached).
"""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAIRS_DIR = Path("/home/nikos_pc/Desktop/Orfium/orfium_audio_attribution_task/qwen/results/real_ai_pairs")
RESULTS_DIR = PAIRS_DIR.parent

# Load mapping
mapping_path = PAIRS_DIR / "real_ai_pairs_mapping.json"
with open(mapping_path) as f:
    pairs = json.load(f)

def cosine(a, b):
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))

# Find available embeddings
def get_emb(name):
    p = PAIRS_DIR / f"{name}_mert.npy"
    if p.exists():
        return np.load(p)
    return None

# Positive: real → AI (same source)
positive_scores = []
pair_details = []
complete = 0

for fid, info in pairs.items():
    real_emb = get_emb(f"real_{fid}")
    if real_emb is None:
        continue
    
    ai_count = 0
    for ai_fn in info['fake_files']:
        # AI file naming: ai_{fid}_{fake_filename_base}.mp3 → ai_{fid}_{fake_filename_base}_mert.npy
        ai_name = f"ai_{fid}_{ai_fn}"
        ai_emb = get_emb(ai_name)
        if ai_emb is not None:
            sim = cosine(real_emb, ai_emb)
            positive_scores.append(sim)
            pair_details.append({
                'fid': fid,
                'title': info['real_title'],
                'artist': info['real_artist'],
                'algorithm': info['algorithm'],
                'ai_file': ai_name,
                'similarity': round(sim, 5),
            })
            ai_count += 1
    
    if ai_count > 0:
        complete += 1

positive_scores = np.array(positive_scores)
print(f"Complete pairs: {complete}")
print(f"Positive pairs (real→AI derivative): {len(positive_scores)}")

if len(positive_scores) < 5:
    print("Too few positive pairs. Exiting.")
    exit(1)

# Negative: real → AI (different source)
negative_scores = []
fids = list(pairs.keys())

for fid_a in fids:
    real_emb = get_emb(f"real_{fid_a}")
    if real_emb is None:
        continue
    
    other_fids = [f for f in fids if f != fid_a]
    np.random.seed(42 + int(fid_a))
    sampled = list(np.random.choice(other_fids, min(3, len(other_fids)), replace=False))
    
    for fid_b in sampled:
        ai_fn = pairs[fid_b]['fake_files'][0]
        ai_name = f"ai_{fid_b}_{ai_fn}"
        ai_emb = get_emb(ai_name)
        if ai_emb is not None:
            sim = cosine(real_emb, ai_emb)
            negative_scores.append(sim)

negative_scores = np.array(negative_scores)
print(f"Negative pairs (real→AI unrelated): {len(negative_scores)}")

if len(negative_scores) < 5:
    print("Too few negative pairs. Exiting.")
    exit(1)

# Statistics
mu_pos = float(np.mean(positive_scores))
sigma_pos = float(np.std(positive_scores))
mu_neg = float(np.mean(negative_scores))
sigma_neg = float(np.std(negative_scores))
gap = mu_pos - mu_neg

print(f"\nPositive: mean={mu_pos:.5f}, std={sigma_pos:.5f}, range=[{np.min(positive_scores):.5f}, {np.max(positive_scores):.5f}]")
print(f"Negative: mean={mu_neg:.5f}, std={sigma_neg:.5f}, range=[{np.min(negative_scores):.5f}, {np.max(negative_scores):.5f}]")
print(f"Gap: {gap:+.5f}")

# AUC
all_scores = np.concatenate([positive_scores, negative_scores])
all_labels = np.concatenate([np.ones(len(positive_scores)), np.zeros(len(negative_scores))])
auc = roc_auc_score(all_labels, all_scores)
print(f"ROC AUC: {auc:.4f}")

# Save calibration
output = {
    'n_pairs': complete,
    'n_positive': int(len(positive_scores)),
    'n_negative': int(len(negative_scores)),
    'positive_mean': round(mu_pos, 5),
    'positive_std': round(sigma_pos, 5),
    'negative_mean': round(mu_neg, 5),
    'negative_std': round(sigma_neg, 5),
    'gap': round(gap, 5),
    'roc_auc': round(auc, 4),
    'gmm': {
        'mu_pos': round(mu_pos, 5),
        'sigma_pos': round(sigma_pos, 5),
        'mu_neg': round(mu_neg, 5),
        'sigma_neg': round(sigma_neg, 5),
    },
    'gmm_params': {
        'mu_pos': round(mu_pos, 5),
        'sigma_pos': round(sigma_pos, 5),
        'mu_neg': round(mu_neg, 5),
        'sigma_neg': round(sigma_neg, 5),
        'roc_auc': round(auc, 4),
        'gap': round(gap, 5),
        'n_positive': int(len(positive_scores)),
        'n_negative': int(len(negative_scores)),
    },
    'pair_details': pair_details,
}

out_path = RESULTS_DIR / "real_ai_calibration.json"
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nCalibration saved → {out_path}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"Real→AI Ground Truth Calibration\nn={complete} pairs, AUC={auc:.4f}, gap={gap:+.5f}",
             fontsize=14, fontweight="bold")

ax = axes[0]
ax.hist(positive_scores, bins=15, alpha=0.5, label=f"Derivative (n={len(positive_scores)})", color="#27AE60", density=True)
ax.hist(negative_scores, bins=15, alpha=0.5, label=f"Unrelated (n={len(negative_scores)})", color="#E74C3C", density=True)
ax.axvline(mu_pos, color="#27AE60", ls="--", lw=1.5)
ax.axvline(mu_neg, color="#E74C3C", ls="--", lw=1.5)
ax.set_xlabel("MERT Cosine Similarity")
ax.set_ylabel("Density")
ax.set_title("Distributions")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1]
by_pair = {}
for d in pair_details:
    key = f"{d['fid']}_{d['title'][:20]}"
    by_pair.setdefault(key, []).append(d['similarity'])

pairs_sorted = sorted(by_pair.items(), key=lambda x: np.mean(x[1]))
means = [np.mean(v) for k, v in pairs_sorted]
mins = [min(v) for k, v in pairs_sorted]
maxs = [max(v) for k, v in pairs_sorted]
labels = [k[:25] for k, v in pairs_sorted]

y = np.arange(len(means))
ax.errorbar(means, y, xerr=[np.array(means) - np.array(mins), np.array(maxs) - np.array(means)],
            fmt='o', color='#3498DB', alpha=0.7, capsize=3)
ax.axvline(mu_neg, color="#E74C3C", ls="--", lw=1, alpha=0.7, label="Unrelated mean")
ax.axvline(mu_pos, color="#27AE60", ls="--", lw=1, alpha=0.7, label="Derivative mean")
ax.set_xlabel("Cosine Similarity")
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=5)
ax.set_title("Per-Pair Similarities (±range)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_png = RESULTS_DIR / "real_ai_calibration.png"
plt.savefig(str(out_png), dpi=150, bbox_inches="tight")
print(f"Plot saved → {out_png}")
