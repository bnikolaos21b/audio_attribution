#!/usr/bin/env python3
"""
Task 1 — Data Leakage Check
=============================
Check whether real songs appear in BOTH positive and negative pairs.
Re-run AUC with strict song-level splits: each real song in EITHER positive OR negative, never both.
"""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

PAIRS_DIR = Path("/home/nikos_pc/Desktop/Orfium/orfium_audio_attribution_task/qwen/results/real_ai_pairs")

# Load transcripts
with open(PAIRS_DIR / "transcripts.json") as f:
    transcripts = json.load(f)

# Load mapping
mapping_path = PAIRS_DIR / "real_ai_pairs_mapping.json"
with open(mapping_path) as f:
    pairs = json.load(f)

# Load results for comparison
with open(PAIRS_DIR.parent / "lyrics_attribution_results.json") as f:
    orig_results = json.load(f)

def get_real_path(fid):
    p = PAIRS_DIR / f"real_{fid}.mp3"
    if p.exists():
        return str(p)
    return None

def get_ai_paths(fid):
    return sorted(PAIRS_DIR.glob(f"ai_{fid}_*.mp3"))

# Build complete pairs list
complete_pairs = {}
for fid, info in pairs.items():
    real_path = get_real_path(fid)
    ai_paths = get_ai_paths(fid)
    if real_path and len(ai_paths) > 0:
        complete_pairs[int(fid)] = {
            'real': real_path,
            'ai': [str(p) for p in ai_paths],
            'title': info['real_title'],
            'artist': info['real_artist'],
        }

def get_transcript_text(path):
    key = Path(path).stem
    return transcripts.get(key, "")

def jaccard_similarity(text1, text2):
    if not text1 or not text2:
        return 0.0
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return 0.0
    return len(words1 & words2) / len(words1 | words2)

# ── Original (leaky) computation ──────────────────────────────────────────────

print("=" * 72)
print("  TASK 1 — DATA LEAKAGE CHECK")
print("=" * 72)

# Original: all real songs can appear in both positive and negative
positive_jaccard_leaky = []
negative_jaccard_leaky = []

fids = list(complete_pairs.keys())

for fid_a in fids:
    real_key = Path(complete_pairs[fid_a]['real']).stem
    real_text = transcripts.get(real_key, "")
    if not real_text:
        continue
    
    # Positive: real → its AI derivatives
    for ai_path in complete_pairs[fid_a]['ai']:
        ai_key = Path(ai_path).stem
        ai_text = transcripts.get(ai_key, "")
        if ai_text:
            sim = jaccard_similarity(real_text, ai_text)
            positive_jaccard_leaky.append(sim)
    
    # Negative: real → unrelated AI (other real songs' AI derivatives)
    other_fids = [f for f in fids if f != fid_a]
    np.random.seed(42 + int(fid_a))
    sampled = list(np.random.choice(other_fids, min(3, len(other_fids)), replace=False))
    
    for fid_b in sampled:
        ai_path = complete_pairs[fid_b]['ai'][0]
        ai_key = Path(ai_path).stem
        ai_text = transcripts.get(ai_key, "")
        if ai_text:
            sim = jaccard_similarity(real_text, ai_text)
            negative_jaccard_leaky.append(sim)

positive_jaccard_leaky = np.array(positive_jaccard_leaky)
negative_jaccard_leaky = np.array(negative_jaccard_leaky)

all_scores_leaky = np.concatenate([positive_jaccard_leaky, negative_jaccard_leaky])
all_labels_leaky = np.concatenate([np.ones(len(positive_jaccard_leaky)), np.zeros(len(negative_jaccard_leaky))])
auc_leaky = roc_auc_score(all_labels_leaky, all_scores_leaky)

print(f"\n  ── Original (leaky) ──")
print(f"  Positive pairs: {len(positive_jaccard_leaky)}")
print(f"  Negative pairs: {len(negative_jaccard_leaky)}")
print(f"  Positive mean:  {np.mean(positive_jaccard_leaky):.4f} ± {np.std(positive_jaccard_leaky):.4f}")
print(f"  Negative mean:  {np.mean(negative_jaccard_leaky):.4f} ± {np.std(negative_jaccard_leaky):.4f}")
print(f"  ROC AUC:        {auc_leaky:.4f}")

# ── Check for leakage ────────────────────────────────────────────────────────

print(f"\n  ── Leakage Analysis ──")

# Count how many real songs appear in both positive and negative
real_songs_in_pos = set()
real_songs_in_neg = set()

for fid_a in fids:
    real_key = Path(complete_pairs[fid_a]['real']).stem
    real_text = transcripts.get(real_key, "")
    if not real_text:
        continue
    
    # Check if this real song has any positive pairs
    has_pos = False
    for ai_path in complete_pairs[fid_a]['ai']:
        ai_key = Path(ai_path).stem
        if transcripts.get(ai_key, ""):
            has_pos = True
            break
    
    # Check if this real song has any negative pairs
    has_neg = False
    other_fids = [f for f in fids if f != fid_a]
    for fid_b in other_fids[:3]:  # Same sampling as original
        ai_path = complete_pairs[fid_b]['ai'][0]
        ai_key = Path(ai_path).stem
        if transcripts.get(ai_key, "") and real_text:
            has_neg = True
            break
    
    if has_pos:
        real_songs_in_pos.add(fid_a)
    if has_neg:
        real_songs_in_neg.add(fid_a)

overlap = real_songs_in_pos & real_songs_in_neg
print(f"  Real songs in positive pairs: {len(real_songs_in_pos)}")
print(f"  Real songs in negative pairs: {len(real_songs_in_neg)}")
print(f"  Real songs in BOTH (leakage): {len(overlap)}")

if overlap:
    print(f"  Leaked real song IDs: {sorted(overlap)}")
    print(f"  ⚠️  EVERY real song appears in both positive and negative pairs.")
    print(f"     The system can learn each real song's lyrics and use that to")
    print(f"     identify its AI derivative — this is CHEATING.")
else:
    print(f"  ✅ No leakage detected")

# ── Strict song-level split ──────────────────────────────────────────────────

print(f"\n  ── Strict Song-Level Split ──")

# Split real songs into two groups: POSITIVE group and NEGATIVE group
# Each real song appears in ONLY ONE group
fids_with_text = []
for fid in fids:
    real_key = Path(complete_pairs[fid]['real']).stem
    if transcripts.get(real_key, ""):
        fids_with_text.append(fid)

np.random.seed(42)
n_pos = len(fids_with_text) // 2
fids_for_pos = set(list(np.random.choice(fids_with_text, n_pos, replace=False)))
fids_for_neg = set(fids_with_text) - fids_for_pos

print(f"  Real songs for POSITIVE pairs only: {len(fids_for_pos)}")
print(f"  Real songs for NEGATIVE pairs only: {len(fids_for_neg)}")
print(f"  Overlap: {len(fids_for_pos & fids_for_neg)}")

# Positive pairs: only from fids_for_pos
positive_jaccard_strict = []
for fid_a in fids_for_pos:
    real_key = Path(complete_pairs[fid_a]['real']).stem
    real_text = transcripts.get(real_key, "")
    
    for ai_path in complete_pairs[fid_a]['ai']:
        ai_key = Path(ai_path).stem
        ai_text = transcripts.get(ai_key, "")
        if ai_text and real_text:
            sim = jaccard_similarity(real_text, ai_text)
            positive_jaccard_strict.append(sim)

# Negative pairs: use fids_for_neg real songs → fids_for_pos AI songs
# This ensures: real songs in negative pairs are DIFFERENT from real songs in positive pairs
negative_jaccard_strict = []
for fid_a in fids_for_neg:
    real_key = Path(complete_pairs[fid_a]['real']).stem
    real_text = transcripts.get(real_key, "")
    if not real_text:
        continue
    
    # Sample AI songs from the OTHER group (fids_for_pos)
    np.random.seed(42 + int(fid_a))
    sampled = list(np.random.choice(list(fids_for_pos), min(3, len(fids_for_pos)), replace=False))
    
    for fid_b in sampled:
        ai_path = complete_pairs[fid_b]['ai'][0]
        ai_key = Path(ai_path).stem
        ai_text = transcripts.get(ai_key, "")
        if ai_text:
            sim = jaccard_similarity(real_text, ai_text)
            negative_jaccard_strict.append(sim)

positive_jaccard_strict = np.array(positive_jaccard_strict)
negative_jaccard_strict = np.array(negative_jaccard_strict)

print(f"\n  Positive pairs: {len(positive_jaccard_strict)}")
print(f"  Negative pairs: {len(negative_jaccard_strict)}")
print(f"  Positive mean:  {np.mean(positive_jaccard_strict):.4f} ± {np.std(positive_jaccard_strict):.4f}")
print(f"  Negative mean:  {np.mean(negative_jaccard_strict):.4f} ± {np.std(negative_jaccard_strict):.4f}")
print(f"  Gap:            {np.mean(positive_jaccard_strict) - np.mean(negative_jaccard_strict):+.4f}")

all_scores_strict = np.concatenate([positive_jaccard_strict, negative_jaccard_strict])
all_labels_strict = np.concatenate([np.ones(len(positive_jaccard_strict)), np.zeros(len(negative_jaccard_strict))])
auc_strict = roc_auc_score(all_labels_strict, all_scores_strict)

print(f"\n  ROC AUC (strict): {auc_strict:.4f}")
print(f"  ROC AUC (leaky):  {auc_leaky:.4f}")
print(f"  Drop:             {auc_leaky - auc_strict:+.4f}")

if auc_leaky - auc_strict > 0.05:
    print(f"\n  ⚠️  AUC dropped by {auc_leaky - auc_strict:.4f} — leakage was inflating the result.")
    print(f"  The {auc_strict:.4f} number is the honest one.")
elif auc_leaky - auc_strict > 0.02:
    print(f"\n  ⚠️  Minor AUC drop ({auc_leaky - auc_strict:.4f}) — some inflation from leakage.")
else:
    print(f"\n  ✅ Minimal AUC impact from leakage ({auc_leaky - auc_strict:+.4f}).")

# ── Save corrected results ───────────────────────────────────────────────────

output = {
    'leaky': {
        'n_positive': int(len(positive_jaccard_leaky)),
        'n_negative': int(len(negative_jaccard_leaky)),
        'positive_mean': round(float(np.mean(positive_jaccard_leaky)), 4),
        'negative_mean': round(float(np.mean(negative_jaccard_leaky)), 4),
        'roc_auc': round(float(auc_leaky), 4),
        'leaked_songs': int(len(overlap)),
    },
    'strict': {
        'n_positive': int(len(positive_jaccard_strict)),
        'n_negative': int(len(negative_jaccard_strict)),
        'positive_mean': round(float(np.mean(positive_jaccard_strict)), 4),
        'negative_mean': round(float(np.mean(negative_jaccard_strict)), 4),
        'roc_auc': round(float(auc_strict), 4),
        'n_pos_songs': len(fids_for_pos),
        'n_neg_songs': len(fids_for_neg),
    },
    'verdict': 'leakage' if len(overlap) > 0 else 'clean',
}

out_path = PAIRS_DIR.parent / "lyrics_task1_leakage_check.json"
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\n  Results saved → {out_path}")

# ── Plot ──────────────────────────────────────────────────────────────────────

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Task 1 — Data Leakage Check", fontsize=14, fontweight="bold")

# Panel 1: Leaky
ax = axes[0]
ax.hist(positive_jaccard_leaky, bins=15, alpha=0.5, label=f"Positive (n={len(positive_jaccard_leaky)})", color="#27AE60", density=True)
ax.hist(negative_jaccard_leaky, bins=15, alpha=0.5, label=f"Negative (n={len(negative_jaccard_leaky)})", color="#E74C3C", density=True)
ax.axvline(np.mean(positive_jaccard_leaky), color="#27AE60", ls="--", lw=1.5)
ax.axvline(np.mean(negative_jaccard_leaky), color="#E74C3C", ls="--", lw=1.5)
ax.set_xlabel("Jaccard Similarity")
ax.set_ylabel("Density")
ax.set_title(f"Leaky Split (AUC = {auc_leaky:.4f})\n{len(overlap)} real songs in both groups")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 2: Strict
ax = axes[1]
ax.hist(positive_jaccard_strict, bins=15, alpha=0.5, label=f"Positive (n={len(positive_jaccard_strict)})", color="#27AE60", density=True)
ax.hist(negative_jaccard_strict, bins=15, alpha=0.5, label=f"Negative (n={len(negative_jaccard_strict)})", color="#E74C3C", density=True)
ax.axvline(np.mean(positive_jaccard_strict), color="#27AE60", ls="--", lw=1.5)
ax.axvline(np.mean(negative_jaccard_strict), color="#E74C3C", ls="--", lw=1.5)
ax.set_xlabel("Jaccard Similarity")
ax.set_ylabel("Density")
ax.set_title(f"Strict Song-Level Split (AUC = {auc_strict:.4f})\nNo real song in both groups")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_png = PAIRS_DIR.parent / "lyrics_task1_leakage_check.png"
plt.savefig(str(out_png), dpi=150, bbox_inches="tight")
print(f"  Plot saved → {out_png}")
