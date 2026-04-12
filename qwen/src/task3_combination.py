#!/usr/bin/env python3
"""
Task 3 — 2D Scatter and Combination Testing
=============================================
After Tasks 1 and 2 confirmed the corrected AUC.
Scatter plot + 5 combination rules on strict splits.
"""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

PAIRS_DIR = Path("/home/nikos_pc/Desktop/Orfium/orfium_audio_attribution_task/qwen/results/real_ai_pairs")
RESULTS_DIR = PAIRS_DIR.parent

# Load transcripts
with open(PAIRS_DIR / "transcripts.json") as f:
    transcripts = json.load(f)

# Load mapping
with open(PAIRS_DIR / "real_ai_pairs_mapping.json") as f:
    pairs = json.load(f)

def get_real_path(fid):
    p = PAIRS_DIR / f"real_{fid}.mp3"
    if p.exists():
        return str(p)
    return None

def get_ai_paths(fid):
    return sorted(PAIRS_DIR.glob(f"ai_{fid}_*.mp3"))

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

def jaccard_similarity(text1, text2):
    if not text1 or not text2:
        return 0.0
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return 0.0
    return len(words1 & words2) / len(words1 | words2)

def cosine(a, b):
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))

# ── Build strict splits ──────────────────────────────────────────────────────

fids = [fid for fid in complete_pairs.keys() 
        if transcripts.get(Path(complete_pairs[fid]['real']).stem, "")]

np.random.seed(42)
n_pos = len(fids) // 2
fids_for_pos = set(list(np.random.choice(fids, n_pos, replace=False)))
fids_for_neg = set(fids) - fids_for_pos

print("=" * 72)
print("  TASK 3 — 2D SCATTER AND COMBINATION TESTING")
print("=" * 72)
print(f"  Positive songs: {len(fids_for_pos)}")
print(f"  Negative songs: {len(fids_for_neg)}")
print(f"  Strict split (no overlap): {len(fids_for_pos & fids_for_neg)}")

# ── Compute MERT and Jaccard for all pairs ────────────────────────────────────

print("\n  ── Computing MERT and Jaccard scores ──")

# Load cached MERT embeddings
def get_cached_mert(path):
    cached = PAIRS_DIR / f"{Path(path).stem}_mert.npy"
    if cached.exists():
        return np.load(cached)
    return None

positive_data = []  # (mert_score, jaccard_score)
negative_data = []

# Positive pairs: fids_for_pos real → their AI derivatives
for fid_a in fids_for_pos:
    real_path = complete_pairs[fid_a]['real']
    real_key = Path(real_path).stem
    real_text = transcripts.get(real_key, "")
    real_emb = get_cached_mert(real_path)
    if not real_text or real_emb is None:
        continue
    
    for ai_path in complete_pairs[fid_a]['ai']:
        ai_key = Path(ai_path).stem
        ai_text = transcripts.get(ai_key, "")
        ai_emb = get_cached_mert(ai_path)
        
        if ai_text and ai_emb is not None:
            jaccard = jaccard_similarity(real_text, ai_text)
            mert_sim = cosine(real_emb, ai_emb)
            positive_data.append({
                'real_fid': fid_a,
                'real_title': complete_pairs[fid_a]['title'],
                'ai_file': ai_key,
                'jaccard': jaccard,
                'mert': mert_sim,
                'label': 1,
            })

# Negative pairs: fids_for_neg real → fids_for_pos AI
for fid_a in fids_for_neg:
    real_path = complete_pairs[fid_a]['real']
    real_key = Path(real_path).stem
    real_text = transcripts.get(real_key, "")
    real_emb = get_cached_mert(real_path)
    if not real_text or real_emb is None:
        continue
    
    np.random.seed(42 + int(fid_a))
    sampled = list(np.random.choice(list(fids_for_pos), min(3, len(fids_for_pos)), replace=False))
    
    for fid_b in sampled:
        ai_path = complete_pairs[fid_b]['ai'][0]
        ai_key = Path(ai_path).stem
        ai_text = transcripts.get(ai_key, "")
        ai_emb = get_cached_mert(ai_path)
        
        if ai_text and ai_emb is not None:
            jaccard = jaccard_similarity(real_text, ai_text)
            mert_sim = cosine(real_emb, ai_emb)
            negative_data.append({
                'real_fid': fid_a,
                'ai_fid': fid_b,
                'real_title': complete_pairs[fid_a]['title'],
                'ai_file': ai_key,
                'jaccard': jaccard,
                'mert': mert_sim,
                'label': 0,
            })

all_data = positive_data + negative_data
print(f"  Positive pairs: {len(positive_data)}")
print(f"  Negative pairs: {len(negative_data)}")
print(f"  Total: {len(all_data)}")

# ── Test combination rules ───────────────────────────────────────────────────

print(f"\n  ── Combination Rules ──")

# Extract arrays
jaccard_all = np.array([d['jaccard'] for d in all_data])
mert_all = np.array([d['mert'] for d in all_data])
labels = np.array([d['label'] for d in all_data])

# (a) Jaccard alone
auc_jaccard = roc_auc_score(labels, jaccard_all)
print(f"  (a) Jaccard alone:           AUC = {auc_jaccard:.4f}")

# (b) MERT alone
auc_mert = roc_auc_score(labels, mert_all)
print(f"  (b) MERT alone:              AUC = {auc_mert:.4f}")

# (c) MERT gate: score = 0 if MERT < 0.88, else Jaccard
mert_threshold = 0.88
gate_score = np.where(mert_all < mert_threshold, 0.0, jaccard_all)
auc_gate = roc_auc_score(labels, gate_score)
print(f"  (c) MERT gate (<{mert_threshold}):   AUC = {auc_gate:.4f}")

# (d) Product: MERT × Jaccard
product_score = mert_all * jaccard_all
auc_product = roc_auc_score(labels, product_score)
print(f"  (d) MERT × Jaccard:          AUC = {auc_product:.4f}")

# (e) Logistic regression with 5-fold song-level CV
# Song-level: group by real_fid, split songs not pairs
real_fids = np.array([d.get('real_fid', None) for d in all_data])
unique_real_fids = np.unique(real_fids)
np.random.seed(42)
np.random.shuffle(unique_real_fids)

n_folds = 5
fold_size = len(unique_real_fids) // n_folds
folds = [unique_real_fids[i*fold_size:(i+1)*fold_size] for i in range(n_folds)]
# Handle remainder
for i in range(len(unique_real_fids) - n_folds * fold_size):
    folds[i].append(unique_real_fids[-(i+1)])

X = np.column_stack([mert_all, jaccard_all])
fold_aucs = []

for i, test_fids in enumerate(folds):
    train_mask = ~np.isin(real_fids, test_fids)
    test_mask = np.isin(real_fids, test_fids)
    
    X_train, y_train = X[train_mask], labels[train_mask]
    X_test, y_test = X[test_mask], labels[test_mask]
    
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        continue
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    probas = model.predict_proba(X_test)[:, 1]
    fold_auc = roc_auc_score(y_test, probas)
    fold_aucs.append(fold_auc)
    print(f"  (e) Fold {i+1}: AUC = {fold_auc:.4f} (n_test={len(y_test)})")

auc_lr = np.mean(fold_aucs)
print(f"  (e) Logistic regression 5-CV:  AUC = {auc_lr:.4f} ± {np.std(fold_aucs):.4f}")

# ── Summary ──────────────────────────────────────────────────────────────────

print(f"\n  ── SUMMARY ──")
methods = [
    ('Jaccard alone', auc_jaccard),
    ('MERT alone', auc_mert),
    (f'MERT gate (<{mert_threshold})', auc_gate),
    ('MERT × Jaccard', auc_product),
    ('Logistic regression (5-CV)', auc_lr),
]

methods.sort(key=lambda x: x[1], reverse=True)
for name, auc in methods:
    marker = "✅" if auc > 0.85 else "⚠️" if auc > 0.7 else "❌"
    print(f"  {marker} {name:<35s}  {auc:.4f}")

best_name, best_auc = methods[0]
print(f"\n  Best: {best_name} (AUC = {best_auc:.4f})")

# ── Save results ──────────────────────────────────────────────────────────────

output = {
    'n_positive': len(positive_data),
    'n_negative': len(negative_data),
    'auc_jaccard': round(float(auc_jaccard), 4),
    'auc_mert': round(float(auc_mert), 4),
    'auc_gate': round(float(auc_gate), 4),
    'auc_product': round(float(auc_product), 4),
    'auc_logistic_cv': round(float(auc_lr), 4),
    'logistic_cv_folds': [round(float(x), 4) for x in fold_aucs],
    'best_method': best_name,
    'best_auc': round(float(best_auc), 4),
    'all_data': [
        {
            'real_fid': d.get('real_fid'),
            'ai_fid': d.get('ai_fid'),
            'real_title': d['real_title'],
            'ai_file': d['ai_file'],
            'jaccard': round(d['jaccard'], 4),
            'mert': round(d['mert'], 4),
            'label': d['label'],
        }
        for d in all_data
    ],
}

out_path = RESULTS_DIR / "lyrics_task3_combination.json"
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\n  Results saved → {out_path}")

# ── Plot ──────────────────────────────────────────────────────────────────────

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Task 3 — MERT vs Lyrics: 2D Analysis", fontsize=14, fontweight="bold")

# Panel 1: Scatter plot
ax = axes[0, 0]
pos_data = [d for d in all_data if d['label'] == 1]
neg_data = [d for d in all_data if d['label'] == 0]

ax.scatter([d['mert'] for d in pos_data], [d['jaccard'] for d in pos_data],
           c='#27AE60', s=30, alpha=0.6, label=f"Derivative (n={len(pos_data)})")
ax.scatter([d['mert'] for d in neg_data], [d['jaccard'] for d in neg_data],
           c='#E74C3C', s=30, alpha=0.6, label=f"Unrelated (n={len(neg_data)})")

ax.axvline(0.88, color="gray", ls="--", lw=1, alpha=0.7, label="MERT gate threshold")
ax.axhline(0.15, color="gray", ls="--", lw=1, alpha=0.7)
ax.set_xlabel("MERT Cosine Similarity")
ax.set_ylabel("Jaccard Similarity (lyrics)")
ax.set_title(f"MERT vs Lyrics\n(AUC: MERT={auc_mert:.3f}, Jaccard={auc_jaccard:.3f})")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 2: AUC comparison
ax = axes[0, 1]
names = [m[0] for m in methods]
aucs = [m[1] for m in methods]
colors = ["#3498DB", "#E74C3C", "#F39C12", "#9B59B6", "#2ECC71"]
bars = ax.barh(names, aucs, color=colors, alpha=0.7)
ax.axvline(0.5, color="gray", ls="--", lw=1, alpha=0.7, label="Random (0.5)")
for bar, auc in zip(bars, aucs):
    ax.text(auc + 0.01, bar.get_y() + bar.get_height()/2, f"{auc:.4f}",
            ha='left', va='center', fontweight='bold', fontsize=9)
ax.set_xlabel("ROC AUC")
ax.set_title("Combination Rules")
ax.set_xlim(0.4, 1.0)
ax.grid(True, axis="x", alpha=0.3)

# Panel 3: Zoom on false positives (high Jaccard, low label)
ax = axes[1, 0]
# Negative pairs with highest Jaccard
neg_sorted = sorted(neg_data, key=lambda x: x['jaccard'], reverse=True)[:20]
ax.scatter([d['mert'] for d in neg_sorted], [d['jaccard'] for d in neg_sorted],
           c='#E74C3C', s=50, alpha=0.7)
for d in neg_sorted[:10]:
    ax.annotate(d['real_title'][:15], (d['mert'], d['jaccard']),
                fontsize=6, alpha=0.7, ha='left')
ax.scatter([d['mert'] for d in pos_data], [d['jaccard'] for d in pos_data],
           c='#27AE60', s=15, alpha=0.3)
ax.set_xlabel("MERT Cosine Similarity")
ax.set_ylabel("Jaccard Similarity")
ax.set_title("Top 20 False Positives (unrelated, high lyrics)")
ax.grid(True, alpha=0.3)

# Panel 4: Zoom on false negatives (low Jaccard, positive label)
ax = axes[1, 1]
pos_sorted = sorted(pos_data, key=lambda x: x['jaccard'])[:20]
ax.scatter([d['mert'] for d in pos_sorted], [d['jaccard'] for d in pos_sorted],
           c='#27AE60', s=50, alpha=0.7)
for d in pos_sorted[:10]:
    ax.annotate(d['real_title'][:15], (d['mert'], d['jaccard']),
                fontsize=6, alpha=0.7, ha='left')
ax.scatter([d['mert'] for d in neg_data], [d['jaccard'] for d in neg_data],
           c='#E74C3C', s=15, alpha=0.3)
ax.set_xlabel("MERT Cosine Similarity")
ax.set_ylabel("Jaccard Similarity")
ax.set_title("Top 20 False Negatives (derivative, low lyrics)")
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_png = RESULTS_DIR / "lyrics_task3_scatter.png"
plt.savefig(str(out_png), dpi=150, bbox_inches="tight")
print(f"  Plot saved → {out_png}")
