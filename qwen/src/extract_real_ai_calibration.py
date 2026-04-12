#!/usr/bin/env python3
"""
Extract MERT embeddings and compute real→AI similarities for all 64 complete pairs.
Fits proper GMM calibration and reports actual AUC.
"""

import json
import time
from pathlib import Path

import numpy as np
import librosa
import torch
from transformers import AutoProcessor, AutoModel
from sklearn.metrics import roc_auc_score

# ── Config ─────────────────────────────────────────────────────────────────────
PAIRS_DIR = Path("/home/nikos_pc/Desktop/Orfium/orfium_audio_attribution_task/qwen/results/real_ai_pairs")
RESULTS_DIR = PAIRS_DIR.parent

MERT_MODEL = "m-a-p/MERT-v1-95M"
MERT_SR = 24_000
WINDOW_SEC = 10
HOP_SEC = 5
MIN_SEG_SEC = 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ── Load mapping ──────────────────────────────────────────────────────────────
mapping_path = PAIRS_DIR / "real_ai_pairs_mapping.json"
with open(mapping_path) as f:
    pairs = json.load(f)

# Identify complete pairs
def get_real_path(fid):
    return PAIRS_DIR / f"real_{fid}.mp3"

def get_ai_paths(fid):
    return sorted(PAIRS_DIR.glob(f"ai_{fid}_*.mp3"))

complete_pairs = {}
for fid, info in pairs.items():
    real_path = get_real_path(fid)
    ai_paths = get_ai_paths(fid)
    if real_path.exists() and len(ai_paths) > 0:
        complete_pairs[fid] = {
            'real': str(real_path),
            'ai': [str(p) for p in ai_paths],
            'title': info['real_title'],
            'artist': info['real_artist'],
            'algorithm': info['algorithm'],
        }

print(f"Complete pairs: {len(complete_pairs)}")

# ── MERT extraction ───────────────────────────────────────────────────────────

def segment_audio(path):
    y, _ = librosa.load(str(path), sr=MERT_SR, mono=True)
    win = int(WINDOW_SEC * MERT_SR)
    hop = int(HOP_SEC * MERT_SR)
    min_len = int(MIN_SEG_SEC * MERT_SR)
    segs = []
    start = 0
    while start < len(y):
        chunk = y[start:start + win]
        if len(chunk) < min_len:
            break
        if len(chunk) < win:
            chunk = np.pad(chunk, (0, win - len(chunk)))
        segs.append(chunk.astype(np.float32))
        start += hop
    return segs

def extract_embedding(segments, processor, model, device):
    model.eval()
    seg_embs = []
    with torch.no_grad():
        for seg in segments:
            inputs = processor(seg, sampling_rate=MERT_SR, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out = model(**inputs)
            emb = out.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
            seg_embs.append(emb.astype(np.float32))
    return np.mean(seg_embs, axis=0), np.array(seg_embs)

def cosine(a, b):
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))

# Load MERT
print("\n── Loading MERT model ──")
processor = AutoProcessor.from_pretrained(MERT_MODEL, trust_remote_code=True)
model = AutoModel.from_pretrained(MERT_MODEL, trust_remote_code=True).to(DEVICE)

# Extract all embeddings
print("\n── Extracting MERT embeddings ──")
embeddings = {}  # path → (file_emb, seg_embs)
all_files = []

for fid, info in complete_pairs.items():
    all_files.append((f"real_{fid}", info['real']))
    for ai_path in info['ai']:
        name = Path(ai_path).stem
        all_files.append((name, ai_path))

for i, (name, path) in enumerate(all_files):
    cached = PAIRS_DIR / f"{name}_mert.npy"
    cached_seg = PAIRS_DIR / f"{name}_mert_segs.npy"
    if cached.exists() and cached_seg.exists():
        emb = np.load(cached)
        seg_emb = np.load(cached_seg)
        embeddings[name] = {'file': emb, 'segments': seg_emb}
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(all_files)}] [cached] {name}")
    else:
        t0 = time.time()
        segments = segment_audio(path)
        emb, seg_emb = extract_embedding(segments, processor, model, DEVICE)
        embeddings[name] = {'file': emb, 'segments': seg_emb}
        np.save(cached, emb)
        np.save(cached_seg, seg_emb)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(all_files)}] [{time.time()-t0:.1f}s] {name} ({len(segments)} segs)")

# ── Compute similarities ──────────────────────────────────────────────────────
print("\n── Computing similarities ──")

# Positive: real → AI (same source)
positive_scores = []
pair_details = []
for fid, info in complete_pairs.items():
    real_name = f"real_{fid}"
    if real_name not in embeddings:
        continue
    real_emb = embeddings[real_name]['file']
    
    for ai_path in info['ai']:
        ai_name = Path(ai_path).stem
        if ai_name not in embeddings:
            continue
        ai_emb = embeddings[ai_name]['file']
        sim = cosine(real_emb, ai_emb)
        positive_scores.append(sim)
        pair_details.append({
            'fid': fid,
            'title': info['title'],
            'artist': info['artist'],
            'algorithm': info['algorithm'],
            'ai_file': ai_name,
            'similarity': round(sim, 5),
        })

positive_scores = np.array(positive_scores)
print(f"Positive pairs (real→AI derivative): {len(positive_scores)}")
print(f"  mean={np.mean(positive_scores):.5f}  std={np.std(positive_scores):.5f}")
print(f"  min={np.min(positive_scores):.5f}  max={np.max(positive_scores):.5f}")

# Negative: real → AI (different source)
negative_scores = []
fids = list(complete_pairs.keys())
for i, fid_a in enumerate(fids):
    real_name = f"real_{fid_a}"
    if real_name not in embeddings:
        continue
    real_emb = embeddings[real_name]['file']
    
    # Pick 2-3 random different-source AI songs
    other_fids = [f for f in fids if f != fid_a]
    np.random.seed(42 + fid_a)
    sampled = list(np.random.choice(other_fids, min(3, len(other_fids)), replace=False))
    
    for fid_b in sampled:
        ai_paths = complete_pairs[fid_b]['ai']
        if ai_paths:
            ai_name = Path(ai_paths[0]).stem
            if ai_name in embeddings:
                ai_emb = embeddings[ai_name]['file']
                sim = cosine(real_emb, ai_emb)
                negative_scores.append(sim)

negative_scores = np.array(negative_scores)
print(f"\nNegative pairs (real→AI unrelated): {len(negative_scores)}")
print(f"  mean={np.mean(negative_scores):.5f}  std={np.std(negative_scores):.5f}")
print(f"  min={np.min(negative_scores):.5f}  max={np.max(negative_scores):.5f}")

# ── Fit GMM calibration ──────────────────────────────────────────────────────
from scipy.stats import norm

mu_pos = float(np.mean(positive_scores))
sigma_pos = float(np.std(positive_scores))
mu_neg = float(np.mean(negative_scores))
sigma_neg = float(np.std(negative_scores))

print(f"\n── GMM Calibration Parameters ──")
print(f"  Positive (derivative):  μ={mu_pos:.5f}  σ={sigma_pos:.5f}")
print(f"  Negative (unrelated):   μ={mu_neg:.5f}  σ={sigma_neg:.5f}")
print(f"  Gap (pos - neg):        {mu_pos - mu_neg:+.5f}")

# Compute AUC
all_scores = np.concatenate([positive_scores, negative_scores])
all_labels = np.concatenate([np.ones(len(positive_scores)), np.zeros(len(negative_scores))])
auc = roc_auc_score(all_labels, all_scores)
print(f"\n  ROC AUC: {auc:.4f}")

# ── Calibrated score examples ─────────────────────────────────────────────────
print(f"\n── Calibrated Score Examples ──")
for raw in [mu_pos, mu_neg, (mu_pos + mu_neg) / 2]:
    like_pos = norm.pdf(raw, mu_pos, sigma_pos)
    like_neg = norm.pdf(raw, mu_neg, sigma_neg)
    cal = like_pos / (like_pos + like_neg + 1e-12)
    print(f"  Raw {raw:.5f} → P(derivative) = {cal:.4f}")

# ── Per-pair breakdown ────────────────────────────────────────────────────────
print(f"\n── Per-Pair Breakdown ──")
print(f"  {'Title':<35s}  {'Alg':<15s}  {'Variant':<10s}  {'Sim':>8s}")
for d in sorted(pair_details, key=lambda x: x['similarity']):
    variant = d['ai_file'].split('_')[-1]  # 0 or 1
    print(f"  {d['title'][:35]:<35s}  {d['algorithm']:<15s}  {variant:<10s}  {d['similarity']:.5f}")

# ── Save results ──────────────────────────────────────────────────────────────
output = {
    'n_pairs': len(complete_pairs),
    'n_positive': int(len(positive_scores)),
    'n_negative': int(len(negative_scores)),
    'positive_mean': round(float(mu_pos), 5),
    'positive_std': round(float(sigma_pos), 5),
    'negative_mean': round(float(mu_neg), 5),
    'negative_std': round(float(sigma_neg), 5),
    'gap': round(float(mu_pos - mu_neg), 5),
    'roc_auc': round(float(auc), 4),
    'gmm_params': {
        'mu_pos': round(mu_pos, 5),
        'sigma_pos': round(sigma_pos, 5),
        'mu_neg': round(mu_neg, 5),
        'sigma_neg': round(sigma_neg, 5),
    },
    'pair_details': pair_details,
}

out_path = RESULTS_DIR / "real_ai_calibration.json"
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved → {out_path}")

# ── Plot ──────────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Real→AI Attribution: Ground Truth Calibration", fontsize=14, fontweight="bold")

# Panel 1: Distributions
ax = axes[0]
ax.hist(positive_scores, bins=15, alpha=0.5, label=f"Derivative (n={len(positive_scores)})", color="#27AE60", density=True)
ax.hist(negative_scores, bins=15, alpha=0.5, label=f"Unrelated (n={len(negative_scores)})", color="#E74C3C", density=True)
ax.axvline(mu_pos, color="#27AE60", ls="--", lw=1.5)
ax.axvline(mu_neg, color="#E74C3C", ls="--", lw=1.5)
ax.set_xlabel("MERT Cosine Similarity")
ax.set_ylabel("Density")
ax.set_title(f"Distributions (AUC = {auc:.4f})")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 2: Per-pair similarities
ax = axes[1]
by_pair = {}
for d in pair_details:
    key = f"{d['fid']}_{d['title'][:25]}"
    by_pair.setdefault(key, []).append(d['similarity'])

pairs_sorted = sorted(by_pair.items(), key=lambda x: np.mean(x[1]))
means = [np.mean(v) for k, v in pairs_sorted]
mins = [min(v) for k, v in pairs_sorted]
maxs = [max(v) for k, v in pairs_sorted]
labels = [k[:30] for k, v in pairs_sorted]

y = np.arange(len(means))
ax.errorbar(means, y, xerr=[np.array(means) - np.array(mins), np.array(maxs) - np.array(means)],
            fmt='o', color='#3498DB', alpha=0.7, capsize=3)
ax.axvline(mu_neg, color="#E74C3C", ls="--", lw=1, alpha=0.7, label="Unrelated mean")
ax.axvline(mu_pos, color="#27AE60", ls="--", lw=1, alpha=0.7, label="Derivative mean")
ax.set_xlabel("Cosine Similarity")
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=6)
ax.set_title("Per-Pair Similarities (±range)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_png = RESULTS_DIR / "real_ai_calibration.png"
plt.savefig(str(out_png), dpi=150, bbox_inches="tight")
print(f"Plot saved → {out_png}")
