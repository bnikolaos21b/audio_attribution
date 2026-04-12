#!/usr/bin/env python3
"""
Lyrics-Based Attribution Experiment
======================================
Tests whether lyrics analysis can solve real→AI attribution that MERT fails at.

Steps:
1. Transcribe all 64 real songs + 149 AI songs with Whisper-tiny
2. Compute Jaccard similarity (word overlap) for positive and negative pairs
3. Compute sentence-transformer cosine similarity for positive and negative pairs
4. Report ROC AUC for both methods
"""

import json
import os
import sys
import time
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

# ── Paths ─────────────────────────────────────────────────────────────────────
PAIRS_DIR = Path("/home/nikos_pc/Desktop/Orfium/orfium_audio_attribution_task/qwen/results/real_ai_pairs")
RESULTS_DIR = PAIRS_DIR.parent

# Load mapping
mapping_path = PAIRS_DIR / "real_ai_pairs_mapping.json"
with open(mapping_path) as f:
    pairs = json.load(f)

# ── Step 1: Whisper transcription ────────────────────────────────────────────

def get_transcript(audio_path):
    """Transcribe audio using whisper-tiny."""
    import whisper
    model = whisper.load_model("tiny")
    result = model.transcribe(str(audio_path), language="en")
    return result["text"].strip()

# Check what audio files we have
def get_real_path(fid):
    p = PAIRS_DIR / f"real_{fid}.mp3"
    if p.exists():
        return p
    # Try the YouTube-ID named file
    for f in PAIRS_DIR.glob("real_*.mp3"):
        if f.stem == f"real_{fid}":
            return f
    return None

def get_ai_paths(fid):
    return sorted(PAIRS_DIR.glob(f"ai_{fid}_*.mp3"))

# Find complete pairs
complete_pairs = {}
for fid, info in pairs.items():
    real_path = get_real_path(fid)
    ai_paths = get_ai_paths(fid)
    if real_path and len(ai_paths) > 0:
        complete_pairs[int(fid)] = {
            'real': str(real_path),
            'ai': [str(p) for p in ai_paths],
            'title': info['real_title'],
            'artist': info['real_artist'],
            'algorithm': info['algorithm'],
        }

print(f"Complete pairs available: {len(complete_pairs)}")

# Load cached transcripts or transcribe
TRANSCRIPTS_CACHE = PAIRS_DIR / "transcripts.json"
if TRANSCRIPTS_CACHE.exists():
    with open(TRANSCRIPTS_CACHE) as f:
        transcripts = json.load(f)
    print(f"Loaded {len(transcripts)} cached transcripts")
else:
    transcripts = {}

# Transcribe all files
all_files = []
for fid, info in complete_pairs.items():
    if info['real'] not in transcripts:
        all_files.append(('real', fid, info['real']))
    for ai_path in info['ai']:
        name = Path(ai_path).stem
        if name not in transcripts:
            all_files.append(('ai', fid, ai_path))

print(f"Files to transcribe: {len(all_files)}")
print(f"Files already cached: {len(transcripts)}")

if all_files:
    import whisper
    print("\n── Loading Whisper-tiny ──")
    model = whisper.load_model("tiny")
    
    for i, (ftype, fid, path) in enumerate(all_files):
        key = Path(path).stem
        if key in transcripts:
            print(f"  [{i+1}/{len(all_files)}] [cached] {key}")
            continue
        
        t0 = time.time()
        try:
            result = model.transcribe(path, language="en")
            text = result["text"].strip()
            transcripts[key] = text
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(all_files)}] [{elapsed:.0f}s] {key} ({len(text)} chars)")
        except Exception as e:
            print(f"  [{i+1}/{len(all_files)}] [ERROR] {key}: {e}")
            transcripts[key] = ""
        
        # Save after each file
        with open(TRANSCRIPTS_CACHE, 'w') as f:
            json.dump(transcripts, f, indent=2)

print(f"\nTotal transcripts: {len(transcripts)}")

# ── Step 2: Compute text similarities ────────────────────────────────────────

print("\n── Computing Jaccard similarity ──")

def jaccard_similarity(text1, text2):
    """Word-level Jaccard similarity."""
    if not text1 or not text2:
        return 0.0
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return 0.0
    intersection = words1 & words2
    union = words1 | words2
    return len(intersection) / len(union)

# Positive pairs: real → AI (same source)
positive_jaccard = []
positive_details = []
for fid, info in complete_pairs.items():
    real_key = Path(info['real']).stem
    real_text = transcripts.get(real_key, "")
    
    for ai_path in info['ai']:
        ai_key = Path(ai_path).stem
        ai_text = transcripts.get(ai_key, "")
        
        if real_text and ai_text:
            sim = jaccard_similarity(real_text, ai_text)
            positive_jaccard.append(sim)
            positive_details.append({
                'fid': fid,
                'title': info['title'],
                'artist': info['artist'],
                'ai_file': ai_key,
                'jaccard': round(sim, 5),
            })

positive_jaccard = np.array(positive_jaccard)
print(f"Positive Jaccard: n={len(positive_jaccard)}, mean={np.mean(positive_jaccard):.4f}, "
      f"std={np.std(positive_jaccard):.4f}, range=[{np.min(positive_jaccard):.4f}, {np.max(positive_jaccard):.4f}]")

# Negative pairs: real → AI (different source)
negative_jaccard = []
fids = list(complete_pairs.keys())
np.random.seed(42)

for fid_a in fids:
    real_key = Path(complete_pairs[fid_a]['real']).stem
    real_text = transcripts.get(real_key, "")
    if not real_text:
        continue
    
    # Sample 2-3 unrelated AI songs
    other_fids = [f for f in fids if f != fid_a]
    sampled = list(np.random.choice(other_fids, min(3, len(other_fids)), replace=False))
    
    for fid_b in sampled:
        ai_path = complete_pairs[fid_b]['ai'][0]
        ai_key = Path(ai_path).stem
        ai_text = transcripts.get(ai_key, "")
        
        if ai_text:
            sim = jaccard_similarity(real_text, ai_text)
            negative_jaccard.append(sim)

negative_jaccard = np.array(negative_jaccard)
print(f"Negative Jaccard: n={len(negative_jaccard)}, mean={np.mean(negative_jaccard):.4f}, "
      f"std={np.std(negative_jaccard):.4f}, range=[{np.min(negative_jaccard):.4f}, {np.max(negative_jaccard):.4f}]")

# ── Step 3: Sentence-transformer embeddings ──────────────────────────────────

print("\n── Loading sentence-transformer ──")

def get_sentence_embedding(text):
    """Get embedding using a lightweight sentence transformer."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode([text])[0]

# Check if we need to compute embeddings
EMBEDDINGS_CACHE = PAIRS_DIR / "sentence_embeddings.json"
sentence_embeddings = {}

if EMBEDDINGS_CACHE.exists():
    print("  Loading cached embeddings...")
    with open(EMBEDDINGS_CACHE) as f:
        data = json.load(f)
    # Convert lists back to numpy arrays
    for key, emb in data.items():
        sentence_embeddings[key] = np.array(emb)
    print(f"  Loaded {len(sentence_embeddings)} cached embeddings")
else:
    print("  Loading model...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Get unique texts to embed
    unique_texts = {}
    for key, text in transcripts.items():
        if text and text not in unique_texts.values():
            unique_texts[key] = text
    
    print(f"  Embedding {len(unique_texts)} unique texts...")
    
    keys = list(unique_texts.keys())
    texts = list(unique_texts.values())
    embeddings = model.encode(texts, show_progress_bar=True)
    
    for key, emb in zip(keys, embeddings):
        sentence_embeddings[key] = emb
    
    # Save as lists for JSON
    save_data = {k: v.tolist() for k, v in sentence_embeddings.items()}
    with open(EMBEDDINGS_CACHE, 'w') as f:
        json.dump(save_data, f)
    print(f"  Saved {len(sentence_embeddings)} embeddings")

def cosine_similarity(a, b):
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))

# Compute sentence-transformer similarities
positive_sent = []
negative_sent = []

for d in positive_details:
    fid = d['fid']
    real_key = Path(complete_pairs[fid]['real']).stem
    ai_key = d['ai_file']
    
    real_emb = sentence_embeddings.get(real_key)
    ai_emb = sentence_embeddings.get(ai_key)
    
    if real_emb is not None and ai_emb is not None:
        sim = cosine_similarity(real_emb, ai_emb)
        positive_sent.append(sim)
        d['sentence_sim'] = round(sim, 5)

positive_sent = np.array(positive_sent)
print(f"\nPositive sentence-transformer: n={len(positive_sent)}, mean={np.mean(positive_sent):.4f}, "
      f"std={np.std(positive_sent):.4f}, range=[{np.min(positive_sent):.4f}, {np.max(positive_sent):.4f}]")

# Negative sentence-transformer
for fid_a in fids:
    real_key = Path(complete_pairs[fid_a]['real']).stem
    real_emb = sentence_embeddings.get(real_key)
    if real_emb is None:
        continue
    
    other_fids = [f for f in fids if f != fid_a]
    np.random.seed(42 + int(fid_a))
    sampled = list(np.random.choice(other_fids, min(3, len(other_fids)), replace=False))
    
    for fid_b in sampled:
        ai_path = complete_pairs[fid_b]['ai'][0]
        ai_key = Path(ai_path).stem
        ai_emb = sentence_embeddings.get(ai_key)
        
        if ai_emb is not None:
            sim = cosine_similarity(real_emb, ai_emb)
            negative_sent.append(sim)

negative_sent = np.array(negative_sent)
print(f"Negative sentence-transformer: n={len(negative_sent)}, mean={np.mean(negative_sent):.4f}, "
      f"std={np.std(negative_sent):.4f}, range=[{np.min(negative_sent):.4f}, {np.max(negative_sent):.4f}]")

# ── Step 4: ROC AUC ──────────────────────────────────────────────────────────

print("\n" + "=" * 72)
print("  RESULTS")
print("=" * 72)

# Jaccard AUC
jaccard_scores = np.concatenate([positive_jaccard, negative_jaccard])
jaccard_labels = np.concatenate([np.ones(len(positive_jaccard)), np.zeros(len(negative_jaccard))])
jaccard_auc = roc_auc_score(jaccard_labels, jaccard_scores)

print(f"\n  Jaccard Similarity (word overlap):")
print(f"    Positive: mean={np.mean(positive_jaccard):.4f} ± {np.std(positive_jaccard):.4f}")
print(f"    Negative: mean={np.mean(negative_jaccard):.4f} ± {np.std(negative_jaccard):.4f}")
print(f"    Gap:      {np.mean(positive_jaccard) - np.mean(negative_jaccard):+.4f}")
print(f"    ROC AUC:  {jaccard_auc:.4f}")

# Sentence-transformer AUC
sent_scores = np.concatenate([positive_sent, negative_sent])
sent_labels = np.concatenate([np.ones(len(positive_sent)), np.zeros(len(negative_sent))])
sent_auc = roc_auc_score(sent_labels, sent_scores)

print(f"\n  Sentence-Transformer Cosine Similarity:")
print(f"    Positive: mean={np.mean(positive_sent):.4f} ± {np.std(positive_sent):.4f}")
print(f"    Negative: mean={np.mean(negative_sent):.4f} ± {np.std(negative_sent):.4f}")
print(f"    Gap:      {np.mean(positive_sent) - np.mean(negative_sent):+.4f}")
print(f"    ROC AUC:  {sent_auc:.4f}")

# Comparison with MERT
print(f"\n  For comparison:")
print(f"    MERT Top-K AUC (real→AI):  0.5725")
print(f"    Jaccard AUC:               {jaccard_auc:.4f}")
print(f"    Sentence-Transformer AUC:  {sent_auc:.4f}")

# ── Per-pair breakdown ───────────────────────────────────────────────────────

print(f"\n  ── Per-Pair Lyrics Similarity ──")
print(f"  {'Title':<30s}  {'Jaccard':>8s}  {'Sentence':>8s}")
print(f"  {'─' * 50}")
for d in sorted(positive_details, key=lambda x: x.get('jaccard', 0)):
    title = f"{d['title'][:30]}"
    jac = d.get('jaccard', 0)
    sent = d.get('sentence_sim', 0)
    print(f"  {title:<30s}  {jac:>8.4f}  {sent:>8.4f}")

# ── Save results ──────────────────────────────────────────────────────────────

output = {
    'jaccard': {
        'positive_mean': round(float(np.mean(positive_jaccard)), 4),
        'positive_std': round(float(np.std(positive_jaccard)), 4),
        'negative_mean': round(float(np.mean(negative_jaccard)), 4),
        'negative_std': round(float(np.std(negative_jaccard)), 4),
        'gap': round(float(np.mean(positive_jaccard) - np.mean(negative_jaccard)), 4),
        'roc_auc': round(float(jaccard_auc), 4),
        'n_positive': int(len(positive_jaccard)),
        'n_negative': int(len(negative_jaccard)),
    },
    'sentence_transformer': {
        'positive_mean': round(float(np.mean(positive_sent)), 4),
        'positive_std': round(float(np.std(positive_sent)), 4),
        'negative_mean': round(float(np.mean(negative_sent)), 4),
        'negative_std': round(float(np.std(negative_sent)), 4),
        'gap': round(float(np.mean(positive_sent) - np.mean(negative_sent)), 4),
        'roc_auc': round(float(sent_auc), 4),
        'n_positive': int(len(positive_sent)),
        'n_negative': int(len(negative_sent)),
    },
    'mert_comparison': {
        'mert_auc': 0.5725,
        'jaccard_auc': round(float(jaccard_auc), 4),
        'sentence_transformer_auc': round(float(sent_auc), 4),
    },
    'pair_details': positive_details,
}

out_path = RESULTS_DIR / "lyrics_attribution_results.json"
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved → {out_path}")

# ── Plot ──────────────────────────────────────────────────────────────────────

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Lyrics-Based Attribution: Can Text Analysis Solve What MERT Cannot?",
             fontsize=14, fontweight="bold")

# Panel 1: Jaccard distributions
ax = axes[0, 0]
ax.hist(positive_jaccard, bins=15, alpha=0.5, label=f"Derivative (n={len(positive_jaccard)})", color="#27AE60", density=True)
ax.hist(negative_jaccard, bins=15, alpha=0.5, label=f"Unrelated (n={len(negative_jaccard)})", color="#E74C3C", density=True)
ax.axvline(np.mean(positive_jaccard), color="#27AE60", ls="--", lw=1.5)
ax.axvline(np.mean(negative_jaccard), color="#E74C3C", ls="--", lw=1.5)
ax.set_xlabel("Jaccard Similarity")
ax.set_ylabel("Density")
ax.set_title(f"Jaccard Similarity (AUC = {jaccard_auc:.4f})")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 2: Sentence-transformer distributions
ax = axes[0, 1]
ax.hist(positive_sent, bins=15, alpha=0.5, label=f"Derivative (n={len(positive_sent)})", color="#27AE60", density=True)
ax.hist(negative_sent, bins=15, alpha=0.5, label=f"Unrelated (n={len(negative_sent)})", color="#E74C3C", density=True)
ax.axvline(np.mean(positive_sent), color="#27AE60", ls="--", lw=1.5)
ax.axvline(np.mean(negative_sent), color="#E74C3C", ls="--", lw=1.5)
ax.set_xlabel("Sentence-Transformer Cosine Similarity")
ax.set_ylabel("Density")
ax.set_title(f"Sentence-Transformer (AUC = {sent_auc:.4f})")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 3: AUC comparison
ax = axes[1, 0]
methods = ['MERT\n(embedding)', 'Jaccard\n(word overlap)', 'Sentence-Transformer\n(text embedding)']
aucs = [0.5725, jaccard_auc, sent_auc]
colors_bar = ["#3498DB", "#2ECC71", "#E74C3C"]
bars = ax.bar(methods, aucs, color=colors_bar, alpha=0.7)
ax.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.7, label="Random (0.5)")
for bar, auc in zip(bars, aucs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{auc:.4f}",
            ha='center', va='bottom', fontweight='bold')
ax.set_ylabel("ROC AUC")
ax.set_title("AUC Comparison")
ax.legend(fontsize=8)
ax.set_ylim(0.4, 1.05)
ax.grid(True, axis="y", alpha=0.3)

# Panel 4: Per-pair comparison
ax = axes[1, 1]
by_pair = {}
for d in positive_details:
    key = f"{d['fid']}_{d['title'][:20]}"
    by_pair.setdefault(key, {'jaccard': [], 'sentence': []})
    by_pair[key]['jaccard'].append(d.get('jaccard', 0))
    by_pair[key]['sentence'].append(d.get('sentence_sim', 0))

pairs_sorted = sorted(by_pair.items(), key=lambda x: np.mean(x[1]['jaccard']))
jaccard_means = [np.mean(v['jaccard']) for k, v in pairs_sorted]
sent_means = [np.mean(v['sentence']) for k, v in pairs_sorted]
labels = [k[:20] for k, v in pairs_sorted]

y = np.arange(len(jaccard_means))
ax.scatter(jaccard_means, y, color="#2ECC71", s=30, alpha=0.7, label="Jaccard")
ax.scatter(sent_means, y, color="#E74C3C", s=30, alpha=0.7, label="Sentence-Transformer")
ax.axvline(np.mean(negative_jaccard), color="#E74C3C", ls="--", lw=1, alpha=0.5)
ax.set_xlabel("Similarity")
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=5)
ax.set_title("Per-Pair Lyrics Similarity")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_png = RESULTS_DIR / "lyrics_attribution_results.png"
plt.savefig(str(out_png), dpi=150, bbox_inches="tight")
print(f"Plot saved → {out_png}")

# ── Final verdict ─────────────────────────────────────────────────────────────

print("\n" + "=" * 72)
print("  VERDICT")
print("=" * 72)

best_lyrics_auc = max(jaccard_auc, sent_auc)
mert_auc = 0.5725

if best_lyrics_auc > mert_auc + 0.05:
    print(f"  ✅ Lyrics analysis OUTPERFORMS MERT (AUC {best_lyrics_auc:.4f} vs {mert_auc:.4f})")
    print(f"  → Lyrics IS the missing signal for real→AI attribution")
elif best_lyrics_auc > 0.7:
    print(f"  ⚠️  Lyrics analysis is MODERATE (AUC {best_lyrics_auc:.4f})")
    print(f"  → Better than MERT but not definitive")
elif best_lyrics_auc > mert_auc:
    print(f"  ⚠️  Lyrics analysis is SLIGHTLY BETTER than MERT (AUC {best_lyrics_auc:.4f} vs {mert_auc:.4f})")
    print(f"  → Marginal improvement, not transformative")
else:
    print(f"  ❌ Lyrics analysis does NOT outperform MERT (AUC {best_lyrics_auc:.4f} vs {mert_auc:.4f})")
    print(f"  → Lyrics is not the missing signal")

print(f"\n  Key insight: {'Jaccard' if jaccard_auc > sent_auc else 'Sentence-transformer'} is the better lyrics metric")
print(f"  The gap between derivative and unrelated lyrics is "
      f"{np.mean(positive_jaccard) - np.mean(negative_jaccard):+.4f} (Jaccard) / "
      f"{np.mean(positive_sent) - np.mean(negative_sent):+.4f} (sentence)")
