#!/usr/bin/env python3
"""
Task 2 — False Positive Investigation
=======================================
Find negative pairs scoring Jaccard > 0.3.
Show real title, AI title, first 4 lines of each transcript.
Also compute MERT scores for these pairs.
"""

import json
import numpy as np
import librosa
import torch
from transformers import AutoProcessor, AutoModel
from pathlib import Path

PAIRS_DIR = Path("/home/nikos_pc/Desktop/Orfium/orfium_audio_attribution_task/qwen/results/real_ai_pairs")

with open(PAIRS_DIR / "transcripts.json") as f:
    transcripts = json.load(f)

mapping_path = PAIRS_DIR / "real_ai_pairs_mapping.json"
with open(mapping_path) as f:
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

def get_first_lines(text, n=4):
    lines = text.strip().split('\n')
    # If no newlines, split by sentences
    if len(lines) <= 1:
        lines = text.strip().split('. ')
    return '. '.join(lines[:n])[:300]

print("=" * 72)
print("  TASK 2 — FALSE POSITIVE INVESTIGATION")
print("=" * 72)

# Find all negative pairs with Jaccard > 0.3
fids = list(complete_pairs.keys())
high_jaccard_negatives = []
all_negatives = []

for fid_a in fids:
    real_key = Path(complete_pairs[fid_a]['real']).stem
    real_text = transcripts.get(real_key, "")
    if not real_text:
        continue
    
    other_fids = [f for f in fids if f != fid_a]
    np.random.seed(42 + int(fid_a))
    sampled = list(np.random.choice(other_fids, min(3, len(other_fids)), replace=False))
    
    for fid_b in sampled:
        ai_path = complete_pairs[fid_b]['ai'][0]
        ai_key = Path(ai_path).stem
        ai_text = transcripts.get(ai_key, "")
        if ai_text and real_text:
            sim = jaccard_similarity(real_text, ai_text)
            entry = {
                'real_fid': fid_a,
                'ai_fid': fid_b,
                'real_title': complete_pairs[fid_a]['title'],
                'real_artist': complete_pairs[fid_a]['artist'],
                'ai_file': ai_key,
                'ai_algorithm': pairs[str(fid_b)]['algorithm'],
                'jaccard': round(sim, 4),
                'real_text_preview': get_first_lines(real_text, 4),
                'ai_text_preview': get_first_lines(ai_text, 4),
            }
            all_negatives.append(entry)
            if sim > 0.3:
                high_jaccard_negatives.append(entry)

all_negatives = sorted(all_negatives, key=lambda x: x['jaccard'], reverse=True)

print(f"\n  Total negative pairs: {len(all_negatives)}")
print(f"  Pairs with Jaccard > 0.3: {len(high_jaccard_negatives)}")
print(f"  Pairs with Jaccard > 0.25: {sum(1 for x in all_negatives if x['jaccard'] > 0.25)}")
print(f"  Pairs with Jaccard > 0.2: {sum(1 for x in all_negatives if x['jaccard'] > 0.2)}")

# Show all high-Jaccard false positives
print(f"\n  ── HIGH JACCARD FALSE POSITIVES (>0.3) ──")
for i, entry in enumerate(high_jaccard_negatives):
    print(f"\n  #{i+1}: Jaccard = {entry['jaccard']:.4f}")
    print(f"    Real: \"{entry['real_title']}\" by {entry['real_artist']} (ID={entry['real_fid']})")
    print(f"    AI:   {entry['ai_file']} (ID={entry['ai_fid']}, {entry['ai_algorithm']})")
    print(f"    Real text: {entry['real_text_preview']}")
    print(f"    AI text:   {entry['ai_text_preview']}")
    # Check for common words
    real_words = set(entry['real_text_preview'].lower().split())
    ai_words = set(entry['ai_text_preview'].lower().split())
    common = real_words & ai_words
    if common:
        print(f"    Common words: {', '.join(sorted(common)[:15])}")

# ── Check for leakage in high-Jaccard negatives ──────────────────────────────

print(f"\n  ── Leakage Check for High-Jaccard Negatives ──")
for entry in high_jaccard_negatives:
    real_fid = entry['real_fid']
    ai_fid = entry['ai_fid']
    
    # Does the real song's AI derivative exist in the positive set?
    # (i.e., is the AI song's source the same as the real song?)
    # This would mean the negative pair is actually real_X → AI_from_Y
    # where real_X's own AI derivative also exists
    ai_from_real = pairs.get(str(real_fid), {})
    ai_from_real_files = ai_from_real.get('fake_files', [])
    
    # The AI file in this negative pair is from fid_b, not fid_a
    # So it's real_A → AI_B, not real_A → AI_A
    # Check if there's any lyrics overlap between real_A and real_B
    real_b_key = Path(complete_pairs[ai_fid]['real']).stem
    real_b_text = transcripts.get(real_b_key, "")
    if real_b_text:
        cross_lyrics_sim = jaccard_similarity(entry['real_text_preview'] + transcripts.get(Path(complete_pairs[real_fid]['real']).stem, ""), 
                                              real_b_text)
    else:
        cross_lyrics_sim = 0
    
    if cross_lyrics_sim > 0.5:
        print(f"  ⚠️  {entry['real_title']} ↔ {entry['ai_file']}: Real songs share {cross_lyrics_sim:.2f} Jaccard — may be covers/remixes")

# ── Compute MERT scores for high-Jaccard false positives ─────────────────────

print(f"\n  ── MERT Scores for High-Jaccard False Positives ──")

def segment_audio(path):
    y, _ = librosa.load(str(path), sr=24000, mono=True)
    win = int(10 * 24000)
    hop = int(5 * 24000)
    min_len = int(3 * 24000)
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

def cosine(a, b):
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {DEVICE}")

# Load cached MERT embeddings or extract new ones
def get_cached_mert(path):
    cached = Path(path).parent / f"{Path(path).stem}_mert.npy"
    if cached.exists():
        return np.load(cached)
    return None

# Try to load MERT from cache
print(f"  Loading MERT embeddings from cache...")
mert_cache = {}
for entry in high_jaccard_negatives:
    real_path = complete_pairs[entry['real_fid']]['real']
    ai_path = PAIRS_DIR / entry['ai_file']
    
    real_key = Path(real_path).stem
    ai_key = entry['ai_file']
    
    if real_key not in mert_cache:
        emb = get_cached_mert(real_path)
        if emb is not None:
            mert_cache[real_key] = emb
    if ai_key not in mert_cache:
        emb = get_cached_mert(str(ai_path))
        if emb is not None:
            mert_cache[ai_key] = emb

print(f"  Cached embeddings: {len(mert_cache)}")

# If we don't have enough cached, extract what we can
missing = 0
for entry in high_jaccard_negatives:
    real_path = complete_pairs[entry['real_fid']]['real']
    ai_path = PAIRS_DIR / entry['ai_file']
    real_key = Path(real_path).stem
    ai_key = entry['ai_file']
    
    if real_key not in mert_cache:
        missing += 1
    if ai_key not in mert_cache:
        missing += 1

if missing > 0:
    print(f"  Missing {missing} embeddings — computing...")
    processor = AutoProcessor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
    model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True).to(DEVICE)
    
    needed = set()
    for entry in high_jaccard_negatives:
        real_path = complete_pairs[entry['real_fid']]['real']
        ai_path = PAIRS_DIR / entry['ai_file']
        if Path(real_path).stem not in mert_cache:
            needed.add(real_path)
        if entry['ai_file'] not in mert_cache:
            needed.add(str(ai_path))
    
    for i, path in enumerate(needed):
        print(f"  [{i+1}/{len(needed)}] Extracting MERT for {Path(path).name}...")
        segs = segment_audio(path)
        model.eval()
        seg_embs = []
        with torch.no_grad():
            for seg in segs:
                inputs = processor(seg, sampling_rate=24000, return_tensors="pt", padding=True)
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                out = model(**inputs)
                emb = out.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
                seg_embs.append(emb.astype(np.float32))
        
        key = Path(path).stem
        mert_cache[key] = np.mean(seg_embs, axis=0)
        # Cache to disk
        np.save(PAIRS_DIR / f"{key}_mert.npy", mert_cache[key])

print(f"\n  {'Real Song':<30s}  {'AI File':<40s}  {'Jaccard':>8s}  {'MERT':>8s}")
print(f"  {'─' * 90}")

for entry in high_jaccard_negatives:
    real_path = complete_pairs[entry['real_fid']]['real']
    ai_path = PAIRS_DIR / entry['ai_file']
    real_key = Path(real_path).stem
    ai_key = entry['ai_file']
    
    real_emb = mert_cache.get(real_key)
    ai_emb = mert_cache.get(ai_key)
    
    if real_emb is not None and ai_emb is not None:
        mert_sim = cosine(real_emb, ai_emb)
    else:
        mert_sim = None
    
    mert_str = f"{mert_sim:.4f}" if mert_sim is not None else "N/A"
    print(f"  {entry['real_title'][:30]:<30s}  {entry['ai_file']:<40s}  {entry['jaccard']:>8.4f}  {mert_str:>8s}")

# Summary
print(f"\n  ── Summary ──")
high_jac_entries = high_jaccard_negatives
mert_scores = []
for entry in high_jac_entries:
    real_path = complete_pairs[entry['real_fid']]['real']
    ai_path = PAIRS_DIR / entry['ai_file']
    real_key = Path(real_path).stem
    ai_key = entry['ai_file']
    real_emb = mert_cache.get(real_key)
    ai_emb = mert_cache.get(ai_key)
    if real_emb is not None and ai_emb is not None:
        mert_scores.append(cosine(real_emb, ai_emb))

if mert_scores:
    print(f"  MERT mean for high-Jaccard negatives: {np.mean(mert_scores):.4f}")
    print(f"  MERT std:  {np.std(mert_scores):.4f}")
    print(f"  MERT range: [{np.min(mert_scores):.4f}, {np.max(mert_scores):.4f}]")
    print(f"  High-Jaccard but low-MERT (<0.90): {sum(1 for x in mert_scores if x < 0.90)}/{len(mert_scores)}")
    print(f"  High-Jaccard AND high-MERT (>0.90): {sum(1 for x in mert_scores if x >= 0.90)}/{len(mert_scores)}")

# Save results
output = {
    'n_high_jaccard': len(high_jaccard_negatives),
    'high_jaccard_pairs': high_jaccard_negatives,
    'mert_scores': {i: round(mert_scores[i], 4) if i < len(mert_scores) else None for i in range(len(high_jaccard_negatives))} if mert_scores else {},
    'mert_mean': round(float(np.mean(mert_scores)), 4) if mert_scores else None,
    'all_negatives_jaccard_gt_0.2': len([x for x in all_negatives if x['jaccard'] > 0.2]),
}

out_path = PAIRS_DIR.parent / "lyrics_task2_false_positives.json"
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"\n  Results saved → {out_path}")
