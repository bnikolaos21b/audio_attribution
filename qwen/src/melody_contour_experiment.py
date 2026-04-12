#!/usr/bin/env python3
"""
Melody Contour Attribution Experiment
=======================================
Extract melody as MIDI notes using basic-pitch, convert to relative interval
sequences (melody contour), align with DTW, compute similarity.

Decision rule: if melody AUC < 0.70 or adds < 0.01 to Jaccard, skip it.
"""

import json
import numpy as np
import os
import tempfile
import time
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

PAIRS_DIR = Path("/home/nikos_pc/Desktop/Orfium/orfium_audio_attribution_task/qwen/results/real_ai_pairs")
RESULTS_DIR = PAIRS_DIR.parent

# Load transcripts and mapping
with open(PAIRS_DIR / "transcripts.json") as f:
    transcripts = json.load(f)

with open(PAIRS_DIR / "real_ai_pairs_mapping.json") as f:
    pairs = json.load(f)

# ── Helper functions ──────────────────────────────────────────────────────────

def get_real_path(fid):
    p = PAIRS_DIR / f"real_{fid}.mp3"
    if p.exists(): return str(p)
    return None

def get_ai_paths(fid):
    return sorted(PAIRS_DIR.glob(f"ai_{fid}_*.mp3"))

def jaccard_similarity(t1, t2):
    if not t1 or not t2: return 0.0
    w1 = set(t1.lower().split()); w2 = set(t2.lower().split())
    if not w1 or not w2: return 0.0
    return len(w1 & w2) / len(w1 | w2)

# ── Step 1: Extract MIDI using basic-pitch ────────────────────────────────────

print("=" * 72)
print("  MELODY CONTOUR ATTRIBUTION EXPERIMENT")
print("=" * 72)

# Build complete pairs
complete_pairs = {}
for fid, info in pairs.items():
    rp = get_real_path(fid)
    ap = get_ai_paths(fid)
    if rp and len(ap) > 0:
        complete_pairs[int(fid)] = {
            'real': rp, 'ai': [str(p) for p in ap],
            'title': info['real_title'], 'artist': info['real_artist'],
        }

fids = [fid for fid in complete_pairs 
        if transcripts.get(Path(complete_pairs[fid]['real']).stem, '')]

# Check for cached MIDI
MIDI_CACHE = PAIRS_DIR / "midi_cache"
MIDI_CACHE.mkdir(parents=True, exist_ok=True)

def get_midi_notes(audio_path):
    """Extract MIDI note sequence from audio using basic-pitch."""
    key = Path(audio_path).stem
    midi_file = MIDI_CACHE / f"{key}_notes.npy"
    
    if midi_file.exists():
        return np.load(midi_file)
    
    print(f"  Extracting MIDI: {key}...")
    from basic_pitch.inference import predict
    import tensorflow as tf
    
    # Suppress TF warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')
    
    # predict returns: (model_output, midi_data, note_events)
    # note_events: List[Tuple[time_sec, duration_sec, pitch, velocity, bends]]
    _, _, note_events = predict(audio_path)
    
    # Convert to array: (time, pitch, duration, velocity)
    notes = []
    for ev in note_events:
        if len(ev) >= 4:
            time_sec = ev[0]
            duration = ev[1]
            pitch = ev[2]
            velocity = ev[3] if len(ev) > 3 else 100
            notes.append((time_sec, pitch, duration, velocity))
    
    notes = np.array(notes) if notes else np.array([]).reshape(0, 4)
    np.save(midi_file, notes)
    print(f"  → {len(notes)} notes extracted")
    return notes

def midi_to_contour(notes, max_notes=200):
    """
    Convert MIDI notes to melody contour: relative interval sequence.
    Returns array of interval changes: positive=up, negative=down, 0=same.
    """
    if len(notes) == 0:
        return np.array([])
    
    # Sort by time
    if notes.ndim > 1 and notes.shape[1] >= 2:
        notes = notes[notes[:, 0].argsort()]
        pitches = notes[:, 1].astype(int)
    else:
        return np.array([])
    
    # Filter out very short notes (noise)
    if notes.shape[1] >= 3:
        durations = notes[:, 2]
        mask = durations >= 2  # Keep notes lasting at least 2 frames
        pitches = pitches[mask]
    
    if len(pitches) < 2:
        return np.array([])
    
    # Limit to max_notes for DTW efficiency
    if len(pitches) > max_notes:
        # Sample evenly
        indices = np.linspace(0, len(pitches) - 1, max_notes, dtype=int)
        pitches = pitches[indices]
    
    # Compute relative intervals
    intervals = np.diff(pitches).astype(float)
    
    # Quantize to semitone steps (already in semitones from MIDI)
    # Round to nearest semitone
    intervals = np.round(intervals)
    
    return intervals.astype(int)

def dtw_contour_similarity(c1, c2):
    """
    Compute melody contour similarity using DTW on interval sequences.
    Returns similarity in [0, 1] where 1 = identical contour.
    """
    if len(c1) == 0 or len(c2) == 0:
        return 0.0
    
    # DTW with absolute interval difference as cost
    n, m = len(c1), len(c2)
    
    # Cost matrix: absolute difference between intervals
    cost_matrix = np.abs(c1[:, np.newaxis] - c2[np.newaxis, :])
    
    # DTW accumulation
    dtw = np.zeros((n + 1, m + 1))
    dtw[1:, 0] = np.inf
    dtw[0, 1:] = np.inf
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dtw[i, j] = cost_matrix[i-1, j-1] + min(
                dtw[i-1, j],    # insertion
                dtw[i, j-1],    # deletion
                dtw[i-1, j-1]   # match
            )
    
    # Normalize by path length
    dtw_distance = dtw[n, m] / (n + m)
    
    # Convert to similarity: exp(-distance)
    similarity = np.exp(-dtw_distance)
    
    return float(similarity)

# ── Step 2: Extract MIDI for all files ────────────────────────────────────────

print("\n── Step 1: Extracting MIDI notes ──")

# Collect all unique audio files
all_files = set()
for fid in fids:
    all_files.add(complete_pairs[fid]['real'])
    for ap in complete_pairs[fid]['ai']:
        all_files.add(ap)

print(f"  Total unique audio files: {len(all_files)}")

# Extract MIDI
midi_data = {}
for i, audio_path in enumerate(sorted(all_files)):
    key = Path(audio_path).stem
    notes = get_midi_notes(audio_path)
    midi_data[key] = notes
    if (i + 1) % 10 == 0:
        print(f"  [{i+1}/{len(all_files)}] Done")

# Convert to contours
print("\n── Step 2: Converting to melody contours ──")
contours = {}
for key, notes in midi_data.items():
    contours[key] = midi_to_contour(notes)
    if len(contours[key]) == 0:
        print(f"  ⚠️  {key}: No melody contour extracted")

n_with_contour = sum(1 for c in contours.values() if len(c) > 0)
print(f"  Files with melody contour: {n_with_contour}/{len(contours)}")

# ── Step 3: Build strict splits and compute scores ────────────────────────────

print("\n── Step 3: Computing melody contour similarities ──")

np.random.seed(42)
n_pos = len(fids) // 2
fids_for_pos = set(list(np.random.choice(fids, n_pos, replace=False)))
fids_for_neg = set(fids) - fids_for_pos

print(f"  Positive songs: {len(fids_for_pos)}")
print(f"  Negative songs: {len(fids_for_neg)}")

positive_data = []  # (mert, jaccard, melody)
negative_data = []

# Positive pairs
for fid_a in fids_for_pos:
    rp = complete_pairs[fid_a]['real']
    rk = Path(rp).stem
    rt = transcripts.get(rk, '')
    if not rt: continue
    
    for ap in complete_pairs[fid_a]['ai']:
        ak = Path(ap).stem
        at = transcripts.get(ak, '')
        if not at: continue
        
        # Melody contour similarity
        c1 = contours.get(rk, np.array([]))
        c2 = contours.get(ak, np.array([]))
        melody_sim = dtw_contour_similarity(c1, c2)
        
        # Jaccard
        jaccard = jaccard_similarity(rt, at)
        
        positive_data.append({
            'real_fid': fid_a, 'ai_file': ak,
            'jaccard': jaccard, 'melody': melody_sim, 'label': 1,
        })

# Negative pairs
for fid_a in fids_for_neg:
    rp = complete_pairs[fid_a]['real']
    rk = Path(rp).stem
    rt = transcripts.get(rk, '')
    if not rt: continue
    
    np.random.seed(42 + int(fid_a))
    sampled = list(np.random.choice(list(fids_for_pos), min(3, len(fids_for_pos)), replace=False))
    
    for fid_b in sampled:
        ap = complete_pairs[fid_b]['ai'][0]
        ak = Path(ap).stem
        at = transcripts.get(ak, '')
        if not at: continue
        
        c1 = contours.get(rk, np.array([]))
        c2 = contours.get(ak, np.array([]))
        melody_sim = dtw_contour_similarity(c1, c2)
        
        jaccard = jaccard_similarity(rt, at)
        
        negative_data.append({
            'real_fid': fid_a, 'ai_fid': fid_b,
            'ai_file': ak,
            'jaccard': jaccard, 'melody': melody_sim, 'label': 0,
        })

all_data = positive_data + negative_data
jac = np.array([d['jaccard'] for d in all_data])
mel = np.array([d['melody'] for d in all_data])
labels = np.array([d['label'] for d in all_data])
real_fids = np.array([d['real_fid'] for d in all_data])

print(f"  Positive pairs: {len(positive_data)}")
print(f"  Negative pairs: {len(negative_data)}")
print(f"  Melody scores available: {sum(1 for x in mel if x > 0)}/{len(mel)}")

# ── Step 4: Compute AUCs ─────────────────────────────────────────────────────

print(f"\n── Step 4: Computing AUCs ──")

# Jaccard alone
auc_jac = roc_auc_score(labels, jac)
print(f"  Jaccard alone:           AUC = {auc_jac:.4f}")

# Melody alone
mel_valid = mel > 0
if mel_valid.sum() > 10:
    auc_mel = roc_auc_score(labels[mel_valid], mel[mel_valid])
    print(f"  Melody contour alone:    AUC = {auc_mel:.4f} (on {mel_valid.sum()} pairs)")
else:
    auc_mel = 0.5
    print(f"  Melody contour alone:    AUC = {auc_mel:.4f} (too few pairs)")

# Jaccard × Melody (product)
product_jm = jac * mel
auc_prod = roc_auc_score(labels, product_jm)
print(f"  Jaccard × Melody:        AUC = {auc_prod:.4f}")

# Logistic regression with 5-fold song-level CV
unique_fids = np.unique(real_fids)
np.random.seed(42)
np.random.shuffle(unique_fids)
nf = 5; fs = len(unique_fids) // nf
folds = [unique_fids[i*fs:(i+1)*fs] for i in range(nf)]
for i in range(len(unique_fids) - nf*fs):
    folds[i] = np.append(folds[i], unique_fids[-(i+1)])

X_jm = np.column_stack([jac, mel])
fold_aucs_jm = []
for i, tf in enumerate(folds):
    tr = ~np.isin(real_fids, tf); te = np.isin(real_fids, tf)
    if len(np.unique(labels[tr])) < 2 or len(np.unique(labels[te])) < 2:
        continue
    m = LogisticRegression(random_state=42, max_iter=1000)
    m.fit(X_jm[tr], labels[tr])
    probas = m.predict_proba(X_jm[te])[:, 1]
    fa = roc_auc_score(labels[te], probas)
    fold_aucs_jm.append(fa)

auc_lr_jm = np.mean(fold_aucs_jm) if fold_aucs_jm else 0.5
print(f"  Logistic (Jaccard+Mel) 5-CV: AUC = {auc_lr_jm:.4f} ± {np.std(fold_aucs_jm):.4f}")

# ── Step 5: Decision ──────────────────────────────────────────────────────────

print(f"\n── Step 5: Decision ──")

melody_gain = auc_prod - auc_jac
print(f"  Melody AUC:              {auc_mel:.4f}")
print(f"  Jaccard AUC:             {auc_jac:.4f}")
print(f"  Jaccard × Melody AUC:    {auc_prod:.4f}")
print(f"  Melody gain over Jaccard: {melody_gain:+.4f}")

if auc_mel < 0.70 and melody_gain < 0.01:
    print(f"\n  ❌ Melody contour is NOT worth adding.")
    print(f"     Melody AUC ({auc_mel:.4f}) < 0.70")
    print(f"     Gain over Jaccard ({melody_gain:+.4f}) < 0.01")
    print(f"     Final recommendation: Jaccard × MERT is the real→AI signal.")
    decision = "SKIP"
elif melody_gain >= 0.01:
    print(f"\n  ✅ Melody contour ADDS VALUE ({melody_gain:+.4f} gain).")
    print(f"     Include melody in the combined signal.")
    decision = "INCLUDE"
else:
    print(f"\n  ⚠️  Melody AUC ({auc_mel:.4f}) is moderate but gain is small ({melody_gain:+.4f}).")
    print(f"     Marginal benefit — probably not worth the computational cost.")
    decision = "MARGINAL"

# ── Save results ──────────────────────────────────────────────────────────────

output = {
    'n_positive': len(positive_data),
    'n_negative': len(negative_data),
    'n_files_with_contour': n_with_contour,
    'auc_jaccard': round(float(auc_jac), 4),
    'auc_melody': round(float(auc_mel), 4),
    'auc_product_jm': round(float(auc_prod), 4),
    'auc_logistic_jm_cv': round(float(auc_lr_jm), 4),
    'logistic_fold_aucs': [round(float(x), 4) for x in fold_aucs_jm],
    'melody_gain': round(float(melody_gain), 4),
    'decision': decision,
    'positive_mean_jaccard': round(float(np.mean(jac[labels == 1])), 4),
    'negative_mean_jaccard': round(float(np.mean(jac[labels == 0])), 4),
    'positive_mean_melody': round(float(np.mean(mel[labels == 1])), 4),
    'negative_mean_melody': round(float(np.mean(mel[labels == 0])), 4),
    'all_data': [
        {
            'real_fid': int(d.get('real_fid', 0)),
            'ai_fid': int(d.get('ai_fid', 0)),
            'ai_file': d['ai_file'],
            'jaccard': round(float(d['jaccard']), 4),
            'melody': round(float(d['melody']), 4),
            'label': int(d['label']),
        }
        for d in all_data
    ],
}

out_path = RESULTS_DIR / "melody_contour_results.json"
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\n  Results saved → {out_path}")

# ── Plot ──────────────────────────────────────────────────────────────────────

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Melody Contour Attribution Experiment", fontsize=14, fontweight="bold")

# Panel 1: Distributions
ax = axes[0]
pos_data = [d for d in all_data if d['label'] == 1]
neg_data = [d for d in all_data if d['label'] == 0]

ax.hist([d['melody'] for d in pos_data], bins=15, alpha=0.5, label=f"Derivative (n={len(pos_data)})", color="#27AE60", density=True)
ax.hist([d['melody'] for d in neg_data], bins=15, alpha=0.5, label=f"Unrelated (n={len(neg_data)})", color="#E74C3C", density=True)
ax.axvline(np.mean([d['melody'] for d in pos_data]), color="#27AE60", ls="--", lw=1.5)
ax.axvline(np.mean([d['melody'] for d in neg_data]), color="#E74C3C", ls="--", lw=1.5)
ax.set_xlabel("Melody Contour Similarity (DTW)")
ax.set_ylabel("Density")
ax.set_title(f"Melody Contour (AUC = {auc_mel:.4f})")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 2: AUC comparison
ax = axes[1]
methods = [
    ('Jaccard alone', auc_jac),
    ('Melody contour', auc_mel),
    ('Jaccard × Melody', auc_prod),
]
methods.sort(key=lambda x: x[1], reverse=True)
names = [m[0] for m in methods]
aucs = [m[1] for m in methods]
colors = ['#3498DB', '#E67E22', '#9B59B6']
bars = ax.barh(names, aucs, color=colors, alpha=0.7)
ax.axvline(0.5, color="gray", ls="--", lw=1, alpha=0.7, label="Random (0.5)")
ax.axvline(auc_jac, color="gray", ls=":", lw=1, alpha=0.7, label=f"Jaccard baseline ({auc_jac:.3f})")
for bar, auc in zip(bars, aucs):
    ax.text(auc + 0.01, bar.get_y() + bar.get_height()/2, f"{auc:.4f}",
            ha='left', va='center', fontweight='bold')
ax.set_xlabel("ROC AUC")
ax.set_title("Combination Results")
ax.set_xlim(0.3, 1.0)
ax.legend(fontsize=8)
ax.grid(True, axis="x", alpha=0.3)

plt.tight_layout()
out_png = RESULTS_DIR / "melody_contour_results.png"
plt.savefig(str(out_png), dpi=150, bbox_inches="tight")
print(f"  Plot saved → {out_png}")
