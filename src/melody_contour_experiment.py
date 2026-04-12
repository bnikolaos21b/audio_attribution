#!/usr/bin/env python3
"""
Melody Contour Attribution Experiment
=======================================
Tests whether melody contour similarity (basic-pitch + DTW) adds discriminative
power on top of Jaccard for real→AI attribution.

Decision: melody is SKIPPED if AUC < 0.70 or gain over Jaccard < 0.01.

Results (melody_contour_results.json):
  Jaccard alone:              AUC = 0.8649
  Melody contour alone:       AUC = 0.5696  (barely above chance)
  Jaccard × Melody (product): AUC = 0.7427  (worse than Jaccard alone)
  Logistic Jaccard+Melody CV: AUC = 0.8428
  Decision:                   SKIP

Usage:
  python src/melody_contour_experiment.py
"""

import json
import numpy as np
import os
import time
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

# ── Paths ─────────────────────────────────────────────────────────────────────
_SRC_DIR     = Path(__file__).parent
_PROJECT_DIR = _SRC_DIR.parent
PAIRS_DIR    = _PROJECT_DIR / "data" / "real_ai_pairs"
RESULTS_DIR  = _PROJECT_DIR / "results" / "real_ai"
MIDI_CACHE   = PAIRS_DIR / "midi_cache"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MIDI_CACHE.mkdir(parents=True, exist_ok=True)

# Load data
with open(PAIRS_DIR / "transcripts.json") as f:
    transcripts = json.load(f)
with open(PAIRS_DIR / "real_ai_pairs_mapping.json") as f:
    pairs = json.load(f)


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_real_path(fid):
    p = PAIRS_DIR / f"real_{fid}.mp3"
    return str(p) if p.exists() else None

def get_ai_paths(fid):
    return sorted(PAIRS_DIR.glob(f"ai_{fid}_*.mp3"))

def jaccard_similarity(t1, t2):
    if not t1 or not t2: return 0.0
    w1 = set(t1.lower().split()); w2 = set(t2.lower().split())
    if not w1 or not w2: return 0.0
    return len(w1 & w2) / len(w1 | w2)


# ── MIDI extraction ───────────────────────────────────────────────────────────

def get_midi_notes(audio_path):
    key       = Path(audio_path).stem
    midi_file = MIDI_CACHE / f"{key}_notes.npy"
    if midi_file.exists():
        return np.load(midi_file)

    print(f"  Extracting MIDI: {key}...")
    from basic_pitch.inference import predict
    import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    _, _, note_events = predict(audio_path)
    notes = []
    for ev in note_events:
        if len(ev) >= 4:
            notes.append((ev[0], ev[2], ev[1], ev[3] if len(ev) > 3 else 100))
    notes = np.array(notes) if notes else np.array([]).reshape(0, 4)
    np.save(midi_file, notes)
    return notes


def midi_to_contour(notes, max_notes=200):
    if len(notes) == 0: return np.array([])
    if notes.ndim < 2 or notes.shape[1] < 2: return np.array([])
    notes = notes[notes[:, 0].argsort()]
    pitches = notes[:, 1].astype(int)
    if notes.shape[1] >= 3:
        durations = notes[:, 2]
        pitches = pitches[durations >= 2]
    if len(pitches) < 2: return np.array([])
    if len(pitches) > max_notes:
        pitches = pitches[np.linspace(0, len(pitches) - 1, max_notes, dtype=int)]
    return np.round(np.diff(pitches)).astype(int)


def dtw_contour_similarity(c1, c2):
    if len(c1) == 0 or len(c2) == 0: return 0.0
    n, m = len(c1), len(c2)
    cost = np.abs(c1[:, None] - c2[None, :])
    dtw  = np.zeros((n + 1, m + 1)); dtw[1:, 0] = dtw[0, 1:] = np.inf
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dtw[i, j] = cost[i-1, j-1] + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
    return float(np.exp(-dtw[n, m] / (n + m)))


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
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
                "real": rp, "ai": [str(p) for p in ap],
                "title": info["real_title"], "artist": info["real_artist"],
            }

    fids = [fid for fid in complete_pairs
            if transcripts.get(Path(complete_pairs[fid]["real"]).stem, "")]

    # Extract MIDI
    print(f"\n── Step 1: Extracting MIDI ({len(fids)} songs) ──")
    all_files = set()
    for fid in fids:
        all_files.add(complete_pairs[fid]["real"])
        for ap in complete_pairs[fid]["ai"]:
            all_files.add(ap)

    midi_data = {}
    for i, ap in enumerate(sorted(all_files)):
        key = Path(ap).stem
        midi_data[key] = get_midi_notes(ap)
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(all_files)}]")

    contours = {k: midi_to_contour(v) for k, v in midi_data.items()}
    n_ok = sum(1 for c in contours.values() if len(c) > 0)
    print(f"  Melody contours extracted: {n_ok}/{len(contours)}")

    # Strict split: half songs for positive, other half for negative
    print(f"\n── Step 2: Building strict splits ──")
    np.random.seed(42)
    n_pos = len(fids) // 2
    fids_pos = set(list(np.random.choice(fids, n_pos, replace=False)))
    fids_neg = set(fids) - fids_pos

    positive_data = []
    negative_data = []

    for fid_a in fids_pos:
        rp = complete_pairs[fid_a]["real"]
        rk = Path(rp).stem
        rt = transcripts.get(rk, "")
        if not rt: continue
        for ap in complete_pairs[fid_a]["ai"]:
            ak = Path(ap).stem
            at = transcripts.get(ak, "")
            if not at: continue
            c1 = contours.get(rk, np.array([]))
            c2 = contours.get(ak, np.array([]))
            positive_data.append({
                "real_fid": fid_a, "ai_fid": 0, "ai_file": ak,
                "jaccard": jaccard_similarity(rt, at),
                "melody": dtw_contour_similarity(c1, c2),
                "label": 1,
            })

    for fid_a in fids_neg:
        rp = complete_pairs[fid_a]["real"]
        rk = Path(rp).stem
        rt = transcripts.get(rk, "")
        if not rt: continue
        np.random.seed(42 + int(fid_a))
        sampled = list(np.random.choice(list(fids_pos), min(3, len(fids_pos)), replace=False))
        for fid_b in sampled:
            ap = complete_pairs[fid_b]["ai"][0]
            ak = Path(ap).stem
            at = transcripts.get(ak, "")
            if not at: continue
            c1 = contours.get(rk, np.array([]))
            c2 = contours.get(ak, np.array([]))
            negative_data.append({
                "real_fid": fid_a, "ai_fid": fid_b, "ai_file": ak,
                "jaccard": jaccard_similarity(rt, at),
                "melody": dtw_contour_similarity(c1, c2),
                "label": 0,
            })

    all_data = positive_data + negative_data
    jac      = np.array([d["jaccard"] for d in all_data])
    mel      = np.array([d["melody"]  for d in all_data])
    labels   = np.array([d["label"]   for d in all_data])
    fid_arr  = np.array([d["real_fid"] for d in all_data])

    auc_jac = roc_auc_score(labels, jac)
    mel_ok  = mel > 0
    auc_mel = roc_auc_score(labels[mel_ok], mel[mel_ok]) if mel_ok.sum() > 10 else 0.5
    auc_prod = roc_auc_score(labels, jac * mel)

    # 5-fold song-level CV logistic regression
    unique_fids = np.unique(fid_arr)
    np.random.seed(42); np.random.shuffle(unique_fids)
    nf = 5; fs = len(unique_fids) // nf
    folds = [unique_fids[i*fs:(i+1)*fs] for i in range(nf)]
    for i in range(len(unique_fids) - nf*fs):
        folds[i] = np.append(folds[i], unique_fids[-(i+1)])

    X = np.column_stack([jac, mel])
    fold_aucs = []
    for tf in folds:
        tr = ~np.isin(fid_arr, tf); te = np.isin(fid_arr, tf)
        if len(np.unique(labels[tr])) < 2 or len(np.unique(labels[te])) < 2:
            continue
        m = LogisticRegression(random_state=42, max_iter=1000)
        m.fit(X[tr], labels[tr])
        fold_aucs.append(roc_auc_score(labels[te], m.predict_proba(X[te])[:, 1]))

    auc_lr = float(np.mean(fold_aucs)) if fold_aucs else 0.5

    melody_gain = auc_prod - auc_jac
    decision = (
        "SKIP" if auc_mel < 0.70 and melody_gain < 0.01
        else "INCLUDE" if melody_gain >= 0.01
        else "MARGINAL"
    )

    print(f"\n── Results ──")
    print(f"  Jaccard alone:              AUC = {auc_jac:.4f}")
    print(f"  Melody contour alone:       AUC = {auc_mel:.4f}")
    print(f"  Jaccard × Melody (product): AUC = {auc_prod:.4f}")
    print(f"  Logistic Jaccard+Mel 5-CV:  AUC = {auc_lr:.4f}")
    print(f"  Melody gain:                {melody_gain:+.4f}")
    print(f"  Decision:                   {decision}")

    output = {
        "n_positive":          len(positive_data),
        "n_negative":          len(negative_data),
        "n_files_with_contour": n_ok,
        "auc_jaccard":         round(auc_jac,  4),
        "auc_melody":          round(auc_mel,  4),
        "auc_product_jm":      round(auc_prod, 4),
        "auc_logistic_jm_cv":  round(auc_lr,   4),
        "logistic_fold_aucs":  [round(float(x), 4) for x in fold_aucs],
        "melody_gain":         round(melody_gain, 4),
        "decision":            decision,
        "positive_mean_jaccard": round(float(np.mean(jac[labels == 1])), 4),
        "negative_mean_jaccard": round(float(np.mean(jac[labels == 0])), 4),
        "positive_mean_melody":  round(float(np.mean(mel[labels == 1])), 4),
        "negative_mean_melody":  round(float(np.mean(mel[labels == 0])), 4),
        "all_data": [
            {
                "real_fid": int(d["real_fid"]), "ai_fid": int(d.get("ai_fid", 0)),
                "ai_file": d["ai_file"],
                "jaccard": round(float(d["jaccard"]), 4),
                "melody":  round(float(d["melody"]), 4),
                "label":   int(d["label"]),
            }
            for d in all_data
        ],
    }

    out_path = RESULTS_DIR / "melody_contour_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved → {out_path}")


if __name__ == "__main__":
    run()
