#!/usr/bin/env python3
"""
Lyrics-Based Attribution Experiment
=====================================
Tests whether lyrics analysis can solve real→AI attribution that MERT fails at.

Results (128 positive + 192 negative real→AI pairs):
  Jaccard word overlap:           ROC AUC = 0.8856
  Sentence-Transformer cosine:    ROC AUC = 0.8749
  MERT Top-K (comparison):        ROC AUC = 0.5725  ← not used

Saved to: results/real_ai/lyrics_attribution_results.json

Usage:
  python src/lyrics_attribution.py
"""

import json
import os
import sys
import time
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

# ── Paths ─────────────────────────────────────────────────────────────────────
_SRC_DIR     = Path(__file__).parent
_PROJECT_DIR = _SRC_DIR.parent
PAIRS_DIR    = _PROJECT_DIR / "data" / "real_ai_pairs"
RESULTS_DIR  = _PROJECT_DIR / "results" / "real_ai"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load mapping (JSON metadata, not audio)
with open(PAIRS_DIR / "real_ai_pairs_mapping.json") as f:
    pairs = json.load(f)


# ── Audio helpers ─────────────────────────────────────────────────────────────

def get_real_path(fid):
    p = PAIRS_DIR / f"real_{fid}.mp3"
    return p if p.exists() else None


def get_ai_paths(fid):
    return sorted(PAIRS_DIR.glob(f"ai_{fid}_*.mp3"))


# ── Step 1: Whisper transcription ─────────────────────────────────────────────

TRANSCRIPT_CACHE = PAIRS_DIR / "transcripts.json"


def load_or_build_transcripts():
    if TRANSCRIPT_CACHE.exists():
        print(f"  Loading cached transcripts from {TRANSCRIPT_CACHE}")
        with open(TRANSCRIPT_CACHE) as f:
            return json.load(f)

    import whisper
    model = whisper.load_model("tiny")
    transcripts = {}

    # Collect all audio files
    all_files = {}
    for fid, info in pairs.items():
        rp = get_real_path(fid)
        if rp:
            all_files[f"real_{fid}"] = str(rp)
        for ap in get_ai_paths(fid):
            all_files[ap.stem] = str(ap)

    print(f"  Transcribing {len(all_files)} audio files with Whisper-tiny...")
    for i, (key, path) in enumerate(all_files.items()):
        result = model.transcribe(path, language="en")
        transcripts[key] = result["text"].strip()
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(all_files)}] done")

    with open(TRANSCRIPT_CACHE, "w") as f:
        json.dump(transcripts, f, indent=2)
    print(f"  Transcripts saved → {TRANSCRIPT_CACHE}")
    return transcripts


# ── Step 2: Build pairs and compute similarities ───────────────────────────────

def jaccard(t1, t2):
    w1 = set(t1.lower().split())
    w2 = set(t2.lower().split())
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)


def sentence_similarity(t1, t2, model):
    e1 = model.encode([t1])[0]
    e2 = model.encode([t2])[0]
    return float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-12))


def run():
    print("=" * 72)
    print("  LYRICS ATTRIBUTION EXPERIMENT")
    print("=" * 72)

    transcripts = load_or_build_transcripts()

    from sentence_transformers import SentenceTransformer
    sent_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Build complete pairs
    complete_pairs = {}
    for fid, info in pairs.items():
        rp = get_real_path(fid)
        aps = get_ai_paths(fid)
        if rp and aps:
            complete_pairs[fid] = {
                "real": rp, "ai": aps,
                "title": info["real_title"], "artist": info["real_artist"],
            }

    print(f"  Complete pairs: {len(complete_pairs)}")

    positive_data = []
    negative_data = []
    fids = list(complete_pairs.keys())

    for fid_a in fids:
        rk = f"real_{fid_a}"
        rt = transcripts.get(rk, "")
        if not rt:
            continue
        for ap in complete_pairs[fid_a]["ai"]:
            ak = ap.stem
            at = transcripts.get(ak, "")
            if not at:
                continue
            positive_data.append({
                "fid": int(fid_a),
                "title": complete_pairs[fid_a]["title"],
                "artist": complete_pairs[fid_a]["artist"],
                "ai_file": ak,
                "jaccard": round(jaccard(rt, at), 5),
                "sentence_sim": round(sentence_similarity(rt, at, sent_model), 5),
                "label": 1,
            })

    np.random.seed(42)
    for fid_a in fids:
        rk = f"real_{fid_a}"
        rt = transcripts.get(rk, "")
        if not rt:
            continue
        neg_fids = [x for x in fids if x != fid_a]
        neg_sample = np.random.choice(neg_fids, min(3, len(neg_fids)), replace=False)
        for fid_b in neg_sample:
            if not complete_pairs[fid_b]["ai"]:
                continue
            ap = complete_pairs[fid_b]["ai"][0]
            ak = ap.stem
            at = transcripts.get(ak, "")
            if not at:
                continue
            negative_data.append({
                "fid": int(fid_a),
                "ai_file": ak,
                "jaccard": round(jaccard(rt, at), 5),
                "sentence_sim": round(sentence_similarity(rt, at, sent_model), 5),
                "label": 0,
            })

    all_data = positive_data + negative_data
    labels   = np.array([d["label"]      for d in all_data])
    jac_vals = np.array([d["jaccard"]    for d in all_data])
    sen_vals = np.array([d["sentence_sim"] for d in all_data])

    auc_jac = roc_auc_score(labels, jac_vals)
    auc_sen = roc_auc_score(labels, sen_vals)

    print(f"\n  Results")
    print(f"  ─────────────────────────────────────────────────────────")
    print(f"  Positive pairs:              {len(positive_data)}")
    print(f"  Negative pairs:              {len(negative_data)}")
    print(f"  Jaccard AUC:                 {auc_jac:.4f}")
    print(f"  Sentence-Transformer AUC:    {auc_sen:.4f}")
    print(f"  MERT (reference, not used):  0.5725")

    output = {
        "jaccard": {
            "positive_mean": round(float(np.mean(jac_vals[labels == 1])), 4),
            "positive_std":  round(float(np.std(jac_vals[labels == 1])),  4),
            "negative_mean": round(float(np.mean(jac_vals[labels == 0])), 4),
            "negative_std":  round(float(np.std(jac_vals[labels == 0])),  4),
            "gap":           round(float(np.mean(jac_vals[labels == 1]) - np.mean(jac_vals[labels == 0])), 4),
            "roc_auc":       round(float(auc_jac), 4),
            "n_positive":    int((labels == 1).sum()),
            "n_negative":    int((labels == 0).sum()),
        },
        "sentence_transformer": {
            "positive_mean": round(float(np.mean(sen_vals[labels == 1])), 4),
            "positive_std":  round(float(np.std(sen_vals[labels == 1])),  4),
            "negative_mean": round(float(np.mean(sen_vals[labels == 0])), 4),
            "negative_std":  round(float(np.std(sen_vals[labels == 0])),  4),
            "gap":           round(float(np.mean(sen_vals[labels == 1]) - np.mean(sen_vals[labels == 0])), 4),
            "roc_auc":       round(float(auc_sen), 4),
            "n_positive":    int((labels == 1).sum()),
            "n_negative":    int((labels == 0).sum()),
        },
        "mert_comparison": {
            "mert_auc": 0.5725,
            "jaccard_auc": round(float(auc_jac), 4),
            "sentence_transformer_auc": round(float(auc_sen), 4),
        },
        "pair_details": [
            {k: v for k, v in d.items() if k != "label"}
            for d in positive_data
        ],
    }

    out_path = RESULTS_DIR / "lyrics_attribution_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved → {out_path}")


if __name__ == "__main__":
    run()
