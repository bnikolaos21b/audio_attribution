#!/usr/bin/env python3
"""
Cross-Category MERT Similarity Investigation
==============================================
Computes mean cosine similarity for every combination of:
  - Track type: Real vs Real, Real vs AI, AI vs AI
  - Genre: same vs different
  - Relationship: unrelated vs derivative

BRUTALLY HONEST about what data is available and what is NOT.
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import json
import time
from pathlib import Path

import numpy as np
import librosa
import torch
from transformers import AutoProcessor, AutoModel

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent  # qwen/
PROJECT = BASE.parent
FEAT_CACHE = Path("/home/nikos_pc/Desktop/Orfium/orfium_audio_attribution_task/qwen/qwen/results/features")
PLOTS = FEAT_CACHE.parent / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

REAL_FOLK_DIR = Path("/home/nikos_pc/Desktop/Orfium/suno")  # Contains real_*.mp3 + Suno folk
SONICS_SCORES = PROJECT / "results" / "sonics" / "sonics_all_scores.csv"

MERT_MODEL = "m-a-p/MERT-v1-95M"
MERT_SR = 24_000
WINDOW_SEC = 10
HOP_SEC = 5
MIN_SEG_SEC = 3.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
    return np.mean(seg_embs, axis=0)


def cosine(a, b):
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))


# ── Data loading ──────────────────────────────────────────────────────────────

def load_available_data():
    """
    Returns all tracks we have MERT embeddings for, with metadata.
    
    What we actually have:
    1. 5 real folk songs (MERT cached from previous run)
    2. 26 Suno folk songs (MP3s on disk — need MERT extraction)
    3. SONICS 80 pair scores (pre-computed — no individual embeddings)
    4. MIPPIA 1294-dim features (on Colab only — NOT accessible locally)
    
    CRITICAL: No genre metadata exists in ANY data file.
    """
    import re
    
    tracks = {}  # name → {embedding, type, genre, notes}
    
    # ── Real folk songs ───────────────────────────────────────────────
    real_mp3s = sorted(REAL_FOLK_DIR.glob("real_*.mp3"))
    for f in real_mp3s:
        cached = FEAT_CACHE / f"{f.stem}_mert.npy"
        if cached.exists():
            emb = np.load(cached)
            tracks[f.name] = {
                "embedding": emb,
                "type": "real",
                "genre": "folk",  # All are folk music
                "source": "local_mp3",
                "notes": "Real folk recording",
            }
    
    # ── Suno folk songs ───────────────────────────────────────────────
    suno_mp3s = [f for f in sorted(REAL_FOLK_DIR.glob("*.mp3")) if not f.name.startswith("real_")]
    
    def canonical(path):
        return re.sub(r"\s*\(1\)\s*$", "", path.stem).strip()
    
    canon_map = {}
    for f in suno_mp3s:
        c = canonical(f)
        canon_map.setdefault(c, []).append(f.name)
    
    for f in suno_mp3s:
        cached = FEAT_CACHE / f"{f.stem}_mert.npy"
        c = canonical(f)
        pair_group = canon_map[c][0] if len(canon_map[c]) == 2 else None
        tracks[f.name] = {
            "embedding": None,  # Will be filled
            "type": "ai_suno",
            "genre": "folk",  # All generated as folk music
            "source": "local_mp3",
            "notes": f"Suno v4.5-all, prompt: '{c}'",
            "canonical": c,
            "pair_group": pair_group,  # Same-prompt pairs share this
        }
    
    return tracks


def extract_suno_embeddings(tracks, processor, model, device):
    """Extract MERT embeddings for Suno folk songs that don't have cached embeddings."""
    suno_dir = REAL_FOLK_DIR
    need_extraction = []
    
    for name, info in tracks.items():
        if info["embedding"] is None and info["type"] == "ai_suno":
            need_extraction.append(name)
    
    if not need_extraction:
        return
    
    print(f"\n── Extracting MERT for {len(need_extraction)} Suno folk songs ──")
    for i, name in enumerate(need_extraction):
        path = suno_dir / name
        t0 = time.time()
        cached = FEAT_CACHE / f"{path.stem}_mert.npy"
        if cached.exists():
            tracks[name]["embedding"] = np.load(cached)
            print(f"  [{i+1}/{len(need_extraction)}] [cached {time.time()-t0:.1f}s] {name}")
        else:
            segs = segment_audio(path)
            emb = extract_embedding(segs, processor, model, DEVICE)
            tracks[name]["embedding"] = emb
            FEAT_CACHE.mkdir(parents=True, exist_ok=True)
            np.save(cached, emb)
            print(f"  [{i+1}/{len(need_extraction)}] [{time.time()-t0:.1f}s] {name}")


# ── Category computation ──────────────────────────────────────────────────────

def compute_category(tracks, category_name, filter_fn):
    """Compute cosine similarities for a category of track pairs."""
    names = list(tracks.keys())
    sims = []
    pair_details = []
    
    for i, na in enumerate(names):
        for j, nb in enumerate(names):
            if i >= j:
                continue  # Upper triangle only
            if filter_fn(na, tracks[na], nb, tracks[nb]):
                s = cosine(tracks[na]["embedding"], tracks[nb]["embedding"])
                sims.append(s)
                pair_details.append({"a": na, "b": nb, "similarity": round(s, 5)})
    
    sims = np.array(sims) if sims else np.array([])
    return {
        "category": category_name,
        "n": int(len(sims)),
        "mean": round(float(np.mean(sims)), 5) if len(sims) > 0 else None,
        "std": round(float(np.std(sims)), 5) if len(sims) > 0 else None,
        "min": round(float(np.min(sims)), 5) if len(sims) > 0 else None,
        "max": round(float(np.max(sims)), 5) if len(sims) > 0 else None,
        "pairs": pair_details[:20],  # First 20 for inspection
    }


def compute_all_categories(tracks):
    """
    Compute all 12 categories where possible.
    
    Categories we CAN compute (with available data):
    - All our data is folk music, so "same genre" is the only genre we have
    - We have Real vs Real (unrelated), Real vs AI (unrelated and derivative via same-prompt), AI vs AI (unrelated and derivative)
    - We CANNOT compute "different genre" categories — all our data is folk
    - We CANNOT compute "same genre, with plagiarism" for Real vs Real — MIPPIA audio is on Colab
    """
    results = []
    names = list(tracks.keys())
    
    def get_pairs(type_combo, same_genre=None, related=None, same_prompt=None):
        """type_combo is a string like 'real_real', 'real_ai', 'ai_ai'."""
        parts = type_combo.split("_")
        type_a, type_b = parts[0], parts[1]
        sims = []
        details = []
        for i, na in enumerate(names):
            for j, nb in enumerate(names):
                if i >= j:
                    continue
                ta, tb = tracks[na], tracks[nb]
                
                # Type filter
                if type_a == "real" and ta["type"] != "real":
                    continue
                if type_b == "real" and tb["type"] != "real":
                    continue
                if type_a == "ai" and not ta["type"].startswith("ai"):
                    continue
                if type_b == "ai" and not tb["type"].startswith("ai"):
                    continue
                
                # Genre filter
                if same_genre is not None:
                    # All our data is folk, so same_genre=True always holds
                    if not same_genre:
                        continue  # We have no different-genre pairs
                
                # Relationship filter
                if related is not None:
                    if related:
                        # Derivative: same-prompt Suno pairs
                        if same_prompt is not None:
                            if same_prompt:
                                if ta.get("canonical") != tb.get("canonical") or ta.get("canonical") is None:
                                    continue
                            else:
                                if ta.get("canonical") == tb.get("canonical") and ta.get("canonical") is not None:
                                    continue
                        else:
                            # Any same-prompt
                            if ta.get("canonical") != tb.get("canonical") or ta.get("canonical") is None:
                                continue
                    else:
                        # Unrelated: different canonical prompts (or real vs anything)
                        if ta["type"] == "real" and tb["type"] == "real":
                            pass  # All real-real pairs are unrelated
                        elif ta["type"] == "real" and tb["type"].startswith("ai"):
                            pass  # Real vs AI are unrelated
                        elif ta["type"].startswith("ai") and tb["type"].startswith("ai"):
                            if ta.get("canonical") == tb.get("canonical") and ta.get("canonical") is not None:
                                continue  # Skip same-prompt
                else:
                    # All pairs regardless of relationship
                    pass
                
                s = cosine(ta["embedding"], tb["embedding"])
                sims.append(s)
                details.append({"a": na, "b": nb, "similarity": round(s, 5)})
        
        return np.array(sims) if sims else np.array([]), details
    
    def stats(sims, details):
        return {
            "n": int(len(sims)),
            "mean": round(float(np.mean(sims)), 5) if len(sims) > 0 else None,
            "std": round(float(np.std(sims)), 5) if len(sims) > 0 else None,
            "min": round(float(np.min(sims)), 5) if len(sims) > 0 else None,
            "max": round(float(np.max(sims)), 5) if len(sims) > 0 else None,
            "median": round(float(np.median(sims)), 5) if len(sims) > 0 else None,
            "pairs_sample": details[:10],
        }
    
    # ── Categories we CAN compute ─────────────────────────────────────
    
    # 1. Real vs Real, same genre, unrelated
    sims, details = get_pairs("real_real", same_genre=True, related=False)
    results.append({"category": "Real vs Real, same genre (folk), unrelated", **stats(sims, details)})
    
    # 5. Real vs AI, same genre, unrelated
    sims, details = get_pairs("real_ai", same_genre=True, related=False)
    results.append({"category": "Real vs AI (Suno), same genre (folk), unrelated", **stats(sims, details)})
    
    # 7. Real vs AI, same genre, derivative (same-prompt)
    sims, details = get_pairs("real_ai", same_genre=True, related=True, same_prompt=True)
    results.append({"category": "Real vs AI (Suno), same genre (folk), derivative (same-prompt)",
                    **stats(sims, details)})
    # Note: this will be empty — real vs AI can't be same-prompt since real has no prompt
    
    # Actually: "derivative" for Real vs AI means the AI was generated from this real song.
    # We don't have that relationship — the Suno songs were generated from text prompts,
    # not from our real folk songs.
    results[-1]["note"] = "EMPTY — Suno songs were generated from text prompts, not derived from our real folk songs"
    
    # 9. AI vs AI, same genre, unrelated
    sims, details = get_pairs("ai_ai", same_genre=True, related=False)
    results.append({"category": "AI (Suno) vs AI (Suno), same genre (folk), unrelated",
                    **stats(sims, details)})
    
    # 11. AI vs AI, same genre, derivative (same source/prompt)
    sims, details = get_pairs("ai_ai", same_genre=True, related=True, same_prompt=True)
    results.append({"category": "AI (Suno) vs AI (Suno), same genre (folk), derivative (same-prompt)",
                    **stats(sims, details)})
    
    # ── Categories we CANNOT compute ──────────────────────────────────
    
    cannot_compute = [
        {
            "category": "Real vs Real, different genre, unrelated",
            "reason": "All our real songs are folk music. No genre diversity in the 5 available real tracks.",
            "data_needed": "MIPPIA real tracks with genre labels — audio files are on Colab, not accessible locally.",
        },
        {
            "category": "Real vs Real, same genre, with plagiarism/copying",
            "reason": "MIPPIA has 39 positive (plagiarism/remake) pairs but the 1294-dim features are only on Colab. No MERT-only cache for MIPPIA tracks locally.",
            "data_needed": "MIPPIA audio files or MERT embeddings for the 60 ori + 60 comp tracks.",
        },
        {
            "category": "Real vs Real, different genre, with plagiarism/copying",
            "reason": "Same as above — MIPPIA data not accessible locally.",
            "data_needed": "MIPPIA audio with genre labels.",
        },
        {
            "category": "Real vs AI, different genre, unrelated",
            "reason": "All our data (both real and AI) is folk music. No cross-genre pairs available.",
            "data_needed": "Real songs + AI songs in different genres.",
        },
        {
            "category": "Real vs AI, same genre, derivative",
            "reason": "Our Suno folk songs were generated from text prompts, not from our real folk songs. There is no derivation relationship between them.",
            "data_needed": "AI songs that were actually generated from specific real source songs.",
        },
        {
            "category": "Real vs AI, different genre, derivative",
            "reason": "Same as above — no derivation relationship exists in our data.",
            "data_needed": "Cross-genre derivation pairs.",
        },
        {
            "category": "AI vs AI, different genre, unrelated",
            "reason": "All our AI songs are folk. SONICS has genre-diverse AI songs but individual embeddings are not accessible locally (only pair-level scores exist).",
            "data_needed": "Individual SONICS AI embeddings across genres.",
        },
        {
            "category": "AI vs AI, different genre, derivative (same source)",
            "reason": "All our AI songs are folk. SONICS has same-source pairs but we only have aggregate scores, not individual embeddings.",
            "data_needed": "Individual SONICS embeddings to compute cross-genre same-source similarity.",
        },
    ]
    
    for item in cannot_compute:
        results.append({**item, "n": 0, "mean": None, "std": None, "min": None, "max": None,
                        "median": None, "pairs_sample": [], "CANNOT_COMPUTE": True})
    
    return results


# ── SONICS score cross-reference ──────────────────────────────────────────────

def analyze_sonics_scores():
    """Analyze the 80 SONICS pair scores for what they tell us about categories."""
    import pandas as pd
    
    df = pd.read_csv(SONICS_SCORES)
    
    results = []
    
    # SONICS positives: same-source AI vs AI (derivative)
    pos = df[df['label'] == 1]
    # These are all same-generator per base_id
    for (ea, eb), group in pos.groupby(['engine_a', 'engine_b']):
        if ea == eb:
            results.append({
                "category": f"AI vs AI, SONICS {ea}-{eb}, same source (derivative)",
                "n": int(len(group)),
                "mean": round(float(group['raw_score'].mean()), 5),
                "std": round(float(group['raw_score'].std()), 5),
                "min": round(float(group['raw_score'].min()), 5),
                "max": round(float(group['raw_score'].max()), 5),
                "median": round(float(group['raw_score'].median()), 5),
                "note": "All SONICS positives are single-generator, same base_id pairs",
            })
    
    # SONICS negatives: unrelated AI vs AI or AI vs Real (cross)
    neg = df[df['label'] == 0]
    for (ea, eb), group in neg.groupby(['engine_a', 'engine_b']):
        results.append({
            "category": f"AI vs AI/Real, SONICS {ea}→{eb}, unrelated (negative)",
            "n": int(len(group)),
            "mean": round(float(group['raw_score'].mean()), 5),
            "std": round(float(group['raw_score'].std()), 5),
            "min": round(float(group['raw_score'].min()), 5),
            "max": round(float(group['raw_score'].max()), 5),
            "median": round(float(group['raw_score'].median()), 5),
            "note": f"SONICS negative pairs: {ea} vs {eb}",
        })
    
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  CROSS-CATEGORY MERT SIMILARITY INVESTIGATION")
    print("  Hard numbers. No speculation. Brutally honest.")
    print("=" * 72)
    
    # ── Load what we have ─────────────────────────────────────────────
    print("\n── Available data ──")
    tracks = load_available_data()
    
    real_tracks = {k: v for k, v in tracks.items() if v["type"] == "real"}
    ai_tracks = {k: v for k, v in tracks.items() if v["type"].startswith("ai")}
    
    print(f"  Real tracks with MERT embeddings: {len(real_tracks)}")
    for name in real_tracks:
        print(f"    {name}")
    print(f"  AI (Suno) tracks: {len(ai_tracks)} (need MERT extraction)")
    
    # ── Extract Suno embeddings ───────────────────────────────────────
    print("\n── Loading MERT model ──")
    processor = AutoProcessor.from_pretrained(MERT_MODEL, trust_remote_code=True)
    model = AutoModel.from_pretrained(MERT_MODEL, trust_remote_code=True).to(DEVICE)
    
    extract_suno_embeddings(tracks, processor, model, DEVICE)
    
    # Re-check
    real_tracks = {k: v for k, v in tracks.items() if v["type"] == "real" and v["embedding"] is not None}
    ai_tracks = {k: v for k, v in tracks.items() if v["type"].startswith("ai") and v["embedding"] is not None}
    
    print(f"\n  Real tracks with embeddings: {len(real_tracks)}")
    print(f"  AI tracks with embeddings: {len(ai_tracks)}")
    
    # ── Compute categories ────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  CATEGORY RESULTS")
    print("=" * 72)
    
    local_results = compute_all_categories(tracks)
    sonics_results = analyze_sonics_scores()
    
    all_results = local_results + sonics_results
    
    for r in all_results:
        print(f"\n  {'─' * 60}")
        print(f"  Category: {r['category']}")
        if r.get("CANNOT_COMPUTE"):
            print(f"  ❌ CANNOT COMPUTE")
            print(f"  Reason: {r['reason']}")
            print(f"  Data needed: {r.get('data_needed', 'N/A')}")
        else:
            if r["n"] == 0:
                print(f"  ⚠️  EMPTY (no pairs match this category)")
                if "note" in r:
                    print(f"  Note: {r['note']}")
            else:
                print(f"  n={r['n']}, mean={r['mean']}, std={r['std']}")
                print(f"  min={r['min']}, max={r['max']}, median={r.get('median')}")
                if "note" in r:
                    print(f"  Note: {r['note']}")
                if r.get("pairs_sample"):
                    print(f"  Sample pairs:")
                    for p in r["pairs_sample"][:5]:
                        print(f"    {p['a'][:40]:40s} ↔ {p['b'][:40]:40s}  sim={p['similarity']:.5f}")
    
    # ── Summary table ─────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  SUMMARY TABLE")
    print("=" * 72)
    
    print(f"\n  {'Category':<65s} {'n':>5} {'mean':>8} {'std':>8} {'Status'}")
    print(f"  {'─' * 100}")
    for r in all_results:
        cat = r['category'][:65]
        if r.get("CANNOT_COMPUTE"):
            print(f"  {cat:<65s} {'—':>5} {'—':>8} {'—':>8}  CANNOT COMPUTE")
        elif r["n"] == 0:
            print(f"  {cat:<65s} {0:>5} {'—':>8} {'—':>8}  EMPTY")
        else:
            print(f"  {cat:<65s} {r['n']:>5} {r['mean']:>8.5f} {r['std']:>8.5f}  OK")
    
    # ── What we can and cannot say ────────────────────────────────────
    print("\n" + "=" * 72)
    print("  HONEST ASSESSMENT")
    print("=" * 72)
    
    print("""
  COMPUTABLE (with current data):
  ✓ Real vs Real, same genre (folk), unrelated — 10 pairs from 5 real songs
  ✓ Real vs AI (Suno), same genre (folk), unrelated — 5×26 = 130 pairs
  ✓ AI vs AI (Suno), same genre (folk), unrelated — all different-prompt pairs
  ✓ AI vs AI (Suno), same genre (folk), derivative — 13 same-prompt pairs
  ✓ SONICS aggregate scores by engine combination (80 pairs, pre-computed)
  
  NOT COMPUTABLE:
  ✗ Any "different genre" category — ALL our data is folk music
  ✗ Real vs Real with plagiarism — MIPPIA audio is on Colab, not local
  ✗ Real vs AI derivative — our Suno songs were prompted, not derived from real
  ✗ Individual SONICS embeddings — only pair-level scores exist locally
  ✗ Cross-genre MIPPIA analysis — no genre metadata exists in ANY data file
  
  CRITICAL LIMITATION:
  There is NO genre metadata anywhere in the MIPPIA or SONICS data files.
  The MIPPIA descriptions contain song titles but no genre labels.
  The SONICS metadata has engine (suno/udio) and base_id but no genre.
  Our 31-song folk dataset is single-genre by construction.
  
  To compute the 12-category breakdown, we would need:
  1. Genre labels for MIPPIA tracks (doesn't exist in any file)
  2. Genre labels for SONICS tracks (doesn't exist in any file)
  3. Access to MIPPIA audio or embeddings (only on Colab)
  4. Access to individual SONICS embeddings (only aggregate scores locally)
""")
    
    # ── Save ──────────────────────────────────────────────────────────
    output = {
        "categories_computable": local_results,
        "sonics_aggregate_scores": sonics_results,
        "data_available": {
            "real_folk_with_mert": len(real_tracks),
            "suno_folk_with_mert": len(ai_tracks),
            "sonics_pair_scores": 80,
            "mippia_features": "ON COLAB ONLY (1294-dim, 127 tracks)",
            "genre_metadata": "NONE — does not exist in any data file",
        },
        "limitations": [
            "All available data is single-genre (folk)",
            "No genre metadata in MIPPIA or SONICS files",
            "MIPPIA audio/embeddings only on Colab",
            "SONICS individual embeddings not accessible locally",
            "No derivation relationship between our real and AI songs",
        ]
    }
    
    out_path = PLOTS / "cross_category_investigation.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved → {out_path}")
    
    print("\n" + "=" * 72)
    print("  INVESTIGATION COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
