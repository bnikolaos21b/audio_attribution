#!/usr/bin/env python3
"""
SONICS All-vs-All MERT Similarity Experiment
==============================================
Downloads a sample of SONICS audio via HTTP range requests (remotezip),
extracts MERT embeddings, and computes ALL pairwise cosine similarities.

Answers: Why do same-source and different-source AI pairs compress to ~0.92-0.94?
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import time
import re
import json
from pathlib import Path

import numpy as np
import librosa
import torch
from transformers import AutoProcessor, AutoModel
from remotezip import RemoteZip

# ── Paths ─────────────────────────────────────────────────────────────────────
CACHE = Path("/home/nikos_pc/Desktop/Orfium/orfium_audio_attribution_task/qwen/results/sonics_sample")
CACHE.mkdir(parents=True, exist_ok=True)
PLOTS = CACHE.parent / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

# ── SONICS zip files on HuggingFace ──────────────────────────────────────────
# The audio is stored in 10 zip parts. Each part contains many MP3 files.
# We'll use remotezip to stream individual files via HTTP range requests.
SONICS_ZIP_URLS = [
    "https://huggingface.co/datasets/awsaf49/sonics/resolve/main/fake_songs/part_01.zip",
    "https://huggingface.co/datasets/awsaf49/sonics/resolve/main/fake_songs/part_02.zip",
    # We only need 1-2 parts to get a diverse sample
]

MERT_MODEL = "m-a-p/MERT-v1-95M"
MERT_SR = 24_000
WINDOW_SEC = 10
HOP_SEC = 5
MIN_SEG_SEC = 3.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

SAMPLE_PER_ZIP = 6  # Download 6 songs per zip part
MAX_TOTAL = 12      # Cap total downloads


def list_zip_contents(url):
    """List all MP3 files in a remote zip via HTTP range requests."""
    print(f"  Listing {url.split('/')[-1]}...")
    rz = RemoteZip(url)
    names = [n for n in rz.namelist() if n.endswith('.mp3')]
    rz.close()
    return names


def download_sample_from_zip(url, n=6):
    """Download n random MP3 files from a remote zip via range requests."""
    import random
    random.seed(42)
    
    names = list_zip_contents(url)
    if not names:
        print(f"  No MP3 files found!")
        return []
    
    # Separate suno and udio
    suno = [n for n in names if 'suno' in n.lower()]
    udio = [n for n in names if 'udio' in n.lower()]
    print(f"  Suno files: {len(suno)}, Udio files: {len(udio)}")
    
    # Sample evenly from both
    n_each = n // 2
    pick_suno = random.sample(suno, min(n_each, len(suno)))
    pick_udio = random.sample(udio, min(n_each, len(udio)))
    picks = pick_suno + pick_udio
    
    downloaded = []
    rz = RemoteZip(url)
    
    for i, name in enumerate(picks):
        local = CACHE / Path(name).name
        if local.exists():
            downloaded.append({"name": Path(name).name, "local": str(local), "zip_name": name})
            print(f"  [{i+1}/{len(picks)}] [cached] {local.name}")
            continue
        
        try:
            data = rz.read(name)
            local.write_bytes(data)
            downloaded.append({"name": Path(name).name, "local": str(local), "zip_name": name})
            print(f"  [{i+1}/{len(picks)}] [{len(data)/1e6:.1f}MB] {local.name}")
        except Exception as e:
            print(f"  [{i+1}/{len(picks)}] [FAIL] {name}: {e}")
    
    rz.close()
    return downloaded


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


def parse_metadata(filename):
    """Extract base_id, engine, variant from filename like fake_10003_suno_0.mp3"""
    m = re.match(r'fake_(\d+)_(suno|udio)_(\d+)\.mp3', filename, re.IGNORECASE)
    if m:
        return {
            "base_id": f"fake_{m.group(1)}",
            "engine": m.group(2).lower(),
            "variant": int(m.group(3)),
        }
    return None


def main():
    print("=" * 72)
    print("  SONICS ALL-vs-ALL MERT SIMILARITY EXPERIMENT")
    print("  Why do same-source and different-source pairs compress?")
    print("=" * 72)
    
    # ── 1. Download sample ──────────────────────────────────────────────
    print("\n── Step 1: Downloading SONICS audio sample via HTTP range requests ──")
    all_tracks = []
    
    for url in SONICS_ZIP_URLS:
        tracks = download_sample_from_zip(url, n=SAMPLE_PER_ZIP)
        all_tracks.extend(tracks)
        if len(all_tracks) >= MAX_TOTAL:
            break
    
    if not all_tracks:
        print("No tracks downloaded. Exiting.")
        return
    
    print(f"\nDownloaded {len(all_tracks)} tracks total:")
    for t in all_tracks:
        meta = parse_metadata(t["name"])
        print(f"  {t['name']:45s}  {meta}")
    
    # ── 2. Load MERT ────────────────────────────────────────────────────
    print("\n── Step 2: Loading MERT model ──")
    processor = AutoProcessor.from_pretrained(MERT_MODEL, trust_remote_code=True)
    model = AutoModel.from_pretrained(MERT_MODEL, trust_remote_code=True).to(DEVICE)
    
    # ── 3. Extract embeddings ───────────────────────────────────────────
    print("\n── Step 3: Extracting MERT embeddings ──")
    
    file_embs = {}  # filename → (768,)
    seg_embs = {}   # filename → (n_seg, 768)
    
    for i, track in enumerate(all_tracks):
        cached = CACHE / f"{Path(track['name']).stem}_mert.npy"
        cached_seg = CACHE / f"{Path(track['name']).stem}_mert_segs.npy"
        if cached.exists() and cached_seg.exists():
            file_embs[track["name"]] = np.load(cached)
            seg_embs[track["name"]] = np.load(cached_seg)
            print(f"  [{i+1}/{len(all_tracks)}] [cached] {track['name']}")
        else:
            t0 = time.time()
            segments = segment_audio(track["local"])
            f_emb, s_emb = extract_embedding(segments, processor, model, DEVICE)
            file_embs[track["name"]] = f_emb
            seg_embs[track["name"]] = s_emb
            np.save(cached, f_emb)
            np.save(cached_seg, s_emb)
            print(f"  [{i+1}/{len(all_tracks)}] [{time.time()-t0:.1f}s] {track['name']} ({len(segments)} segs)")
    
    # ── 4. Group by engine and base_id ──────────────────────────────────
    names = list(file_embs.keys())
    metas = {n: parse_metadata(n) for n in names}
    
    suno_names = [n for n in names if metas[n] and metas[n]["engine"] == "suno"]
    udio_names = [n for n in names if metas[n] and metas[n]["engine"] == "udio"]
    
    # Check for same base_id pairs
    base_id_groups = {}
    for n in names:
        m = metas[n]
        if m:
            base_id_groups.setdefault(m["base_id"], []).append(n)
    
    same_base_pairs = []
    diff_base_pairs = []
    for n1 in names:
        for n2 in names:
            if n1 >= n2:
                continue
            m1, m2 = metas[n1], metas[n2]
            if m1 and m2 and m1["base_id"] == m2["base_id"]:
                same_base_pairs.append((n1, n2))
            else:
                diff_base_pairs.append((n1, n2))
    
    print(f"\n  Total tracks: {len(names)}")
    print(f"  Suno: {len(suno_names)}, Udio: {len(udio_names)}")
    print(f"  Same base_id pairs: {len(same_base_pairs)}")
    print(f"  Different base_id pairs: {len(diff_base_pairs)}")
    print(f"  Total pairs: {len(same_base_pairs) + len(diff_base_pairs)}")
    
    # ── 5. ALL pairwise cosine similarities ─────────────────────────────
    print("\n" + "=" * 72)
    print("  STEP 5: ALL PAIRWISE COSINE SIMILARITIES")
    print("=" * 72)
    
    all_sims = []
    same_base_sims = []
    diff_base_sims = []
    same_engine_sims = []
    diff_engine_sims = []
    
    same_base_details = []
    diff_base_details = []
    
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if i >= j:
                continue
            s = cosine(file_embs[n1], file_embs[n2])
            all_sims.append(s)
            
            m1, m2 = metas[n1], metas[n2]
            same_base = m1 and m2 and m1["base_id"] == m2["base_id"]
            same_eng = m1 and m2 and m1["engine"] == m2["engine"]
            
            if same_base:
                same_base_sims.append(s)
                same_base_details.append({"a": n1, "b": n2, "sim": round(s, 5),
                                          "engine_a": m1["engine"], "engine_b": m2["engine"]})
            else:
                diff_base_sims.append(s)
                diff_base_details.append({"a": n1, "b": n2, "sim": round(s, 5),
                                          "engine_a": m1["engine"] if m1 else "?", 
                                          "engine_b": m2["engine"] if m2 else "?"})
            
            if same_eng:
                same_engine_sims.append(s)
            else:
                diff_engine_sims.append(s)
    
    all_sims = np.array(all_sims)
    same_base_sims = np.array(same_base_sims)
    diff_base_sims = np.array(diff_base_sims)
    same_engine_sims = np.array(same_engine_sims)
    diff_engine_sims = np.array(diff_engine_sims)
    
    def stats(arr, label):
        if len(arr) == 0:
            print(f"\n  {label}: NO PAIRS")
            return
        print(f"\n  {label} (n={len(arr)}):")
        print(f"    mean   = {np.mean(arr):.5f}")
        print(f"    std    = {np.std(arr):.5f}")
        print(f"    min    = {np.min(arr):.5f}")
        print(f"    max    = {np.max(arr):.5f}")
        print(f"    median = {np.median(arr):.5f}")
        print(f"    range  = {np.max(arr) - np.min(arr):.5f}")
        print(f"    var    = {np.var(arr):.6f}")
    
    stats(all_sims, "ALL pairs (any combination)")
    stats(same_base_sims, "SAME BASE ID (positive / derivative pairs)")
    stats(diff_base_sims, "DIFFERENT BASE ID (negative / unrelated pairs)")
    stats(same_engine_sims, "SAME ENGINE (Suno-Suno or Udio-Udio)")
    stats(diff_engine_sims, "DIFFERENT ENGINE (Suno-Udio or Udio-Suno)")
    
    # The key comparison
    if len(same_base_sims) > 0 and len(diff_base_sims) > 0:
        gap = np.mean(same_base_sims) - np.mean(diff_base_sims)
        print(f"\n  ═══ KEY RESULT ═══")
        print(f"  Same-source mean:  {np.mean(same_base_sims):.5f}")
        print(f"  Diff-source mean:  {np.mean(diff_base_sims):.5f}")
        print(f"  GAP:               {gap:+.5f}")
        print(f"  Gap as % of mean:  {abs(gap)/np.mean(all_sims)*100:.1f}%")
        print()
        
        # Effect size
        pooled_std = np.sqrt((np.var(same_base_sims, ddof=1) + np.var(diff_base_sims, ddof=1)) / 2)
        cohen_d = gap / (pooled_std + 1e-12)
        print(f"  Cohen's d: {cohen_d:.3f}")
        if abs(cohen_d) < 0.2:
            print(f"  → Negligible effect size")
        elif abs(cohen_d) < 0.5:
            print(f"  → Small effect size")
        elif abs(cohen_d) < 0.8:
            print(f"  → Medium effect size")
        else:
            print(f"  → Large effect size")
    
    # ── 6. Per-track breakdown ──────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  STEP 6: PER-TRACK BREAKDOWN")
    print("=" * 72)
    
    print("\n  Same-base pairs detail:")
    for d in same_base_details:
        print(f"    {d['a']:45s} ↔ {d['b']:45s}  sim={d['sim']:.5f}  ({d['engine_a']}-{d['engine_b']})")
    
    # For each track, show its average similarity to same-engine vs diff-engine
    print(f"\n  Per-track average similarity:")
    print(f"  {'Track':45s}  {'Same-engine':>12s}  {'Diff-engine':>12s}  {'n_same':>8s}  {'n_diff':>8s}")
    print(f"  {'─' * 100}")
    for n1 in sorted(names):
        same_eng_sims_t = []
        diff_eng_sims_t = []
        for n2 in names:
            if n1 == n2:
                continue
            s = cosine(file_embs[n1], file_embs[n2])
            m1, m2 = metas[n1], metas[n2]
            if m1 and m2 and m1["engine"] == m2["engine"]:
                same_eng_sims_t.append(s)
            else:
                diff_eng_sims_t.append(s)
        print(f"  {n1:45s}  {np.mean(same_eng_sims_t) if same_eng_sims_t else 0:>12.5f}  "
              f"{np.mean(diff_eng_sims_t) if diff_eng_sims_t else 0:>12.5f}  "
              f"{len(same_eng_sims_t):>8d}  {len(diff_eng_sims_t):>8d}")
    
    # ── 7. Segment-level analysis for one same-base pair ────────────────
    print("\n" + "=" * 72)
    print("  STEP 7: SEGMENT-LEVEL ANALYSIS (same-base pair)")
    print("=" * 72)
    
    if same_base_details and seg_embs:
        d = same_base_details[0]
        n1, n2 = d["a"], d["b"]
        s1 = seg_embs[n1]
        s2 = seg_embs[n2]
        
        s1_n = s1 / (np.linalg.norm(s1, axis=1, keepdims=True) + 1e-12)
        s2_n = s2 / (np.linalg.norm(s2, axis=1, keepdims=True) + 1e-12)
        sim_mat = s1_n @ s2_n.T
        
        print(f"\n  {n1} ({s1.shape[0]} segs) vs {n2} ({s2.shape[0]} segs):")
        print(f"  Same base_id: {metas[n1]['base_id']}")
        print(f"  Segment sim matrix: mean={np.mean(sim_mat):.5f} std={np.std(sim_mat):.5f}")
        print(f"  min={np.min(sim_mat):.5f}  max={np.max(sim_mat):.5f}")
        print(f"  median={np.median(sim_mat):.5f}")
        print(f"  Top-10 mean: {np.mean(np.sort(sim_mat, axis=None)[-min(10, sim_mat.size):]):.5f}")
        
        # Compare to a random different-base pair
        if diff_base_details:
            rd = diff_base_details[0]
            rn1, rn2 = rd["a"], rd["b"]
            rs1 = seg_embs[rn1]
            rs2 = seg_embs[rn2]
            rs1_n = rs1 / (np.linalg.norm(rs1, axis=1, keepdims=True) + 1e-12)
            rs2_n = rs2 / (np.linalg.norm(rs2, axis=1, keepdims=True) + 1e-12)
            r_sim_mat = rs1_n @ rs2_n.T
            
            print(f"\n  COMPARISON: {rn1} ({rs1.shape[0]} segs) vs {rn2} ({rs2.shape[0]} segs):")
            print(f"  Different base_id")
            print(f"  Segment sim matrix: mean={np.mean(r_sim_mat):.5f} std={np.std(r_sim_mat):.5f}")
            print(f"  min={np.min(r_sim_mat):.5f}  max={np.max(r_sim_mat):.5f}")
            print(f"  median={np.median(r_sim_mat):.5f}")
            print(f"  Top-10 mean: {np.mean(np.sort(r_sim_mat, axis=None)[-min(10, r_sim_mat.size):]):.5f}")
    
    # ── 8. Plot ─────────────────────────────────────────────────────────
    print("\n  ── Plotting ──")
    plot_all(all_sims, same_base_sims, diff_base_sims, same_engine_sims, diff_engine_sims,
             same_base_details, diff_base_details, names, metas, file_embs)
    
    # ── 9. Save ─────────────────────────────────────────────────────────
    output = {
        "n_tracks": len(names),
        "n_suno": len(suno_names),
        "n_udio": len(udio_names),
        "all_pairs": {
            "mean": round(float(np.mean(all_sims)), 5),
            "std": round(float(np.std(all_sims)), 5),
            "min": round(float(np.min(all_sims)), 5),
            "max": round(float(np.max(all_sims)), 5),
            "n": int(len(all_sims)),
        },
        "same_base": {
            "mean": round(float(np.mean(same_base_sims)), 5) if len(same_base_sims) > 0 else None,
            "std": round(float(np.std(same_base_sims)), 5) if len(same_base_sims) > 0 else None,
            "n": int(len(same_base_sims)),
            "pairs": same_base_details,
        },
        "diff_base": {
            "mean": round(float(np.mean(diff_base_sims)), 5) if len(diff_base_sims) > 0 else None,
            "std": round(float(np.std(diff_base_sims)), 5) if len(diff_base_sims) > 0 else None,
            "n": int(len(diff_base_sims)),
        },
        "same_engine": {
            "mean": round(float(np.mean(same_engine_sims)), 5),
            "std": round(float(np.std(same_engine_sims)), 5),
            "n": int(len(same_engine_sims)),
        },
        "diff_engine": {
            "mean": round(float(np.mean(diff_engine_sims)), 5),
            "std": round(float(np.std(diff_engine_sims)), 5),
            "n": int(len(diff_engine_sims)),
        },
        "gap": round(float(np.mean(same_base_sims) - np.mean(diff_base_sims)), 5) if len(same_base_sims) > 0 and len(diff_base_sims) > 0 else None,
    }
    
    out_json = PLOTS / "sonics_allpairwise_similarity.json"
    with open(out_json, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"  Saved → {out_json}")
    
    print("\n" + "=" * 72)
    print("  EXPERIMENT COMPLETE")
    print("=" * 72)


def plot_all(all_sims, same_base, diff_base, same_eng, diff_eng,
             same_details, diff_details, names, metas, file_embs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    fig.suptitle("SONICS All-vs-All MERT Cosine Similarity\n"
                 f"n={len(names)} tracks, {len(all_sims)} pairs",
                 fontsize=14, fontweight="bold")
    
    # Panel 1: All pairs histogram
    ax = axes[0, 0]
    ax.hist(all_sims, bins=30, alpha=0.7, color="#2C3E50", edgecolor="black")
    ax.axvline(np.mean(all_sims), color="red", ls="--", lw=2, label=f"Mean: {np.mean(all_sims):.4f}")
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Count")
    ax.set_title("All Pairwise Similarities")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Same-base vs different-base
    ax = axes[0, 1]
    if len(same_base) > 0:
        ax.hist(same_base, bins=20, alpha=0.5, label=f"Same base (n={len(same_base)})", color="#27AE60", density=True)
    ax.hist(diff_base, bins=20, alpha=0.5, label=f"Diff base (n={len(diff_base)})", color="#E74C3C", density=True)
    if len(same_base) > 0:
        ax.axvline(np.mean(same_base), color="#27AE60", ls="--", lw=1.5, label=f"Same: {np.mean(same_base):.4f}")
    ax.axvline(np.mean(diff_base), color="#E74C3C", ls="--", lw=1.5, label=f"Diff: {np.mean(diff_base):.4f}")
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Density")
    ax.set_title("Same-Source vs Different-Source")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Same-engine vs different-engine
    ax = axes[0, 2]
    ax.hist(same_eng, bins=20, alpha=0.5, label=f"Same engine (n={len(same_eng)})", color="#2980B9", density=True)
    ax.hist(diff_eng, bins=20, alpha=0.5, label=f"Diff engine (n={len(diff_eng)})", color="#E67E22", density=True)
    ax.axvline(np.mean(same_eng), color="#2980B9", ls="--", lw=1.5, label=f"Same: {np.mean(same_eng):.4f}")
    ax.axvline(np.mean(diff_eng), color="#E67E22", ls="--", lw=1.5, label=f"Diff: {np.mean(diff_eng):.4f}")
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Density")
    ax.set_title("Same-Engine vs Different-Engine")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Panel 4: Similarity matrix
    ax = axes[1, 0]
    n = len(names)
    sim_mat = np.zeros((n, n))
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            sim_mat[i, j] = cosine(file_embs[n1], file_embs[n2])
    
    # Color by engine
    colors = []
    for n1 in names:
        row_colors = []
        for n2 in names:
            m1, m2 = metas[n1], metas[n2]
            if m1 and m2 and m1["base_id"] == m2["base_id"]:
                row_colors.append(1.0)  # Same base
            elif m1 and m2 and m1["engine"] == m2["engine"]:
                row_colors.append(0.6)  # Same engine
            else:
                row_colors.append(0.0)  # Different engine
        colors.append(row_colors)
    colors = np.array(colors)
    
    # Show similarity with overlay
    im = ax.imshow(sim_mat, cmap="YlOrRd", vmin=0.8, vmax=1.0, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    short_names = [Path(x).name[:12] for x in names]
    ax.set_xticklabels(short_names, rotation=90, fontsize=6)
    ax.set_yticklabels(short_names, fontsize=6)
    plt.colorbar(im, ax=ax, label="Cosine Similarity")
    ax.set_title("Full Similarity Matrix")
    
    # Panel 5: Per-track stats
    ax = axes[1, 1]
    track_means = []
    track_stds = []
    track_labels = []
    for n1 in sorted(names):
        sims = []
        for n2 in names:
            if n1 == n2:
                continue
            sims.append(cosine(file_embs[n1], file_embs[n2]))
        track_means.append(np.mean(sims))
        track_stds.append(np.std(sims))
        track_labels.append(Path(n1).name[:18])
    
    x = np.arange(len(track_means))
    ax.barh(x, track_means, xerr=track_stds, capsize=3, alpha=0.7, color="#3498DB")
    ax.set_yticks(x)
    ax.set_yticklabels(track_labels, fontsize=6)
    ax.set_xlabel("Mean Cosine Similarity to All Other Tracks")
    ax.set_title("Per-Track Similarity (±1 std)")
    ax.set_xlim(0.85, 1.0)
    ax.grid(True, axis="x", alpha=0.3)
    
    # Panel 6: Box plot by category
    ax = axes[1, 2]
    categories = []
    cat_labels = []
    cat_colors = []
    if len(same_base) > 0:
        categories.append(same_base)
        cat_labels.append("Same\nBase")
        cat_colors.append("#27AE60")
    categories.append(diff_base)
    cat_labels.append("Diff\nBase")
    cat_colors.append("#E74C3C")
    categories.append(same_eng)
    cat_labels.append("Same\nEngine")
    cat_colors.append("#2980B9")
    categories.append(diff_eng)
    cat_labels.append("Diff\nEngine")
    cat_colors.append("#E67E22")
    
    bp = ax.boxplot(categories, labels=cat_labels, vert=True, patch_artist=True,
                    showmeans=True, meanline=True)
    for patch, col in zip(bp['boxes'], cat_colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.6)
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Distribution by Category")
    ax.grid(True, axis="y", alpha=0.3)
    
    plt.tight_layout()
    out_png = PLOTS / "sonics_allpairwise_similarity.png"
    plt.savefig(str(out_png), dpi=150, bbox_inches="tight")
    print(f"  Plot saved → {out_png}")


if __name__ == "__main__":
    main()
