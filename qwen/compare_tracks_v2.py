#!/usr/bin/env python3
"""
Audio Attribution v2 — Multi-Stage Pipeline
=============================================
Usage:
  python compare_tracks_v2.py <track_a> <track_b>                    # General mode (default)
  python compare_tracks_v2.py <track_a> <track_b> --mode sonics       # Original SONICS-only mode
  python compare_tracks_v2.py <track_a> <track_b> --device cuda       # Force GPU

General mode: 5-stage pipeline (classify → route → MERT → calibrate → explain)
SONICS mode: Original Top-K MERT + GMM calibration (backward compatible)
"""

import sys
import os
import argparse
import time
import threading
from pathlib import Path


def _spinner_thread(msg: str, stop_event: threading.Event):
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    i = 0
    while not stop_event.is_set():
        frame = frames[i % len(frames)]
        print(f"\r  {frame} {msg}...", end="", flush=True)
        i += 1
        time.sleep(0.1)
    print("\r" + " " * (len(msg) + 10) + "\r", end="", flush=True)


def run_general_mode(track_a, track_b, device, cache_dir):
    """Run the new general-purpose attribution system."""
    from src.general_attribution import attribute_tracks

    result = attribute_tracks(track_a, track_b, cache_dir=cache_dir, device=device, k=10)
    print()
    print(result.summary())
    return result


def run_sonics_mode(track_a, track_b, device, cache_dir):
    """Run the original SONICS-only Top-K MERT + GMM calibration."""
    import numpy as np
    from scipy.stats import norm
    proj_root = Path(__file__).parent.parent
    
    # Pre-populate cache for .npy inputs
    for path in [track_a, track_b]:
        p = Path(path)
        if p.suffix.lower() == ".npy":
            cache_path = Path(cache_dir) / f"{p.stem}_mert.npy"
            if not cache_path.exists():
                arr = np.load(str(p), mmap_mode="r")
                mert_slice = arr[:, :768].astype(np.float32)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(str(cache_path), mert_slice)
    
    # Load MERT segments and compute Top-K similarity inline
    # (avoids complex import chain from original sonics_similarity.py)
    def _load_mert_segs(path):
        p = Path(path)
        if p.suffix.lower() == ".npy":
            cache = Path(cache_dir) / f"{p.stem}_mert.npy"
            if cache.exists():
                return np.load(cache)
            arr = np.load(path, mmap_mode="r")
            return arr[:, :768].astype(np.float32)
        return None
    
    seg_a = _load_mert_segs(track_a)
    seg_b = _load_mert_segs(track_b)
    
    if seg_a is None or seg_b is None:
        print("Error: .npy feature files required for SONICS mode")
        sys.exit(1)
    
    # L2 normalize
    a_norm = seg_a / (np.linalg.norm(seg_a, axis=1, keepdims=True) + 1e-12)
    b_norm = seg_b / (np.linalg.norm(seg_b, axis=1, keepdims=True) + 1e-12)
    sim_matrix = a_norm @ b_norm.T
    
    # Top-K k=10
    top1 = np.max(sim_matrix, axis=1)
    top10 = np.sort(top1)[-10:]
    raw_score = float(np.mean(top10))
    
    # GMM calibration (from results/calibration/gmm_calibration_params.json)
    mu_pos, sigma_pos = 0.935767, 0.018309
    mu_neg, sigma_neg = 0.924318, 0.017772
    pi_pos = 0.5
    
    like_pos = norm.pdf(raw_score, mu_pos, sigma_pos)
    like_neg = norm.pdf(raw_score, mu_neg, sigma_neg)
    cal_score = float(pi_pos * like_pos / (pi_pos * like_pos + (1 - pi_pos) * like_neg + 1e-12))
    
    if cal_score >= 0.75:
        verdict = "LIKELY DERIVATIVE"
    elif cal_score >= 0.55:
        verdict = "POSSIBLY RELATED"
    elif cal_score >= 0.35:
        verdict = "WEAK SIMILARITY"
    else:
        verdict = "UNRELATED"

    bar = "█" * max(0, min(24, int(round(cal_score * 24)))) + "░" * max(0, min(24, 24 - int(round(cal_score * 24))))
    print()
    print("=" * 65)
    print(f"  Score:   {cal_score:.4f}  [{bar}]  {cal_score * 100:.1f}%")
    print(f"  Verdict: {verdict}")
    print(f"  Raw:     {raw_score:.4f}  →  P(derivative)={cal_score:.4f}  [GMM]")
    print(f"  Method:  Top-K MERT k=10 + GMM calibration  (SONICS)")
    print("=" * 65)
    return {"score": cal_score, "verdict": verdict, "raw_score": raw_score}


def main():
    parser = argparse.ArgumentParser(
        description="Audio Attribution v2 — General-purpose + SONICS modes"
    )
    parser.add_argument("track_a", help="First audio file (.mp3/.wav/.flac) or .npy features")
    parser.add_argument("track_b", help="Second audio file (.mp3/.wav/.flac) or .npy features")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: auto-detect)")
    parser.add_argument("--mode", choices=["general", "sonics"], default="general",
                        help="Attribution mode: general (5-stage) or sonics (original)")
    parser.add_argument("--cache-dir", default="features", help="Feature cache directory")
    args = parser.parse_args()

    for path, label in [(args.track_a, "track_a"), (args.track_b, "track_b")]:
        if not os.path.exists(path):
            print(f"Error: {label} not found: {path!r}", file=sys.stderr)
            sys.exit(1)

    device = args.device
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    os.makedirs(args.cache_dir, exist_ok=True)

    if args.mode == "sonics":
        print(f"\n  ── Mode: SONICS (original Top-K MERT + GMM) ──")
        stop = threading.Event()
        t = threading.Thread(target=_spinner_thread, args=("Comparing", stop), daemon=True)
        t.start()
        try:
            result = run_sonics_mode(args.track_a, args.track_b, device, args.cache_dir)
        finally:
            stop.set()
            t.join()
    else:
        print(f"\n  ── Mode: General (5-stage pipeline) ──")
        stop = threading.Event()
        t = threading.Thread(target=_spinner_thread, args=("Analyzing", stop), daemon=True)
        t.start()
        try:
            result = run_general_mode(args.track_a, args.track_b, device, args.cache_dir)
        finally:
            stop.set()
            t.join()


if __name__ == "__main__":
    main()
