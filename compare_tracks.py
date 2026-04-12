#!/usr/bin/env python3
"""
Audio Attribution
Usage:
  python compare_tracks.py <track_a> <track_b>              # auto mode (default)
  python compare_tracks.py <track_a> <track_b> --mode sonics # SONICS GMM (backward compat)
  python compare_tracks.py <track_a> <track_b> --device cuda  # GPU

Accepts audio files (.mp3, .wav, .flac) or pre-extracted feature files (.npy).

Modes
─────
  auto    5-stage pipeline: classify tracks → route to best calibration →
          MERT Top-K k=10 → calibrated score → honest explanation.
          Uses Lyrics Jaccard for real→AI pairs (AUC 0.89),
          SONICS GMM for AI/AI same-engine (AUC 0.85),
          MIPPIA min-max for real/real (AUC 0.84).

  sonics  Original MERT Top-K k=10 + GMM calibration (SONICS-only).
          Accepts .npy pre-extracted features.
          ROC AUC 0.850 on SONICS benchmark.
"""

# ── Self-bootstrapping venv ───────────────────────────────────────────────────
import sys
import os
import subprocess
from pathlib import Path


def _in_project_venv() -> bool:
    venv_dir = Path(__file__).parent / "venv"
    return Path(sys.prefix).resolve() == venv_dir.resolve()


def _bootstrap() -> None:
    project_root = Path(__file__).parent
    venv_dir     = project_root / "venv"
    if os.name != "nt":
        venv_python = venv_dir / "bin" / "python"
        venv_pip    = venv_dir / "bin" / "pip"
    else:
        venv_python = venv_dir / "Scripts" / "python.exe"
        venv_pip    = venv_dir / "Scripts" / "pip.exe"

    if not venv_dir.exists():
        print("First run — setting up environment (this takes ~2 minutes)...", flush=True)
        subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])
        subprocess.check_call([str(venv_python), "-m", "pip", "install", "--upgrade", "pip", "-q"])
        print("Installing PyTorch (CPU)...", flush=True)
        subprocess.check_call([
            str(venv_pip), "install", "torch", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cpu", "-q",
        ])
        print("Installing dependencies...", flush=True)
        subprocess.check_call([
            str(venv_pip), "install",
            "numpy", "librosa", "transformers", "scikit-learn", "pandas",
            "scipy", "pyloudnorm", "matplotlib",
            "openai-whisper", "sentence-transformers", "-q",
        ])
        print("Environment ready.\n", flush=True)

    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(str(venv_python), [str(venv_python)] + sys.argv)


if not _in_project_venv():
    _bootstrap()

# ── Imports (only reached inside the venv) ────────────────────────────────────
import time
import argparse
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class _Spinner:
    _FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, msg: str) -> None:
        self._msg  = msg
        self._stop = threading.Event()
        self._t    = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        i = 0
        while not self._stop.is_set():
            print(f"\r  {self._FRAMES[i % len(self._FRAMES)]} {self._msg}...", end="", flush=True)
            i += 1
            time.sleep(0.1)

    def __enter__(self):
        self._t.start(); return self

    def __exit__(self, *_):
        self._stop.set(); self._t.join()
        print("\r" + " " * (len(self._msg) + 10) + "\r", end="", flush=True)


def _ensure_mert_cache(track_path: str, cache_dir: str) -> None:
    """Pre-populate MERT cache for .npy inputs."""
    import numpy as np
    p = Path(track_path)
    if p.suffix.lower() != ".npy":
        return
    cache_path = Path(cache_dir) / f"{p.stem}_mert.npy"
    if cache_path.exists():
        return
    arr = np.load(str(p), mmap_mode="r")
    mert_slice = arr[:, :768].astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(cache_path), mert_slice)


def _run_auto(track_a, track_b, device, cache_dir):
    from src.general_attribution import attribute_tracks
    result = attribute_tracks(track_a, track_b, cache_dir=cache_dir, device=device, k=10)
    print()
    print(result.summary())
    return result


def _run_sonics(track_a, track_b, device, cache_dir):
    """Original SONICS Top-K MERT + GMM — kept for backward compatibility."""
    import numpy as np
    from scipy.stats import norm

    for p in [track_a, track_b]:
        _ensure_mert_cache(p, cache_dir)

    from src.sonics_similarity import compare_tracks_sonics_topk_gmm
    result  = compare_tracks_sonics_topk_gmm(track_a, track_b, k=10, cache_dir=cache_dir, device=device)
    score   = result["score"]
    verdict = result["verdict"]
    raw     = result["raw_score"]
    bar     = "█" * max(0, min(24, int(round(score * 24)))) + "░" * max(0, 24 - int(round(score * 24)))

    print()
    print("=" * 65)
    print(f"  Score:   {score:.4f}  [{bar}]  {score * 100:.1f}%")
    print(f"  Verdict: {verdict}")
    print(f"  Raw:     {raw:.4f}  →  P(derivative)={score:.4f}  [GMM]")
    print(f"  Method:  Top-K MERT k=10 + GMM calibration  (SONICS)")
    print("=" * 65)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Audio attribution — general-purpose (default) or SONICS-only."
    )
    parser.add_argument("track_a", help="First audio file or .npy features")
    parser.add_argument("track_b", help="Second audio file or .npy features")
    parser.add_argument("--mode",   choices=["auto", "sonics"], default="auto",
                        help="auto = general 5-stage pipeline (default); sonics = SONICS GMM only")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: auto-detect)")
    parser.add_argument("--cache-dir", default="features", dest="cache_dir",
                        help="Feature cache directory (default: features/)")
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

    mode_label = "General (5-stage)" if args.mode == "auto" else "SONICS (GMM)"
    print(f"\n  ── Mode: {mode_label} ──")

    fn = _run_auto if args.mode == "auto" else _run_sonics
    t0 = time.time()
    with _Spinner("Analyzing" if args.mode == "auto" else "Comparing"):
        result = fn(args.track_a, args.track_b, device, args.cache_dir)
    elapsed = time.time() - t0
    print(f"  Time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
