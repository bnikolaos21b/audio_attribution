#!/usr/bin/env python3
"""
Audio Attribution — Top-K MERT k=10
Usage: python compare_tracks.py <track_a> <track_b> [--device cpu|cuda]

Accepts audio files (.mp3, .wav, .flac) or pre-extracted feature files (.npy).
"""

# ── Self-bootstrapping venv ───────────────────────────────────────────────────
# If not inside the project venv, create it, install dependencies, re-launch.
import sys
import os
import subprocess
from pathlib import Path


def _in_project_venv():
    venv_dir = Path(__file__).parent / "venv"
    return Path(sys.prefix).resolve() == venv_dir.resolve()


def _bootstrap():
    """Create venv, install deps, re-launch inside it."""
    project_root = Path(__file__).parent
    venv_dir = project_root / "venv"
    if os.name != "nt":
        venv_python = venv_dir / "bin" / "python"
        venv_pip = venv_dir / "bin" / "pip"
    else:
        venv_python = venv_dir / "Scripts" / "python.exe"
        venv_pip = venv_dir / "Scripts" / "pip.exe"

    if not venv_dir.exists():
        print("First run — setting up environment (this takes ~2 minutes)...", flush=True)
        subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])
        subprocess.check_call(
            [str(venv_python), "-m", "pip", "install", "--upgrade", "pip", "-q"]
        )
        print("Installing PyTorch (CPU)...", flush=True)
        subprocess.check_call([
            str(venv_pip), "install", "torch", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cpu", "-q",
        ])
        print("Installing dependencies...", flush=True)
        subprocess.check_call([
            str(venv_pip), "install",
            "numpy", "librosa", "transformers", "scikit-learn", "pandas", "-q",
        ])
        print("Environment ready.\n", flush=True)

    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(str(venv_python), [str(venv_python)] + sys.argv)


if not _in_project_venv():
    _bootstrap()

# ── Imports (only reached inside the venv) ───────────────────────────────────
import time
import argparse
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SONICS_LO = 0.922
SONICS_HI = 0.9336


def _ensure_mert_cache(track_path: str, cache_dir: str) -> None:
    """Pre-populate MERT cache for .npy inputs so _get_mert_features never
    calls _load_audio() on a feature array."""
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


class _Spinner:
    _FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, msg: str) -> None:
        self._msg = msg
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        i = 0
        while not self._stop.is_set():
            frame = self._FRAMES[i % len(self._FRAMES)]
            print(f"\r  {frame} {self._msg}...", end="", flush=True)
            i += 1
            time.sleep(0.1)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join()
        print("\r" + " " * (len(self._msg) + 10) + "\r", end="", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Audio attribution — Top-K MERT k=10 (ROC AUC 0.850)."
    )
    parser.add_argument("track_a", help="First audio file (.mp3/.wav/.flac) or .npy features")
    parser.add_argument("track_b", help="Second audio file (.mp3/.wav/.flac) or .npy features")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: auto-detect)")
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

    cache_dir = "features"
    os.makedirs(cache_dir, exist_ok=True)

    _ensure_mert_cache(args.track_a, cache_dir)
    _ensure_mert_cache(args.track_b, cache_dir)

    with _Spinner("Comparing"):
        t0 = time.time()
        from src.sonics_similarity import compare_tracks_sonics_topk
        result = compare_tracks_sonics_topk(
            args.track_a, args.track_b, k=10, cache_dir=cache_dir, device=device
        )
        elapsed = time.time() - t0

    score = result["score"]
    verdict = result["verdict"]
    raw = result["raw_score"]
    bar = "█" * max(0, min(24, int(round(score * 24)))) + "░" * max(0, min(24, 24 - int(round(score * 24))))

    print()
    print("=" * 65)
    print(f"  Score:   {score:.4f}  [{bar}]  {score * 100:.1f}%")
    print(f"  Verdict: {verdict}")
    print(f"  Raw:     {raw:.4f}  (lo={SONICS_LO:.4f}  hi={SONICS_HI:.4f})")
    print(f"  Method:  Top-K MERT k=10  ({elapsed:.1f}s)")
    print("=" * 65)


if __name__ == "__main__":
    main()
