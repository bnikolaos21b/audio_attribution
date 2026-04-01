"""
compare_tracks_api — public entry points for audio attribution.

    compare_tracks(song_id_1, song_id_2)       ← brief-specified interface
    compare_tracks_from_paths(path_a, path_b)  ← path-based interface

Method: Top-K MERT k=10 (ROC AUC 0.850).
"""

try:
    from src.registry import TRACK_REGISTRY
except ImportError:
    from registry import TRACK_REGISTRY


def compare_tracks(song_id_1: str, song_id_2: str, **kwargs) -> dict:
    """
    Compare two tracks by song ID.  Required interface per assessment brief.

    IDs are resolved to file paths via TRACK_REGISTRY (src/registry.py).

    Args:
        song_id_1: registered ID for the first track  (e.g. "001_ori")
        song_id_2: registered ID for the second track (e.g. "001_comp")
        **kwargs:  forwarded to compare_tracks_from_paths (features_dir, device)

    Returns:
        dict with keys: score (float), verdict (str), track_a (str), track_b (str)

    Raises:
        KeyError: if either ID is not in TRACK_REGISTRY
    """
    if song_id_1 not in TRACK_REGISTRY:
        raise KeyError(
            f"Unknown song_id_1: {song_id_1!r}. "
            f"Registered IDs: {sorted(TRACK_REGISTRY)}"
        )
    if song_id_2 not in TRACK_REGISTRY:
        raise KeyError(
            f"Unknown song_id_2: {song_id_2!r}. "
            f"Registered IDs: {sorted(TRACK_REGISTRY)}"
        )
    return compare_tracks_from_paths(
        TRACK_REGISTRY[song_id_1],
        TRACK_REGISTRY[song_id_2],
        **kwargs,
    )


def compare_tracks_from_paths(
    track_a: str,
    track_b: str,
    features_dir: str = "features",
    device: str = "cpu",
) -> dict:
    """
    Compare two audio tracks using Top-K MERT k=10.

    Args:
        track_a:      path to audio file (.mp3, .wav, .flac) or pre-extracted .npy.
        track_b:      path to audio file or .npy.
        features_dir: directory to cache MERT .npy features.
        device:       "cpu" or "cuda" (default: "cpu").

    Returns:
        dict with keys: score (float), verdict (str), track_a (str), track_b (str)
    """
    import numpy as np
    from pathlib import Path
    from src.sonics_similarity import compare_tracks_sonics_topk

    # Pre-populate MERT cache for .npy inputs so _get_mert_features never
    # calls _load_audio() on a feature array.
    def _ensure_mert_cache(track_path: str) -> None:
        p = Path(track_path)
        if p.suffix.lower() != ".npy":
            return
        cache_path = Path(features_dir) / f"{p.stem}_mert.npy"
        if cache_path.exists():
            return
        arr = np.load(str(p), mmap_mode="r")
        mert_slice = arr[:, :768].astype(np.float32)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(cache_path), mert_slice)

    _ensure_mert_cache(track_a)
    _ensure_mert_cache(track_b)

    result = compare_tracks_sonics_topk(
        track_a, track_b, k=10, cache_dir=features_dir, device=device
    )
    return {
        "score":   result["score"],
        "verdict": result["verdict"],
        "track_a": track_a,
        "track_b": track_b,
    }


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    if len(sys.argv) < 3:
        print("Usage:")
        print("  By ID:   python src/compare_tracks_api.py <id_1> <id_2>")
        print("  By path: python src/compare_tracks_api.py <track_a> <track_b>")
        print(f"\nRegistered IDs: {sorted(TRACK_REGISTRY)}")
        sys.exit(1)

    arg1, arg2 = sys.argv[1], sys.argv[2]
    if arg1 in TRACK_REGISTRY and arg2 in TRACK_REGISTRY:
        result = compare_tracks(arg1, arg2)
    else:
        result = compare_tracks_from_paths(arg1, arg2)

    print(f"Score:   {result['score']:.4f}")
    print(f"Verdict: {result['verdict']}")
    print(f"Track A: {result['track_a']}")
    print(f"Track B: {result['track_b']}")
