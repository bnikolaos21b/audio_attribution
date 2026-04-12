"""
Track registry — maps song IDs to local feature/audio file paths.

Add entries here to make tracks accessible via compare_tracks(song_id_1, song_id_2).
Values can be .npy feature files (782-dim) or raw audio files (.mp3, .wav, etc.).

Keys are opaque string IDs; the underlying comparison logic is path-based and
never sees the ID. This file is the only place that needs to change when new
tracks are registered.
"""

TRACK_REGISTRY: dict[str, str] = {
    "001_ori":  "demo/001_ori.npy",
    "001_comp": "demo/001_comp.npy",
}
