# Audio Attribution

Compare two audio tracks and return a score in [0, 1] indicating how likely one is a derivative of the other.

**Final system:** Top-K MERT k=10 — ROC AUC **0.850**
**Report:** `report/audio_attribution_report.pdf`

---

## Quick Start

```bash
python compare_tracks.py track_a.mp3 track_b.mp3
```

The script runs Top-K MERT k=10 and prints a score with verdict. On first run it creates a local venv and installs dependencies (~2 min). Subsequent runs go straight to comparison.

---

## Demo (no audio needed)

Pre-extracted demo features are included:

```bash
python compare_tracks.py demo/001_ori.npy demo/001_comp.npy
```

Expected output:

```
=================================================================
  Score:   1.0000  [████████████████████████]  100.0%
  Verdict: LIKELY DERIVATIVE
  Raw:     0.9644  (lo=0.9220  hi=0.9336)
  Method:  Top-K MERT k=10  (0.6s)
=================================================================
```

> **Note on perfect scores:** A score of 1.0000 does not mean the function is
> broken or clipped. The SONICS distribution is naturally compressed: even
> unrelated AI-generated tracks share high MERT similarity in absolute terms
> (negatives average raw ~0.924). The calibration window is intentionally
> tight (lo=0.9220, hi=0.9336) to separate that narrow gap. The included demo
> pair scores raw 0.9644 — well above the ceiling — so it saturates to 1.0.
> The **raw score** is the more informative value when comparing pairs that
> both score 1.0. Unrelated pairs score 0.0 and weakly-related pairs land in
> between.

---

## How it works

MERT-v1-95M (768-dim transformer embeddings) is extracted per 10-second segment. For each segment in Track A, the most similar segment anywhere in Track B is found using cosine similarity — regardless of timestamp. The top-10 per-segment maxima are averaged and calibrated to [0, 1].

**Why Top-K instead of banded diagonal:** AI generators do not preserve the temporal structure of the source. Banded diagonal matching assumes borrowed content appears at the same relative position in both tracks — which is wrong for AI attribution. Top-K removes this constraint.

**Why no learned model:** The 200 available SONICS base IDs (400 tracks) are not sufficient to train a neural model that beats a well-calibrated cosine baseline. Top-K cosine on MERT achieves ROC AUC 0.850, above the best trained Siamese model (0.838 on MIPPIA).

---

## API

```python
from src.compare_tracks_api import compare_tracks, compare_tracks_from_paths

# By registered ID (see src/registry.py)
result = compare_tracks("001_ori", "001_comp")

# By file path
result = compare_tracks_from_paths("track_a.mp3", "track_b.mp3")

# result: {"score": 1.0, "verdict": "LIKELY DERIVATIVE", "track_a": ..., "track_b": ...}
```

To register new tracks, add entries to `src/registry.py`.

---

## Verdicts

| Score range | Verdict |
|-------------|---------|
| ≥ 0.75 | LIKELY DERIVATIVE |
| 0.55 – 0.74 | POSSIBLY RELATED |
| 0.35 – 0.54 | WEAK SIMILARITY |
| < 0.35 | UNRELATED |

Scores are calibrated using SONICS distribution bounds: lo=0.9220, hi=0.9336.

---

## Limitations

- **Calibration scope:** Estimated from 200 SONICS base IDs (parts 01–02, single generator per ID). Cross-generator pairs (Suno ↔ Udio from the same source) are untested — SONICS parts 03–10 are needed.
- **FakeMusicCaps not used:** 10-second TTM clips require a different pipeline; deprioritised given GPU and time constraints.
- **GPU recommended:** MERT extraction is slow on CPU (~10 min per track). Pre-extracted `.npy` files skip extraction entirely.
- **Score saturation:** Positive pairs in the demo score 1.0 due to calibration bounds compression. The raw MERT similarity (0.9644) carries more signal than the calibrated score in the upper tail.
