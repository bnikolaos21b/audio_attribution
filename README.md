# Audio Attribution

Compare two audio tracks and return a score in [0, 1] indicating how likely one is a derivative of the other.

This is a general-purpose framework covering three pair types:

| Pair type | Signal | ROC AUC | Status |
|-----------|--------|---------|--------|
| Real vs Real (plagiarism) | MERT Top-K + MIPPIA calibration | 0.838 | Validated |
| AI vs AI same-engine | MERT Top-K + SONICS GMM | 0.850 | Validated |
| Real → AI (lyrics match) | Jaccard word overlap + Gaussian calibration | 0.886 | Validated |
| Cross-engine AI vs AI | No derivative data | — | Unvalidated |

**Report:** `report/audio_attribution_report.tex` (compiled → `audio_attribution_report.pdf`)

---

## Quick Start

```bash
# Auto mode: detects pair type and applies best calibration
python compare_tracks.py track_a.mp3 track_b.mp3

# SONICS mode (backward compatible, accepts .npy pre-extracted features)
python compare_tracks.py feat_a.npy feat_b.npy --mode sonics
```

On first run the script creates a local venv and installs all dependencies
(~2 min). Subsequent runs skip straight to comparison.

**Demo** (no audio required):

```bash
python compare_tracks.py demo/001_ori.npy demo/001_comp.npy --mode sonics
```

Expected output:
```
  Score:   0.7843  [███████████████████░░░░░]  78.4%
  Verdict: LIKELY DERIVATIVE
  Raw:     0.9644  →  P(derivative)=0.7843  [GMM]
  Method:  Top-K MERT k=10 + GMM calibration  (SONICS)
```

---

## How It Works

### The 5-Stage Pipeline (auto mode)

```
Input: Track A + Track B
      │
  Stage 1: Classify each track  (HNR + LUFS + autocorrelation)
      │
  Stage 2: Route to calibration
      ├─ real/real    → MIPPIA min-max  (AUC 0.838)
      ├─ AI/AI        → SONICS GMM     (AUC 0.850)
      ├─ real/AI      → Lyrics Jaccard (AUC 0.886)  ← Whisper transcription
      └─ cross-engine → LOW confidence flag
      │
  Stage 3: MERT Top-K k=10
           (segment cosine similarity, position-free)
      │
  Stage 4: Calibrated score
           (pair-type-appropriate posterior)
      │
  Stage 5: Explanation
           (verdict + warnings + segment matches)
```

**Why Top-K instead of banded diagonal:** AI generators do not preserve temporal
structure. A melody from minute 1 of the source may appear at minute 3 of the
derivative, or not at all. Top-K removes the positional constraint.

**Why Jaccard for real→AI:** MERT captures acoustic character (timbre, rhythm,
generator fingerprint), not lyrical content. For AI derivatives of real songs,
MERT achieves AUC = 0.57 (chance) because the AI generator's vocoder signature
dominates the source song relationship. Jaccard word overlap achieves AUC = 0.886
because the AI sings the same lyrics even when the melody is reinterpreted.

**Why not a trained model:** 200 SONICS base IDs are insufficient to train a neural
model that beats a well-calibrated cosine baseline. MERT Top-K cosine achieves
AUC 0.850, above the best trained Siamese MLP (0.838).

---

## Pair-Type Output

### Real → AI (Jaccard route)

```
  Score:       0.8621  [████████████████████░░░░]  86.2%
  Verdict:     LIKELY DERIVATIVE
  Reliability: HIGH
  Pair type:   real/ai
  Calibration: Lyrics Jaccard (128 pos + 192 neg real→AI pairs)

  ── Track A (real, conf=0.84) ──
    HNR: 14.2  |  LUFS: -16.8  |  Autocorr: 0.023 at 11.3ms

  ── Track B (suno, conf=0.91) ──
    HNR: 28.4  |  LUFS: -15.1  |  Autocorr: 0.147 at 18.6ms

  ── Lyrics-Based Attribution (primary signal) ──
    Jaccard similarity:   0.2814  (threshold μ_pos=0.277, μ_neg=0.129)
    Sentence similarity:  0.5972
    Combined (70/30):     0.3761
    MERT AUC (real→AI):   0.5725 — not used as primary signal
    Jaccard AUC:          0.8856 — primary signal for this pair type
```

### AI/AI Same-Engine (MERT GMM route)

```
  Score:       0.7843  [███████████████████░░░░░]  78.4%
  Verdict:     LIKELY DERIVATIVE
  Reliability: HIGH
  Pair type:   same-engine-ai
  Calibration: SONICS GMM (40 pos + 40 neg, chirp-v3.5 + udio-120s)
  Raw:         0.9380

  Note: Raw score 0.9380 falls within same-prompt sibling range
  (0.941 ± 0.027). Derivative and independent generations
  are indistinguishable here.
```

---

## Verdicts

| Score | Verdict |
|-------|---------|
| ≥ 0.75 | LIKELY DERIVATIVE |
| 0.55 – 0.74 | POSSIBLY RELATED |
| 0.35 – 0.54 | WEAK SIMILARITY |
| < 0.35 | UNRELATED |

---

## Source Layout

```
compare_tracks.py          ← entry point (auto + sonics modes)
src/
  general_attribution.py   ← 5-stage pipeline (general mode)
  sonics_similarity.py     ← MERT Top-K + GMM (SONICS mode)
  features.py              ← 1294-dim feature extraction
  compare_tracks_api.py    ← programmatic API
  registry.py              ← track ID → path mapping
  lyrics_attribution.py    ← Jaccard + sentence-transformer experiment
  melody_contour_experiment.py ← DTW melody experiment (rejected)
  real_ai_calibration.py   ← MERT real→AI calibration (AUC=0.57)
  suno_analysis/           ← 7 forensic analysis scripts
results/
  plots/                   ← all figures (.png)
  calibration/             ← SONICS GMM params
  real_ai/                 ← real/AI experiment results
  sonics/                  ← SONICS benchmark results
  baseline/                ← MIPPIA Siamese baseline results
data/
  real_ai_pairs/           ← 64 real + 149 AI songs, transcripts, MERT cache
  manifests/               ← feature manifests
report/
  audio_attribution_report.tex   ← full LaTeX report
  audio_attribution_report.pdf   ← compiled report
```

---

## Python API

```python
from src.general_attribution import attribute_tracks

# Any pair of audio files or .npy features
result = attribute_tracks("song_a.mp3", "song_b.mp3")

print(result.score)         # calibrated P(derivative) in [0, 1]
print(result.verdict)       # LIKELY DERIVATIVE | POSSIBLY RELATED | ...
print(result.pair_type)     # real/ai | real/real | same-engine-ai | ...
print(result.reliability)   # HIGH | MODERATE | LOW

# Full formatted explanation
print(result.summary())

# Top segment matches (for MERT route)
for m in result.top_matches[:5]:
    print(f"  A@{m['seg_a_time_sec']}s ↔ B@{m['seg_b_time_sec']}s  {m['similarity']:.4f}")
```

---

## Limitations

**Real/Real route (MIPPIA calibration):**
- Calibrated on MIPPIA domain (acoustic recordings with direct borrowing).
- May not transfer to electronic music or remixes with heavy production changes.

**AI/AI same-engine route (SONICS GMM):**
- Same-prompt siblings are indistinguishable from derivatives in MERT space
  (both score ~0.941). Explicit warning is issued when raw score is in this range.
- Calibrated on chirp-v3.5 and udio-120s only; cross-version performance untested.
- Suno v4+ (chirp-auk) has only 26 profiled songs; may degrade Stage-1 classification.

**Real/AI route (Lyrics Jaccard):**
- Requires lyrics: instrumental music falls back to MERT (unreliable, AUC = 0.57).
- Only Suno derivatives tested; Udio-derived real songs have no calibration.
- Calibrated on 64 songs — modest sample. Non-English lyrics untested.
- Whisper-tiny transcription errors affect both Jaccard and sentence similarity.

**Cross-engine AI/AI:**
- No positive derivative pairs exist. LOW reliability flag; do not use for verdicts.

**GPU:** MERT extraction runs ~10 min/track on CPU. Pre-extracted `.npy` files
skip extraction entirely. Pass `--device cuda` for GPU acceleration.
