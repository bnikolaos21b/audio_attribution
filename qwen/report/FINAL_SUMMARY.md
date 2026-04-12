# Final System Summary

## The Complete Picture

This document summarizes everything we learned about audio attribution through extensive experimentation.

## What We Built

A **5-stage general-purpose attribution system** that handles any pair type honestly:

```
Stage 1: Track classifier (real/suno/udio) — HNR + LUFS + autocorrelation
Stage 2: Pair type router → selects best calibration
Stage 3: MERT Top-K k=10 (for same-domain pairs)
Stage 4: Calibrated score (lyrics Jaccard for real→AI, GMM for AI/AI, minmax for real/real)
Stage 5: Honest explanation with reliability labels
```

## The Big Discovery: Lyrics Solve Real→AI Attribution

| Method | Positive Mean | Negative Mean | Gap | ROC AUC |
|--------|-------------|--------------|-----|---------|
| **Lyrics Jaccard** | **0.277** | **0.129** | **+0.148** | **0.8856** |
| Lyrics Sentence-Transformer | 0.616 | 0.392 | +0.224 | 0.8749 |
| MERT Top-K | 0.871 | 0.885 | -0.014 | 0.5725 |

**Lyrics similarity achieves AUC 0.89 vs MERT's 0.57** — transforming an unsolvable problem into a strong one.

**Why:** AI generators take lyrics as input. When generating from real song lyrics, the AI sings the same or similar words — even if the melody is completely different. MERT captures musical features (timbre, harmony, rhythm) but NOT lyrical content.

## Validated Calibrations

| Pair Type | Calibration | AUC | N Pairs |
|-----------|------------|-----|---------|
| Real vs Real | MIPPIA min-max (lo=0.88, hi=0.99) | ~0.84 | 39 pos + 117 neg |
| Same-engine AI/AI | SONICS GMM (μ_pos=0.936, μ_neg=0.924) | ~0.85 | 40 pos + 40 neg |
| Real → AI (lyrics) | Jaccard (μ_pos=0.277, μ_neg=0.129) | 0.89 | 128 pos + 192 neg |
| Cross-engine AI/AI | None (no derivative data) | — | 21 neg, 0 pos |

## What Doesn't Work

1. **MERT for real→AI** — AUC 0.57, negative gap (-0.014). The AI "fingerprint" is louder than the source relationship.
2. **Autocorrelation lag for version ID** — ±5ms variance on MP3s makes it unreliable per-track.
3. **Mel skewness for generator ID** — genre-dependent, not generator-dependent.
4. **Bayesian probability multiplication** — events aren't independent; invalid.

## Key Insights

1. **All music lives in a ~22° cone in MERT space** — unrelated songs score ~0.92-0.94 by default
2. **Same-prompt AI siblings score higher than real→AI derivatives** (0.941 vs 0.901) — MERT can't distinguish them
3. **Generator identity matters more than source identity** in MERT space — engine gap (0.021) > source gap (0.010)
4. **SONICS data has NO real songs** — all 160 tracks are AI (72 Suno + 88 Udio)
5. **The 75 ground truth real→AI pairs** were found by matching lyrics hashes between fake_songs.csv and real_songs.csv

## Files

All work is inside `qwen/` — nothing outside was modified:

```
qwen/src/general_attribution.py          # Main 5-stage pipeline
qwen/src/lyrics_attribution_experiment.py # Whisper transcription + similarity
qwen/src/ai_detector.py                   # AI detection (recalibrated)
qwen/src/generator_classifier.py          # Generator classification
qwen/src/suno_version_classifier.py       # Version identification
qwen/src/compute_real_ai_calibration.py   # MERT calibration from cached data
qwen/results/real_ai_calibration.json     # MERT real→AI params (AUC=0.57)
qwen/results/lyrics_attribution_results.json # Lyrics params (AUC=0.89)
qwen/results/real_ai_pairs/               # 192 transcripts + 64 real + 149 AI audio
qwen/report/LYRICS_ATTRIBUTION_RESULTS.md # Full lyrics experiment report
qwen/report/REAL_AI_GROUND_TRUTH.md       # 75-pair mapping with YouTube links
qwen/report/REAL_AI_CALIBRATION_RESULTS.md # MERT real→AI calibration report
qwen/README.md                            # System documentation
```

## How to Use

```bash
# General mode — uses best calibration per pair type
python compare_tracks_v2.py track_a.mp3 track_b.mp3

# SONICS mode — original MERT-only system (for .npy inputs)
python compare_tracks_v2.py track_a.npy track_b.npy --mode sonics
```

## The Honest Bottom Line

- **Real vs Real plagiarism:** MIPPIA calibration works (AUC ~0.84)
- **AI vs AI same generator:** SONICS GMM works (AUC ~0.85)
- **Real → AI attribution:** Lyrics Jaccard works (AUC 0.89) — this was the breakthrough
- **Cross-engine AI/AI:** Cannot calibrate (no derivative data)
- **Same-prompt AI siblings:** Cannot distinguish from derivatives (MERT limitation)

A system that knows its own limits is more valuable than one that confidently gives wrong answers.
