# Audio Attribution v2 — Honest General-Purpose System

## Core Principle

A system that knows its own limits is more valuable than one that confidently gives wrong answers.

## Architecture

```
Input: Track A + Track B
       ↓
Stage 1: Track Classifier (real / suno / udio / unknown)
  → HNR (margin=3.0, voiced frames), LUFS, autocorrelation peak
       ↓
Stage 2: Pair Type Router → selects calibration
  → real/real        → MIPPIA min-max ✅
  → same-engine AI   → SONICS GMM ✅
  → cross-engine AI  → Flag as uncalibrated ⚠️
  → real/AI          → Ground truth calibration (AUC=0.57) ⚠️
  → unknown          → SONICS GMM fallback ⚠️
       ↓
Stage 3: MERT Top-K k=10 (unchanged from original)
       ↓
Stage 4: Calibrated score with source labeled
       ↓
Stage 5: Explanation with honesty warnings
```

## What Works (Validated)

| Pair Type | Calibration | AUC | Status |
|-----------|------------|-----|--------|
| Real vs Real (plagiarism) | MIPPIA min-max (lo=0.88, hi=0.99) | ~0.84 | ✅ Validated on 39 pos + 117 neg |
| Same-engine AI/AI | SONICS GMM (μ_pos=0.936, μ_neg=0.924) | ~0.85 | ✅ Validated on 40 pos + 40 neg |
| **Real → AI (lyrics-based)** | **Jaccard word overlap (μ_pos=0.277, μ_neg=0.129)** | **0.89** | ✅ Validated on 128 pos + 192 neg |

## What Doesn't Work (Honestly Reported)

| Pair Type | Problem | What System Says |
|-----------|---------|-----------------|
| **Cross-engine AI/AI** | No derivative calibration | "Likely unrelated — no cross-engine derivative data exists" |
| **Same-prompt AI siblings** | Score ~0.941 (same as derivatives) | Warning: "Cannot distinguish from derivatives" |
| **Real → AI (MERT only)** | AUC=0.57, no discrimination | Superseded by lyrics analysis |

## Usage

```bash
# General mode (default) — honest 5-stage pipeline
python compare_tracks_v2.py track_a.mp3 track_b.mp3

# SONICS mode — original system (for .npy inputs)
python compare_tracks_v2.py track_a.npy track_b.npy --mode sonics
```

## Ground Truth Data

We matched **75 real→AI derivative pairs** by comparing lyrics hashes between SONICS `fake_songs.csv` (49,074 AI songs) and `real_songs.csv` (48,090 real songs). Each real song has a YouTube ID for download.

- 64 real songs downloaded from YouTube
- 149 AI songs downloaded from HuggingFace zips
- 30 complete pairs with MERT embeddings
- **ROC AUC: 0.5725** (barely above chance)

Full results: `report/REAL_AI_CALIBRATION_RESULTS.md`
Calibration params: `results/real_ai_calibration.json`

## File Structure

```
qwen/
├── compare_tracks_v2.py                    # Main entry point
├── src/
│   ├── general_attribution.py              # 5-stage honest pipeline
│   ├── ai_detector.py                      # AI detection (recalibrated)
│   ├── generator_classifier.py             # Generator classification
│   ├── suno_version_classifier.py          # Version identification
│   ├── combined_pipeline.py                # Diagnostic attribution
│   ├── cross_category_investigation.py     # Cross-category analysis
│   ├── mert_geometry_investigation.py      # MERT embedding geometry
│   ├── sonics_allpairwise_experiment.py    # All-vs-all SONICS
│   ├── combined_fingerprint_analysis.py    # Fingerprint analysis
│   ├── download_real_ai_pairs.py           # Real→AI pair download
│   ├── download_ai_songs.py                # AI song download
│   ├── extract_real_ai_calibration.py      # MERT extraction + calibration
│   └── compute_real_ai_calibration.py      # Quick calibration from cached
├── report/
│   ├── METHODOLOGY.md                       # Full reasoning chain
│   ├── VALIDATION_REPORT.md                # Fingerprint validation
│   ├── GENERAL_SYSTEM_VALIDATION.md        # New system validation
│   ├── REAL_AI_GROUND_TRUTH.md             # 75-pair mapping
│   └── REAL_AI_CALIBRATION_RESULTS.md      # Actual AUC results
├── results/
│   ├── real_ai_pairs/                      # Downloaded audio + embeddings
│   ├── real_ai_calibration.json            # Ground truth calibration
│   ├── real_ai_calibration.png             # Calibration plot
│   └── plots/                              # Analysis figures
└── tests/
    └── validate_on_folk.py                 # Validation on folk songs
```

## Key Findings

1. **MERT is great for same-domain comparison** (real vs real, AI vs AI) but **poor for cross-domain** (real vs AI) — AUC=0.57
2. **Lyrics similarity solves the real→AI problem** — Jaccard AUC=0.89, transforming an unsolvable problem into a strong one
3. **The AI generation fingerprint is louder than the source song fingerprint** in MERT space
4. **75 ground truth real→AI pairs** with YouTube links — the missing calibration data is now public
5. **Same-prompt AI siblings** (two independent gens from same lyrics) score higher than real→AI derivatives in MERT space
6. **The system never fabricates confidence** — it uses the best available signal for each pair type

## Honest Limitations

- Real/AI lyrics calibration: 128 positive + 192 negative pairs, AUC=0.89 — strong but limited to vocal music
- Cross-engine AI/AI: 0 derivative pairs, cannot calibrate
- All MERT calibration data is from SONICS (chirp-v2/v3/v3.5) — v4.5+ is untested
- Track classifier can misclassify real rock/electronic music as Udio (low HNR from percussion)
- Lyrics analysis requires Whisper transcription — slow on CPU (~1.3 min per file)
- No end-to-end ROC AUC measured for the combined system
