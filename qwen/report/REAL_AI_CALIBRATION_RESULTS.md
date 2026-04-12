# Ground Truth Real→AI Calibration Results

## Experiment

Matched 75 real songs to their AI-generated derivatives in the SONICS dataset by comparing lyrics hashes. Downloaded real songs from YouTube and AI songs from HuggingFace. Extracted MERT embeddings and computed file-level cosine similarity.

## What We Have

- **64 real songs** downloaded from YouTube (11 failed — copyright-restricted)
- **149 AI songs** downloaded from HuggingFace zips (most pairs have 2 variants)
- **30 complete pairs** with both real and at least one AI variant having MERT embeddings

## Results

| Metric | Value |
|--------|-------|
| Positive pairs (real→AI derivative) | 60 |
| Negative pairs (real→AI unrelated) | 34 |
| **ROC AUC** | **0.5725** |
| **Positive mean** | **0.871** |
| **Negative mean** | **0.885** |
| **Gap (pos - neg)** | **-0.014** |

## The Honest Verdict

**ROC AUC = 0.57** — barely above random chance (0.50).

**The gap is NEGATIVE** (-0.014): real→AI derivatives actually score LOWER on average than unrelated real→AI pairs. This is because:
1. MERT captures the AI generation "fingerprint" (vocoder artifacts, HNR profile) more strongly than the source song relationship
2. Some real→AI derivative pairs score very high (0.946) — likely when the AI closely follows the source melody
3. Others score very low (0.323) — when the AI interprets the lyrics with completely different musical content
4. The high variance (σ=0.104) makes the distributions overlap almost completely

## What This Means

MERT Top-K cosine similarity **cannot reliably detect whether an AI song was derived from a specific real source song**. The system works for:
- **Real vs Real plagiarism** (MIPPIA calibration, AUC ~0.84)
- **AI vs AI same-source detection** (SONICS GMM, AUC ~0.85)
- **AI detection** (fingerprint analysis, HNR d>1.3)

But it does NOT work for:
- **Real → AI attribution** (AUC = 0.57 — not significantly above chance)

## What Would Be Needed

Real→AI attribution requires analysis that MERT embeddings don't capture:
1. **Melodic contour matching** — does the AI melody follow the source melody?
2. **Chord progression analysis** — are the harmonic structures similar?
3. **Lyric-to-melody alignment** — does the AI set the lyrics to a similar melody?
4. **Structural analysis** — verse/chorus/bridge matching

MERT embeddings capture "what kind of music this is" but not "this specific melody from this specific source."

## Calibration Parameters (for reference)

```json
{
  "mu_pos": 0.87077,
  "sigma_pos": 0.10404,
  "mu_neg": 0.88454,
  "sigma_neg": 0.02405,
  "roc_auc": 0.5725,
  "gap": -0.01377,
  "n_positive": 60,
  "n_negative": 34
}
```

These are saved in `qwen/results/real_ai_calibration.json`.

## Per-Pair Results

Some notable pairs:
- **Post Malone → Go Flex AI**: 0.941, 0.946 (highest — AI closely followed source)
- **Frank Ocean → Novacane AI**: 0.878, 0.844 (moderate)
- **Linkin Park → In the End AI**: 0.910, 0.887 (moderate)
- Some pairs scored as low as 0.323 (AI completely reinterpreted the lyrics)

The wide range (0.32–0.95) confirms that AI generation from lyrics produces highly variable musical output — sometimes close to the source, sometimes completely different.
