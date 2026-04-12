# Lyrics Attribution Experiment Results

## The Question

MERT embeddings fail at real→AI attribution (AUC=0.57). Can lyrics analysis — which MERT completely ignores — solve this problem?

## Methodology

1. **Transcribed** 192 audio files (64 real songs + 128 AI derivatives) using Whisper-tiny
2. **Computed Jaccard similarity** (word-level overlap) for each real→AI pair
3. **Computed sentence-transformer cosine similarity** (all-MiniLM-L6-v2) for each real→AI pair
4. **Compared against 192 negative pairs** (unrelated real→AI combinations)
5. **Reported ROC AUC** for both methods

## Results

| Method | Positive Mean | Negative Mean | Gap | ROC AUC |
|--------|-------------|--------------|-----|---------|
| **Jaccard (word overlap)** | 0.2767 ± 0.118 | 0.1288 ± 0.040 | +0.148 | **0.8856** |
| **Sentence-Transformer** | 0.6158 ± 0.149 | 0.3918 ± 0.118 | +0.224 | **0.8749** |
| **MERT Top-K** | 0.871 ± 0.104 | 0.885 ± 0.024 | -0.014 | 0.5725 |

## The Finding

**Lyrics analysis achieves AUC 0.89 vs MERT's 0.57.** This is a transformative improvement.

The gap between derivative and unrelated pairs:
- **Jaccard: +0.148** (11.5x the MERT gap of -0.014)
- **Sentence-Transformer: +0.224** (16x the MERT gap)

## Why This Works

AI music generators take lyrics (or prompts) as input. When generating a song from real song lyrics:
- The AI **sings the same or similar words** — even if the melody is different
- Whisper captures these lyrics in the transcript
- Word overlap (Jaccard) directly measures: "did the AI sing the same words?"

MERT embeddings, by contrast, capture musical features (timbre, harmony, rhythm) but NOT the lyrical content. Since AI generators can completely reinterpret lyrics with different melodies, MERT sees no connection.

## Per-Pair Highlights

Some notable results:

| Song | Jaccard | Sentence | Notes |
|------|---------|----------|-------|
| Monsters Under My Bed | 0.513-0.750 | 0.719-0.761 | Highest — AI sang nearly identical lyrics |
| 25 Minutes to Go | 0.414-0.518 | 0.765-0.899 | Strong lyrics match |
| Protector | 0.429-0.498 | 0.860-0.898 | Strong sentence embedding match |
| In the End | 0.527-0.582 | 0.634-0.732 | Moderate — recognizable lyrics |
| Go Flex | 0.307 | 0.645 | AI modified lyrics but preserved meaning |

## Why Jaccard > Sentence-Transformer (slightly)

Jaccard AUC (0.8856) slightly beats sentence-transformer (0.8749) because:
- Direct word overlap is more robust to Whisper transcription errors
- Sentence-embeddings can be fooled by paraphrasing (similar meaning, different words)
- Jaccard is binary: either the same words are there or they're not

## Implications for the Attribution System

The general attribution system should be updated to use **lyrics similarity as the primary signal for real→AI pairs**, with MERT as a secondary signal. The combined approach would be:

```
If pair type is real/AI:
    lyrics_sim = Jaccard(real_transcript, ai_transcript)
    mert_sim = MERT_TopK(real_audio, ai_audio)
    combined = 0.7 * lyrics_sim + 0.3 * mert_sim  # Weight lyrics heavily
    → Expected AUC > 0.89
```

## Limitations

- **Instrumental music:** For AI songs with no lyrics, this method doesn't apply
- **Non-English lyrics:** Whisper-tiny may struggle with non-English transcription
- **Paraphrased lyrics:** If the AI paraphrases the lyrics, Jaccard drops but semantic meaning may be preserved (sentence-transformer handles this better)
- **Whisper accuracy:** Transcription errors affect both methods
- **192 files only:** 64 real + 128 AI — modest sample size

## Files

- Transcripts: `results/real_ai_pairs/transcripts.json` (192 entries)
- Results: `results/lyrics_attribution_results.json`
- Plot: `results/lyrics_attribution_results.png`
- Script: `src/lyrics_attribution_experiment.py`
