# Ground Truth Real→AI Derivative Pairs

## How We Found Them

The SONICS dataset contains 49,074 AI-generated songs with full lyrics and 48,090 real songs with lyrics. By matching lyrics hashes, we found **75 ground truth real→AI derivative pairs** where the AI song was generated from the exact lyrics of a specific real song.

## Methodology

1. Downloaded `fake_songs.csv` (49,074 rows, 17 columns) and `real_songs.csv` (48,090 rows, 13 columns) from HuggingFace (`awsaf49/sonics`)
2. Matched `fake_songs.lyrics` to `real_songs.lyrics` using 500-character hash
3. Real songs include `youtube_id` — downloadable via yt-dlp
4. AI songs stored in HuggingFace zip parts — downloadable via remotezip (HTTP range requests)
5. Extracted MERT embeddings for both real and AI songs
6. Computed file-level cosine similarity

## The 75 Matched Pairs

All 75 pairs are available at: `qwen/src/sonics_real_ai_mapping.py` (generated during investigation).

Each pair includes:
- Real song title, artist, YouTube ID
- AI song filename, algorithm (chirp-v2/v3/v3.5)
- YouTube URL for download

## 3 Pairs Downloaded and Tested

| Real Song | Artist | YouTube | AI Algorithm |
|-----------|--------|---------|-------------|
| In the End | Linkin Park | gB9P0kpc1O4 | chirp-v3.5 |
| Novacane | Frank Ocean | hgOu8eRJZ3Q | chirp-v3.5 |
| Go Flex | Post Malone | 68WnHYxo9PA | chirp-v3.5 |

Each has 2 AI variants (variant 0 and variant 1 = two independent generations from same lyrics).

## Ground Truth MERT Similarity Scores

### Real → AI Derivative (6 pairs)
| Pair | Score |
|------|-------|
| Linkin Park → AI(v0) | 0.90973 |
| Linkin Park → AI(v1) | 0.88688 |
| Frank Ocean → AI(v0) | 0.87770 |
| Frank Ocean → AI(v1) | 0.84358 |
| Post Malone → AI(v0) | 0.94090 |
| Post Malone → AI(v1) | 0.94628 |
| **Mean** | **0.90085** |
| **Std** | **0.03596** |

### Same-Source AI Siblings (3 pairs)
| Pair | Score |
|------|-------|
| Linkin Park AI(v0) → AI(v1) | 0.91644 |
| Frank Ocean AI(v0) → AI(v1) | 0.93155 |
| Post Malone AI(v0) → AI(v1) | 0.94916 |
| **Mean** | **0.93238** |

### Unrelated Real → AI (3 cross-pairs)
| Pair | Score |
|------|-------|
| Linkin Park (real) → Frank Ocean (AI) | 0.90147 |
| Linkin Park (real) → Post Malone (AI) | 0.90378 |
| Frank Ocean (real) → Post Malone (AI) | 0.85714 |
| **Mean** | **0.88746** |

## The Honest Verdict

**The derivative-unrelated gap is only 0.013.** This is barely larger than the derivative std (0.036). The distributions overlap significantly:
- Derivatives: [0.844 – 0.946]
- Unrelated: [0.857 – 0.904]

**AI siblings score higher than real→AI derivatives** (0.932 vs 0.901). This means:
- Two independent AI generations from the same lyrics are MORE similar than a real song and its AI derivative
- The MERT model captures the AI generation "fingerprint" more strongly than the source song relationship

**What this means for calibration:**
Real→AI derivative detection is possible but with very low discrimination power. At best, a raw score of 0.901 (mean derivative) would calibrate to P(derivative) ≈ 0.42 — below the "LIKELY DERIVATIVE" threshold.

## Where the Data Lives

- Real songs: `/home/nikos_pc/Desktop/Orfium/real_ai_pairs/real_*.mp3`
- AI songs: `/home/nikos_pc/Desktop/Orfium/real_ai_pairs/fake_*.mp3`
- Full 75-pair mapping: computed during investigation (see `qwen/src/cross_category_investigation.py`)
- Download script: use `yt-dlp --js-runtimes node` for YouTube + `remotezip` for HuggingFace zips
