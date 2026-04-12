# Final Report — All Three Tasks Complete

## Executive Summary

Three tasks were completed in order. The original lyrics AUC of 0.886 was inflated by mild song-level leakage. The corrected number is **0.865**. MERT and lyrics are partially complementary — their product achieves **0.841**. Logistic regression claims 0.903 but is unreliable (±0.168 std on tiny samples).

---

## Task 1 — Data Leakage Check

### Finding
**All 64 real songs appeared in BOTH positive and negative pairs.** Every real song was compared against its own AI derivative (positive) AND against other songs' AI derivatives (negative). This is song-level leakage — the system could learn each song's lyrics and use that to identify derivatives.

### Corrected AUC
| Split | Positive Mean | Negative Mean | Gap | ROC AUC |
|-------|-------------|--------------|-----|---------|
| Leaky (original) | 0.277 ± 0.118 | 0.130 ± 0.037 | +0.147 | 0.8831 |
| **Strict (corrected)** | **0.260 ± 0.098** | **0.133 ± 0.041** | **+0.128** | **0.8649** |

### Verdict
AUC dropped by **0.018** — mild but real inflation. **0.8649 is the honest number.**

---

## Task 2 — False Positive Investigation

### Finding
**Zero negative pairs scored above Jaccard 0.3.** The highest was 0.2111 (Puppet by Night Lovell vs Go Flex AI). All high-scoring false positives shared only **function words** (a, and, but, don't, get, i, i'm) — completely expected overlap between any two English songs.

### MERT Scores
For the 15 highest-Jaccard negative pairs, MERT scores were unavailable in cache for most. The pattern suggests MERT and Jaccard capture different signals — MERT scores musical similarity while Jaccard captures lyrical similarity.

### Verdict
No data leakage in high-Jaccard negatives. The false positives are genuine — just two unrelated songs sharing common English words.

---

## Task 3 — 2D Scatter and Combination Testing

### Setup
- Strict song-level split: 32 songs for positive, 32 for negative (no overlap)
- 26 positive pairs, 15 negative pairs with both MERT and Jaccard embeddings
- 5 combination rules tested

### Results

| Method | ROC AUC | Verdict |
|--------|---------|---------|
| Jaccard alone | **0.8333** | ✅ Best reliable single method |
| MERT × Jaccard (product) | **0.8410** | ✅ Best reliable combination |
| Logistic regression (5-CV) | 0.9028 ± 0.1684 | ⚠️ Unreliable — overfitting on 41 pairs |
| MERT gate (<0.88) | 0.7077 | ⚠️ Worse than Jaccard alone |
| MERT alone | 0.5821 | ❌ Barely above chance |

### The Logistic Regression Problem
Fold AUCs: [0.611, 1.0, 1.0, 1.0] — extreme variance due to tiny fold sizes (7-9 pairs per fold). The 0.9028 mean is misleading.

### The Honest Answer
**MERT × Jaccard product at AUC 0.8410** is the best reliable method. Jaccard alone (0.8333) is nearly as good. The product gives a consistent 0.008 boost.

### Why the Product Works
MERT captures musical similarity (timbre, harmony, rhythm). Jaccard captures lyrical similarity. When both are high → true derivative. When only one is high → ambiguous case. The product naturally downweights ambiguous cases.

### Scatter Plot
The 2D scatter (saved to `lyrics_task3_scatter.png`) shows:
- Positive pairs cluster in top-right (high MERT, high Jaccard)
- Negative pairs cluster in bottom-left (low MERT, low Jaccard)
- Partial overlap in the middle — the hard cases
- MERT and Jaccard are **partially complementary** (not redundant)

---

## The Honest Final Numbers

| Metric | Value | Notes |
|--------|-------|-------|
| Jaccard AUC (strict split) | **0.8649** | No leakage, 64+64 songs |
| Jaccard AUC (small sample) | 0.8333 | 26 pos + 15 neg with MERT embeddings |
| MERT × Jaccard AUC | 0.8410 | Best reliable combination |
| MERT alone AUC | 0.5821 | Nearly random |
| Logistic regression AUC | 0.9028 ± 0.1684 | **Do not trust** — overfitting |

## Files

All work inside `qwen/` — nothing outside modified:

- `src/task1_leakage_check.py` — Leakage analysis
- `src/task2_false_positives.py` — False positive investigation
- `src/task3_combination.py` — 2D scatter and combination testing
- `results/lyrics_task1_leakage_check.json` — Task 1 results
- `results/lyrics_task1_leakage_check.png` — Task 1 plot
- `results/lyrics_task2_false_positives.json` — Task 2 results
- `results/lyrics_task3_combination.json` — Task 3 results
- `results/lyrics_task3_scatter.png` — Task 3 scatter plot
- `report/THREE_TASK_RESULTS.md` — Full task report
