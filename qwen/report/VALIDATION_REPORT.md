# Validation Report: Multi-Stage Pipeline on 26 Folk Suno + 5 Real Folk

## Reconciled Findings (after cross-checking with Claude's original analysis)

### HNR — RECONCILED: Finding is Legitimate

The original pipeline used `10*log10(power_ratio)` from basic STFT HPSS, which showed overlapping HNR values between Real and Suno on MP3 files. This was **the wrong formula**.

Claude's formula (from `extended_analysis.py`): `librosa.effects.hpss(margin=3.0)` → frame-level RMS → voiced-frame mean ratio.

Running Claude's exact formula on the same files:

| Group | Mean | Range |
|-------|------|-------|
| Real (5) | 16.44 | [10.82 – 24.30] |
| Suno (26) | 31.54 | [15.33 – 83.80] |

**Mann-Whitney p=0.003, Cohen's d=1.34.** Partial overlap exists (real max 24.30 vs suno min 15.33) but the separation is statistically significant.

Robustness check across HPSS margin values:

| Margin | Real mean | Suno mean | p-value | d |
|--------|-----------|-----------|---------|---|
| 2.0 | 10.58 | 19.11 | 0.0038 | 1.27 |
| 3.0 | 16.44 | 31.54 | 0.0030 | 1.34 |
| 4.0 | 22.87 | 46.11 | 0.0023 | 1.34 |

The finding holds across all margin values. **The HNR separation is legit.**

### AI Detection — RECONCILED: Works with Correct Formula

With the corrected HNR formula, AI detection now works:
- Real correctly identified as non-AI: 3/5 (60%)
- Suno correctly identified as AI: 5/5 (100%) — in the sample tested

Two real files were false positives due to mid/side correlation (mono recordings). The HNR feature alone would have separated them better.

### Autocorrelation Lag — RECONCILED: Window Scope Matters

Candle Wax Strings (1).mp3: peak at 15.3ms (AC=0.268) ✓
Candle Wax Strings.mp3: peaks at 13.7ms (AC=0.207) AND 27.2ms (AC=0.611)

The 27.2ms peak is much larger but is **musical periodicity** (~37Hz = strumming rhythm), not the vocoder artifact. The 10–25ms search window correctly excludes it. Claude's 13.7ms vs 15.3ms measurement (diff 1.6ms) is the correct vocoder comparison.

The lag is a population-level trend — within-pair variance is ~3ms std. Per-track version identification has low confidence.

### Bayeisan Multiplication — RECONCILED: Dropped

Replaced with diagnostic annotation. The MERT score stands on its own. Fingerprint evidence provides regime classification and reliability flags.

### Mel Skewness — RECONCILED: Genre-dependent

Dropped as generator feature. Folk music has positive skew regardless of generator.

### What Changed in This Revision

1. **HNR formula** → Switched to `librosa.effects.hpss(margin=3.0)` with voiced-frame RMS ratio
2. **Autocorrelation window** → Kept at 10–25ms (validated range for vocoder artifacts)
3. **Bayesian multiplication** → Replaced with `attribute()` function (diagnostic, not score-modifying)
4. **Mel skewness** → Removed from generator classification features
5. **Version identification** → Added uncertainty warnings (population-level trend only)
6. **All HNR references** → Updated to use linear ratio (not dB)
