# Quick Reference: Audio Attribution Fingerprint Findings

## What distinguishes AI from Real

| Feature | AI | Real | Effect Size | Formula |
|---------|-----|------|-------------|---------|
| **HNR (HPSS margin=3.0, voiced RMS ratio)** | 31.5 mean | 16.4 mean | **d=1.34, p=0.003** | `librosa.effects.hpss(margin=3.0)` → RMS → voiced-frame mean ratio |
| **Autocorrelation 10–25ms peak** | ✅ Present (AC > 0.03) | ❌ Usually absent | Strong | Neural vocoder frame artifact. Universal across generators. |
| **Mid/Side correlation** | Variable | Variable | Weak | Only works on stereo files — SONICS destroyed this signal |

**Note:** An earlier version of this pipeline used `10*log10(power_ratio)` for HNR, which showed no separation on MP3s. The correct formula is the HPSS voiced-frame RMS ratio from the original Orfium analysis.

## What distinguishes Suno from Udio

| Feature | Suno | Udio | Effect Size | Notes |
|---------|------|------|-------------|-------|
| **LUFS** | -15.7 dB (chirp-v3.5) | -10.1 dB | **d=3.01** | Udio is 5.4dB louder — mastering choice |
| **HNR** | Higher (cleaner) | Lower (more noise) | d=1.66 | Suno = language model (clean), Udio = diffusion (noisy) |
| **Crest factor** | 12–14 dB | 13 dB | Moderate | v3.5 more compressed |
| **Autocorrelation lag** | 15.7–19.0ms | ~17ms | Weak | Both have vocoder peak — class-level, not gen-level |

## What distinguishes Suno versions

| Version | Lag (SONICS) | Lag (MP3 folk) | LUFS | Notes |
|---------|-------------|----------------|------|-------|
| **chirp-v2** | 15.7ms ± 2.0 | N/A | -11.6 dB | Loud, high variance |
| **chirp-v3** | 17.6ms ± 2.0 | N/A | -17.7 dB | Overcorrected loudness |
| **chirp-v3.5** | 19.0ms ± 2.0 | N/A | -15.7 dB | Tight normalization |
| **v4.5-auk** | N/A | 15.3ms ± 5.0 | ~-15.5 dB | **New architecture** |

**⚠️ WARNING:** The lag drift (15.7→17.6→19.0ms) is beautiful on SONICS data but
**NOT stable on MP3-compressed audio**. Same-prompt duplicates vary by ±5ms.
Version identification on MP3 inputs should be treated as LOW CONFIDENCE.
Only on WAV/lossless does the lag become a reliable version discriminator.

## What destroyed our confidence

1. **Mel skewness is genre-dependent** — Our 26 folk songs showed positive skewness for Suno. SONICS Suno showed negative. Folk genre has different spectral shape.
2. **SONICS stripped ID3 tags** — Encoder fingerprint (Lavc60.31) unverifiable.
3. **SONICS re-encoded to mono** — Mid/side ratio unverifiable.
4. **v4.5 uses chirp-auk** — Different architecture from anything in SONICS.

## The Bayesian pipeline

```
P(attribution) = prior × P(AI) × P(generator) × P(MERT related)

Weights: MERT 50%, Prior 20%, AI 15%, Generator 15%
```

## Priors from MERT geometry

| A → B | Prior |
|-------|-------|
| Real → Suno | 0.25 |
| Real → Udio | 0.25 |
| Suno → Suno | 0.55 |
| Udio → Udio | 0.55 |
| Suno ↔ Udio | 0.20 |
| Real → Real | 0.50 |

## The honest truth

- The autocorrelation peak detects **neural vocoders**, not specific generators
- LUFS is the strongest **generator discriminator** (d=3.01 Suno vs Udio)
- HNR is the strongest **AI vs Real discriminator** (d>2.0)
- Mel skewness is **unreliable** — genre-dependent
- Mid/side ratio is **powerful but fragile** — destroyed by mono re-encoding
- v4.5+ is **uncharted territory** — only 26 songs profiled
- The combined pipeline has **no end-to-end ROC AUC measured** yet
