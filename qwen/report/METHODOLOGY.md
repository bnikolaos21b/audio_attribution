# Methodology: From Single-Stage MERT to Multi-Stage Attribution

> This document traces the analytical reasoning that led from the original submission
> to the v2 multi-stage pipeline. It records findings, dead ends, and honest uncertainties.

## 1. The Original System

The Orfium submission solved a well-defined problem: **given two audio tracks, score how likely one is a derivative of the other.**

The solution was MERT Top-K k=10 with GMM calibration, achieving ROC AUC 0.850.

### What the original system understood

- **AI generators don't preserve temporal structure** → Top-K beats banded diagonal
- **Neural embeddings are enough** → No trained model needed; cosine similarity beats Siamese MLP
- **Calibration matters** → GMM posterior is more interpretable than min-max clipping

### What it didn't address

- Is the track AI-generated at all? (Pre-filter question)
- Which generator made it? (Suno vs Udio vs different Suno versions)
- How should generator identity affect the attribution score?

---

## 2. The First Extension: Suno Forensic Analysis

We started with 26 Suno-generated folk tracks (v4.5-all) and 5 real folk recordings.

### Features analyzed (7 scripts, 20+ features)

| Feature | Cohen's d (Suno vs Real) | Verdict |
|---------|-------------------------|---------|
| HNR | >2.0 (estimated) | **Strong AI indicator** |
| Mid/Side correlation | >2.0 (estimated) | **Strong — but only on stereo** |
| Autocorrelation 17ms peak | Present in 100% | **Strong — but class-level, not gen-level** |
| LUFS | ~1.0 | Moderate |
| Crest factor | ~0.8 | Weak |
| Mel skewness | ~1.5 | **Genre-dependent — unreliable** |
| Noise floor consistency | ~1.2 | Moderate |

### The critical finding: vocoder autocorrelation

The 17ms peak in the waveform autocorrelation comes from the neural vocoder's frame-rate periodicity. Every neural vocoder divides audio into frames and processes them independently, creating a subtle periodicity at the frame boundary interval.

**This is a CLASS fingerprint, not a GENERATOR fingerprint.** It exists in both Suno and Udio because both use neural vocoders. You cannot use it to tell Suno from Udio.

But it CAN tell you which Suno VERSION was used — because the lag drifts across generations.

---

## 3. Scaling to SONICS: 160 Tracks

The SONICS dataset provided 72 Suno + 88 Udio tracks (all chirp-v3.5 for Suno).

### What confirmed at scale

| Finding | 26 Songs | 160 SONICS | Decision |
|---------|----------|------------|----------|
| HNR elevated in AI | ✅ | ✅ Confirmed | **Keep** |
| Autocorrelation peak | ✅ Both | ✅ Both have it | **Keep (class-level)** |
| LUFS difference | Moderate | **Huge (d=3.01)** | **Keep — strongest gen discriminator** |

### What broke

| Finding | 26 Songs | 160 SONICS | Why |
|---------|----------|------------|-----|
| Mel skewness positive for Suno | ✅ | ❌ Flipped negative | **Genre-dependent, not generator-dependent** |
| Mid/Side ratio | ✅ Strong | ❌ Cannot verify | **SONICS re-encoded to mono** |
| Encoder fingerprint | ✅ Lavc60.31 | ❌ Cannot verify | **SONICS stripped ID3 tags** |

### The biggest surprise: Suno vs Udio separation

**LUFS at d=3.01** is enormous. Udio is 5.4dB louder than Suno on average. This is a deliberate mastering choice baked into Udio's pipeline — not an artifact, not a side effect. It's the single most reliable discriminator between generators.

**HNR at d=1.66** tells a different story. Suno produces cleaner harmonics (more tonal, less noisy) while Udio has more percussive character. This reflects their architectures: Suno's language model approach vs Udio's diffusion process.

---

## 4. Cross-Version Suno Analysis

SONICS contains three Suno versions: chirp-v2, chirp-v3, chirp-v3.5.

### The lag drift discovery

| Version | Median lag | LUFS | Crest |
|---------|-----------|------|-------|
| chirp-v2 | 15.7ms | -11.6 dB | 14.0 dB |
| chirp-v3 | 17.6ms | -17.7 dB | 14.0 dB |
| chirp-v3.5 | 19.0ms | -15.7 dB | 12.0 dB |

The lag drifts monotonically: **v2 → v3 → v3.5**. Each generation uses a slightly different codec frame rate. This is measurable and consistent.

But then v4.5 (chirp-auk) **resets to 15.3ms** — breaking the trend entirely. v4 uses a different internal architecture.

### LUFS normalization trend

- v2: Loud (-11.6 dB) with high variance (±1.7)
- v3: Overcorrected (-17.7 dB) — quieter than v3.5
- v3.5: Tightest spread (±0.7) — deliberate loudness normalization

This suggests Suno iterated on their mastering pipeline across versions.

### Crest factor compression

v3.5 has lower crest factor (12.0 dB) vs v2/v3 (~14 dB) — more peak limiting. This is consistent with the tighter LUFS normalization.

---

## 5. The Version Naming Problem

Suno's public UI names (v2, v3, v3.5, v4, v4.5, v5, v5.5) do NOT map cleanly to internal names:

| UI Name | Internal Name | In SONICS? |
|---------|--------------|------------|
| v2 | chirp-v2-xxl-alpha | ✅ |
| v3 | chirp-v3 | ✅ |
| v3.5 | chirp-v3.5 | ✅ |
| v4 | Unknown (new naming) | ❌ |
| v4.5 | chirp-auk | ❌ |
| v4.5+ | Unknown | ❌ |
| v5 / v5.5 | Unknown | ❌ |

At v4, Suno abandoned the `chirp-v*` naming. Our 26 folk songs were generated with v4.5-all (chirp-auk internally) — a completely different architecture from anything in SONICS.

**Practical implication:** Everything we know about chirp versions applies only to v3.5 and earlier. v4+ is uncharted territory.

---

## 6. Design Decisions for v2

### Why Bayesian combination, not a new trained model?

The 200 SONICS base IDs (400 tracks) are insufficient to train a model that learns the interaction between fingerprint features and MERT similarity. A Bayesian approach:

1. Uses empirically measured priors from MERT geometry
2. Incorporates fingerprint evidence as likelihood factors
3. Is interpretable — you can see exactly how each stage contributes
4. Doesn't require labeled training data beyond what we already have

### Why geometric mean, not arithmetic?

Arithmetic means saturate: if one factor is near 1.0 and another near 0.0, the mean is 0.5 regardless. Geometric mean is more sensitive to disagreement between factors — which is appropriate because if the AI detector says "not AI" but MERT says "very similar", we should have lower confidence.

### Why keep MERT as the dominant factor?

MERT Top-K has the highest individual AUC (0.850). The fingerprint features have high effect sizes (HNR d>2, LUFS d=3) but they answer a different question: "is this AI-generated?" not "is B derived from A specifically?" The MERT score is the most direct answer to the attribution question.

### What we chose NOT to do

1. **Replace MERT with fingerprint-only scoring** — Fingerprint features detect AI generation, not attribution to a specific source. A track can be AI-generated without being derived from your specific source.
2. **Train a neural classifier for generator ID** — Too little data per generator. Gaussian likelihood models are more sample-efficient for this regime.
3. **Build version detection for Udio** — No version information available for Udio tracks in SONICS.
4. **Use FakeMusicCaps** — 10-second TTM clips require a different pipeline; deprioritized given GPU and time constraints.

---

## 7. Honest Limitations

### Calibration scope
- GMM fitted on 80 SONICS pairs (chirp-v3.5 only)
- Cross-generator pairs (suno↔udio) are all negative — no positive calibration data
- Cross-version calibration (v2 vs v3.5) not tested

### Statistical power
- v4.5 profiles from 26 songs only — not robust
- Real folk: only 5 songs — extremely underpowered
- Udio: 88 tracks from SONICS, but all re-encoded to mono 128kbps

### Generalization
- All findings from folk music domain — may not transfer to pop, classical, etc.
- Mel skewness proved genre-dependent — other features may be too
- Autocorrelation lag may vary by genre (percussive vs sustained instruments)

### Pipeline evaluation
- **No end-to-end ROC AUC measured for the combined score** — this is the biggest gap
- Individual stages have empirical support, but the combination hasn't been validated
- Weight tuning (50/20/15/15) was design choice, not optimized

---

## 8. What Would Make This Production-Ready

1. **End-to-end evaluation:** ROC AUC for the combined score on held-out data
2. **Cross-genre validation:** Test on pop, classical, electronic music
3. **More generators:** Add Stable Audio, MusicGen, Riffusion fingerprints
4. **Version expansion:** Profile v5, v5.5 when data becomes available
5. **Weight optimization:** Learn the combination weights from labeled pairs
6. **Speed optimization:** AI detection pre-filter should run BEFORE MERT to skip expensive computation on clearly non-AI tracks
7. **Uncertainty quantification:** Confidence intervals on the combined score

---

## 9. The Complete Evidence Chain

```
26 folk Suno songs (v4.5-all) + 5 real folk
    ↓
7 forensic analysis scripts → 20+ features
    ↓
HNR + autocorrelation + mid/side identified as AI indicators
    ↓
Scale to 160 SONICS tracks (72 Suno chirp-v3.5 + 88 Udio)
    ↓
CONFIRMED: HNR (d=1.66), LUFS (d=3.01), autocorrelation (both generators)
DISCARDED: Mel skewness (genre-dependent), mid/side (SONICS mono), encoder (stripped)
    ↓
Cross-version analysis (chirp-v2/v3/v3.5 from SONICS HuggingFace)
    ↓
DISCOVERED: Lag drift (15.7→17.6→19.0ms), LUFS normalization trend, crest factor compression
    ↓
v4.5 profiling: 15.3ms lag (chirp-auk, new architecture)
    ↓
Multi-stage pipeline: AI detect → Gen classify → MERT → Combine
```

---

*This methodology document is a living record. As more data becomes available
(more generators, more versions, more genres), the findings and decisions
documented here should be re-evaluated.*
