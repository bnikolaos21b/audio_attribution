"""
Stage 4: Attribution Result with Diagnostic Context
=====================================================
Instead of multiplying fingerprint probabilities into the MERT score
(which violates independence assumptions — see report/METHODOLOGY.md),
this module uses fingerprint evidence as CONTEXT for interpreting the MERT score.

The MERT Top-K GMM-calibrated score (ROC AUC 0.850) remains the primary output.
Fingerprint analysis adds:
  - Annotation: "Track B shows AI fingerprints: vocoder peak at Xms, HNR=Y"
  - Flag: "Both tracks are Suno — attribution unreliable (gap=0.002)"
  - Regime selection: different interpretation for AI→AI vs Real→AI vs Real→Real

Design rationale:
  P(attribution | context) is not P(attribution) × P(context).
  The context tells us which regime we're in, not how to scale the score.
"""

import json
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np

_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

from .ai_detector import AIDetectionResult
from .generator_classifier import GeneratorResult

# ── Logistic regression calibration ──────────────────────────────────────────
# Trained on 80 SONICS pairs (parts 01–02).  Features: MERT cosine, mean HNR,
# mean LUFS, mean autocorr peak strength.  Inputs must be standardised with the
# saved scaler before applying coef + intercept.
#
# Training results (80 pairs, 5-fold stratified CV):
#   MERT alone 5-fold AUC:  0.6656
#   All 4 features 5-fold:  0.6219  (delta=−0.044)
#
# Note: adding HNR/LUFS/autocorr does NOT improve CV AUC at n=80 — those
# features discriminate generator identity (Suno vs Udio) rather than
# pair-level similarity.  The GMM on MERT alone (AUC 0.850 on 200-pair test)
# remains the recommended calibration for the primary score output.
# This LR is provided for research purposes and as the replacement for the
# invalid Bayesian multiplication that was previously here.

_LR_PARAMS_PATH = _project_root / "results" / "calibration" / "logistic_calibration_params.json"

def _load_lr_params():
    if not _LR_PARAMS_PATH.exists():
        return None
    with open(_LR_PARAMS_PATH) as f:
        return json.load(f)

_LR_PARAMS = _load_lr_params()


def score_with_lr(mert_cosine: float,
                  mean_hnr: float,
                  mean_lufs: float,
                  mean_autocorr: float) -> Optional[float]:
    """
    Logistic regression score using 4 validated features.

    Trained on 80 SONICS pairs.  Applies saved StandardScaler then
    sigmoid(coef · x + intercept).

    Returns P(derivative | features), or None if params not loaded.

    Caveat: 5-fold CV AUC = 0.622, slightly below MERT-alone (0.666).
    The GMM calibration on MERT cosine alone (AUC 0.850) is superior for
    production use.
    """
    if _LR_PARAMS is None:
        return None
    x = np.array([mert_cosine, mean_hnr, mean_lufs, mean_autocorr], dtype=float)
    mu  = np.array(_LR_PARAMS["scaler_mean"])
    std = np.array(_LR_PARAMS["scaler_std"])
    x_scaled = (x - mu) / (std + 1e-12)
    coef      = np.array(_LR_PARAMS["coef"])
    intercept = _LR_PARAMS["intercept"]
    logit = float(np.dot(coef, x_scaled) + intercept)
    return float(1.0 / (1.0 + np.exp(-logit)))


@dataclass
class AttributionResult:
    # Primary output — MERT Top-K GMM-calibrated score
    score: float           # [0, 1] GMM-calibrated attribution score
    verdict: str           # "LIKELY DERIVATIVE" / "POSSIBLY RELATED" / etc.

    # Diagnostic context (not multiplied into score)
    track_a_ai: Optional[AIDetectionResult]
    track_b_ai: Optional[AIDetectionResult]
    track_a_gen: Optional[GeneratorResult]
    track_b_gen: Optional[GeneratorResult]

    # Context flags
    regime: str            # "real_to_real", "real_to_ai", "ai_to_ai", "cross_generator"
    annotations: list      # Human-readable diagnostic notes
    reliability: str       # "HIGH", "MODERATE", "LOW" — how much to trust the score

    def summary(self) -> str:
        lines = []
        lines.append("=" * 72)
        lines.append("  ATTRIBUTION ANALYSIS")
        lines.append("=" * 72)

        bar = "█" * max(0, min(24, int(round(self.score * 24)))) + "░" * max(0, min(24, 24 - int(round(self.score * 24))))
        lines.append(f"  Score:       {self.score:.4f}  [{bar}]  {self.score * 100:.1f}%")
        lines.append(f"  Verdict:     {self.verdict}")
        lines.append(f"  Regime:      {self.regime}")
        lines.append(f"  Reliability: {self.reliability}")
        lines.append("")

        if self.annotations:
            lines.append("  ── Diagnostics ──")
            for note in self.annotations:
                lines.append(f"    • {note}")
            lines.append("")

        if self.track_b_ai:
            lines.append(f"  ── Track B AI Detection ──")
            lines.append(f"  AI likely:     {self.track_b_ai.is_likely_ai} (confidence={self.track_b_ai.confidence:.3f})")
            lines.append(f"  Evidence:      {self.track_b_ai.evidence_count}/3 indicators")
            if self.track_b_ai.autocorr_peak_lag_ms:
                lines.append(f"  Vocoder lag:   {self.track_b_ai.autocorr_peak_lag_ms}ms")
            if self.track_b_ai.hnr_ratio:
                lines.append(f"  HNR:           {self.track_b_ai.hnr_ratio:.1f}")
            lines.append(f"  Notes:         {self.track_b_ai.notes}")
            lines.append("")

        if self.track_b_gen:
            lines.append(f"  ── Generator Classification ──")
            a_label = f"{self.track_a_gen.predicted_generator}" if self.track_a_gen else "unknown"
            if self.track_a_gen and self.track_a_gen.predicted_version:
                a_label += f" ({self.track_a_gen.predicted_version})"
            b_label = f"{self.track_b_gen.predicted_generator}"
            if self.track_b_gen.predicted_version:
                b_label += f" ({self.track_b_gen.predicted_version})"
            lines.append(f"  Track A:       {a_label}")
            lines.append(f"  Track B:       {b_label}")
            lines.append(f"  Confidence:    {self.track_b_gen.confidence:.3f}")
            lines.append("=" * 72)
        return "\n".join(lines)


def _get_verdict(score: float) -> str:
    if score >= 0.75:
        return "LIKELY DERIVATIVE"
    elif score >= 0.55:
        return "POSSIBLY RELATED"
    elif score >= 0.35:
        return "WEAK SIMILARITY"
    else:
        return "UNRELATED"


def _determine_regime(track_a_gen: Optional[GeneratorResult],
                      track_b_gen: Optional[GeneratorResult]) -> str:
    a = track_a_gen.predicted_generator if track_a_gen else "unknown"
    b = track_b_gen.predicted_generator if track_b_gen else "unknown"

    if a == "real" and b == "real":
        return "real_to_real"
    elif (a == "real" and b in ("suno", "udio")) or (b == "real" and a in ("suno", "udio")):
        return "real_to_ai"
    elif a in ("suno", "udio") and b in ("suno", "udio"):
        if a == b:
            return "ai_to_ai_same_generator"
        else:
            return "ai_to_ai_cross_generator"
    else:
        return "unknown"


def _determine_reliability(regime: str,
                           track_a_gen: Optional[GeneratorResult],
                           track_b_gen: Optional[GeneratorResult]) -> str:
    if regime == "real_to_real":
        return "HIGH"  # MERT works well for compositional plagiarism
    elif regime == "real_to_ai":
        return "HIGH"  # AI model trained on/prompted with real track → MERT captures it
    elif regime == "ai_to_ai_same_generator":
        return "LOW"   # Same-prompt vs different-prompt gap is only 0.002 in MERT space
    elif regime == "ai_to_ai_cross_generator":
        return "LOW"   # No positive cross-generator calibration data
    else:
        return "MODERATE"


def _generate_annotations(regime: str,
                          track_a_ai: Optional[AIDetectionResult],
                          track_b_ai: Optional[AIDetectionResult],
                          track_a_gen: Optional[GeneratorResult],
                          track_b_gen: Optional[GeneratorResult]) -> list:
    annotations = []

    # AI detection annotations
    if track_b_ai and track_b_ai.is_likely_ai:
        details = []
        if track_b_ai.autocorr_peak_lag_ms:
            details.append(f"vocoder peak at {track_b_ai.autocorr_peak_lag_ms:.1f}ms")
        if track_b_ai.hnr_ratio:
            details.append(f"HNR={track_b_ai.hnr_ratio:.1f}")
        annotations.append(f"Track B shows AI fingerprints: {'; '.join(details)}")
    elif track_b_ai and not track_b_ai.is_likely_ai:
        annotations.append("Track B shows no strong AI fingerprints")

    if track_a_ai and track_a_ai.is_likely_ai:
        details = []
        if track_a_ai.autocorr_peak_lag_ms:
            details.append(f"vocoder peak at {track_a_ai.autocorr_peak_lag_ms:.1f}ms")
        if track_a_ai.hnr_ratio:
            details.append(f"HNR={track_a_ai.hnr_ratio:.1f}")
        annotations.append(f"Track A shows AI fingerprints: {'; '.join(details)}")

    # Regime-specific warnings
    if regime == "ai_to_ai_same_generator":
        a = track_a_gen.predicted_generator if track_a_gen else "?"
        annotations.append(
            f"Both tracks are {a} — attribution is unreliable. "
            f"Same-prompt and different-prompt {a} pairs have a MERT gap of only 0.002."
        )
    elif regime == "ai_to_ai_cross_generator":
        a = track_a_gen.predicted_generator if track_a_gen else "?"
        b = track_b_gen.predicted_generator if track_b_gen else "?"
        annotations.append(
            f"Cross-generator pair ({a} → {b}) — no positive calibration data exists. "
            f"SONICS has no same-prompt cross-generator pairs."
        )
    elif regime == "real_to_ai":
        b = track_b_gen.predicted_generator if track_b_gen else "AI"
        ver = ""
        if track_b_gen and track_b_gen.predicted_version:
            ver = f" ({track_b_gen.predicted_version})"
        annotations.append(
            f"Real → {b}{ver}: MERT score captures whether the AI output was prompted with Track A."
        )

    return annotations


def attribute(score: float,
              track_a_ai: Optional[AIDetectionResult] = None,
              track_b_ai: Optional[AIDetectionResult] = None,
              track_a_gen: Optional[GeneratorResult] = None,
              track_b_gen: Optional[GeneratorResult] = None) -> AttributionResult:
    """
    Produce attribution result with diagnostic context.

    The score is the MERT Top-K GMM-calibrated score — unchanged.
    Fingerprint evidence provides annotations and reliability flags, not score modification.
    """
    regime = _determine_regime(track_a_gen, track_b_gen)
    reliability = _determine_reliability(regime, track_a_gen, track_b_gen)
    annotations = _generate_annotations(regime, track_a_ai, track_b_ai, track_a_gen, track_b_gen)

    verdict = _get_verdict(score)

    return AttributionResult(
        score=score,
        verdict=verdict,
        track_a_ai=track_a_ai,
        track_b_ai=track_b_ai,
        track_a_gen=track_a_gen,
        track_b_gen=track_b_gen,
        regime=regime,
        annotations=annotations,
        reliability=reliability,
    )
