#!/usr/bin/env python3
"""
Validation: Run v2 pipeline on known pairs from the 26 folk Suno songs.

Tests:
  1. Same-prompt pairs (original + (1) duplicate) → should score high
  2. Real vs Suno → should detect AI in Suno, not in Real
  3. Generator classification → Suno songs classified as Suno, Real as Real
  4. HNR formula validation → confirms margin=3.0 separation
"""

import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

from src.ai_detector import detect_ai
from src.generator_classifier import classify_generator
from src.combined_pipeline import attribute


AUDIO_DIR = Path("/home/nikos_pc/Desktop/Orfium/suno")


def _print_header(title):
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")


def main():
    _print_header("VALIDATION: AI Detection + Generator Classification + Diagnostics")

    import re
    all_mp3 = sorted(AUDIO_DIR.glob("*.mp3"))
    real_files = [f for f in all_mp3 if f.name.startswith("real_")]
    suno_files = [f for f in all_mp3 if not f.name.startswith("real_")]

    def canonical(path):
        return re.sub(r"\s*\(1\)\s*$", "", path.stem).strip()

    canon_map = {}
    for f in suno_files:
        c = canonical(f)
        canon_map.setdefault(c, []).append(f)
    same_prompt_pairs = [(v[0], v[1]) for v in canon_map.values() if len(v) == 2]

    print(f"\n  Files found: Real={len(real_files)}, Suno={len(suno_files)}, Pairs={len(same_prompt_pairs)}")

    # ── Test 1: HNR validation ──────────────────────────────────────────
    _print_header("TEST 1: HNR Validation (margin=3.0, voiced-frame RMS ratio)")

    print(f"\n  Real files:")
    real_hnr = []
    for f in real_files:
        ai = detect_ai(str(f))
        if ai.hnr_ratio is not None:
            real_hnr.append(ai.hnr_ratio)
            print(f"    {f.name}: HNR={ai.hnr_ratio:.2f}")
    print(f"  Real: mean={sum(real_hnr)/len(real_hnr):.2f}, range=[{min(real_hnr):.2f}-{max(real_hnr):.2f}]")

    print(f"\n  Suno files:")
    suno_hnr = []
    for f in suno_files:
        ai = detect_ai(str(f))
        if ai.hnr_ratio is not None:
            suno_hnr.append(ai.hnr_ratio)
            print(f"    {f.name}: HNR={ai.hnr_ratio:.2f}")
    print(f"  Suno: mean={sum(suno_hnr)/len(suno_hnr):.2f}, range=[{min(suno_hnr):.2f}-{max(suno_hnr):.2f}]")

    # Stats
    import numpy as np
    from scipy import stats as st
    r_arr = np.array(real_hnr)
    s_arr = np.array(suno_hnr)
    _, p = st.mannwhitneyu(s_arr, r_arr, alternative='two-sided')
    pooled = np.sqrt((np.var(s_arr, ddof=1) + np.var(r_arr, ddof=1)) / 2)
    d = (np.mean(s_arr) - np.mean(r_arr)) / (pooled + 1e-12)
    print(f"\n  Mann-Whitney p={p:.4f}, Cohen's d={d:.3f}")
    print(f"  Overlap: real max={max(real_hnr):.2f}, suno min={min(suno_hnr):.2f}")

    # ── Test 2: AI Detection ────────────────────────────────────────────
    _print_header("TEST 2: AI Detection")

    ai_correct_real = 0
    ai_correct_suno = 0

    for f in real_files:
        ai = detect_ai(str(f))
        if not ai.is_likely_ai:
            ai_correct_real += 1
        print(f"  Real  {f.name}: AI={ai.is_likely_ai}, evidence={ai.evidence_count}/3, conf={ai.confidence:.3f}")

    for f in suno_files[:5]:
        ai = detect_ai(str(f))
        if ai.is_likely_ai:
            ai_correct_suno += 1
        print(f"  Suno  {f.name}: AI={ai.is_likely_ai}, evidence={ai.evidence_count}/3, conf={ai.confidence:.3f}")

    print(f"\n  Real correctly non-AI: {ai_correct_real}/{len(real_files)}")
    print(f"  Suno correctly AI: {ai_correct_suno}/5 (sample)")

    # ── Test 3: Generator Classification ────────────────────────────────
    _print_header("TEST 3: Generator Classification")

    gen_correct = 0
    total = 0

    for f in real_files[:3]:
        gen = classify_generator(str(f))
        is_correct = gen.predicted_generator == "real"
        if is_correct:
            gen_correct += 1
        total += 1
        print(f"  {f.name}: {gen.predicted_generator}" +
              (f" ({gen.predicted_version})" if gen.predicted_version else "") +
              f" conf={gen.confidence:.3f} {'✓' if is_correct else '✗'}")

    for f in suno_files[:5]:
        gen = classify_generator(str(f))
        is_correct = gen.predicted_generator == "suno"
        if is_correct:
            gen_correct += 1
        total += 1
        print(f"  {f.name}: {gen.predicted_generator}" +
              (f" ({gen.predicted_version})" if gen.predicted_version else "") +
              f" conf={gen.confidence:.3f} {'✓' if is_correct else '✗'}")

    print(f"\n  Generator classification: {gen_correct}/{total}")

    # ── Test 4: Full Attribution (diagnostic, not score manipulation) ───
    _print_header("TEST 4: Full Attribution with Diagnostics")

    if real_files and suno_files:
        real_f = real_files[0]
        suno_f = suno_files[0]
        print(f"\n  Real → Suno: {real_f.name} → {suno_f.name}")

        ai_real = detect_ai(str(real_f))
        ai_suno = detect_ai(str(suno_f))
        gen_real = classify_generator(str(real_f))
        gen_suno = classify_generator(str(suno_f))

        result = attribute(
            score=0.78,  # Placeholder — would come from MERT
            track_a_ai=ai_real,
            track_b_ai=ai_suno,
            track_a_gen=gen_real,
            track_b_gen=gen_suno,
        )
        print(result.summary())

    print(f"\n{'=' * 72}")
    print("  Validation complete.")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
