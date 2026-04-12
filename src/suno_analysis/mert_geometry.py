"""
MERT latent-space geometry: same-prompt Suno pairs vs different-prompt Suno pairs
vs real-vs-real pairs.

Embedding approach matches the Orfium assignment pipeline exactly:
  - m-a-p/MERT-v1-95M
  - 24 kHz mono, 10-second windows, 5-second hop
  - outputs.last_hidden_state.mean(dim=1)  →  768-dim per segment
  - per-file embedding = mean of all segment embeddings  →  768-dim
"""

import warnings
warnings.filterwarnings("ignore")

import os
import itertools
import re
from pathlib import Path

import numpy as np
import librosa
import torch
from transformers import AutoProcessor, AutoModel
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# ── Config ────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
AUDIO_DIR     = Path(os.environ.get("SUNO_AUDIO_DIR", str(_PROJECT_ROOT / "data" / "suno")))
OUT_PNG       = _PROJECT_ROOT / "results" / "plots" / "suno_mert_geometry.png"
CACHE_FILE    = AUDIO_DIR / "mert_embeddings.npz"

MERT_MODEL  = "m-a-p/MERT-v1-95M"
MERT_SR     = 24_000
MERT_DIM    = 768
WINDOW_SEC  = 10
HOP_SEC     = 5
MIN_SEG_SEC = 3.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ── File lists ─────────────────────────────────────────────────────────────────
all_files  = sorted(AUDIO_DIR.glob("*.mp3"))
real_files = [f for f in all_files if f.name.startswith("real_")]
suno_files = [f for f in all_files if not f.name.startswith("real_")]
print(f"Suno: {len(suno_files)}   Real: {len(real_files)}")

# ── Identify same-prompt Suno pairs ───────────────────────────────────────────
# "Candle Wax Strings.mp3" and "Candle Wax Strings (1).mp3" share the same prompt.
# Strip " (1)" to get canonical title.

def canonical(path: Path) -> str:
    return re.sub(r"\s*\(1\)\s*$", "", path.stem).strip()

canon_map: dict[str, list[Path]] = {}
for f in suno_files:
    c = canonical(f)
    canon_map.setdefault(c, []).append(f)

suno_pairs   = [(v[0], v[1]) for v in canon_map.values() if len(v) == 2]
suno_singles = [v[0] for v in canon_map.values() if len(v) == 1]

print(f"\nSuno same-prompt pairs : {len(suno_pairs)}")
for a, b in suno_pairs:
    print(f"  '{a.name}'  ↔  '{b.name}'")
print(f"Suno unpaired          : {len(suno_singles)}")

# ── Audio loading ─────────────────────────────────────────────────────────────

def segment_audio(path: Path):
    """Load full file at MERT_SR, split into 10-s windows with 5-s hop."""
    y, _ = librosa.load(str(path), sr=MERT_SR, mono=True)
    win  = int(WINDOW_SEC  * MERT_SR)
    hop  = int(HOP_SEC     * MERT_SR)
    min_ = int(MIN_SEG_SEC * MERT_SR)
    segs = []
    start = 0
    while start < len(y):
        chunk = y[start:start + win]
        if len(chunk) < min_:
            break
        if len(chunk) < win:                      # zero-pad short tail
            chunk = np.pad(chunk, (0, win - len(chunk)))
        segs.append(chunk.astype(np.float32))
        start += hop
    return segs

# ── MERT embedding ─────────────────────────────────────────────────────────────

def embed_file(path: Path, processor, model) -> np.ndarray:
    """Return one 768-dim vector (mean of all segment embeddings)."""
    segs = segment_audio(path)
    seg_embs = []
    model.eval()
    with torch.no_grad():
        for seg in segs:
            inputs = processor(seg, sampling_rate=MERT_SR,
                               return_tensors="pt", padding=True)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            out    = model(**inputs)
            emb    = out.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
            seg_embs.append(emb.astype(np.float32))
    return np.mean(seg_embs, axis=0)   # (768,)

# ── Load or compute embeddings ────────────────────────────────────────────────

all_ordered = suno_files + real_files   # deterministic order

if CACHE_FILE.exists():
    print(f"\nLoading cached embeddings from {CACHE_FILE}")
    data   = np.load(CACHE_FILE, allow_pickle=True)
    emb_matrix = data["embeddings"]             # (31, 768)
    names      = list(data["names"])
    # Rebuild index
    name_to_idx = {n: i for i, n in enumerate(names)}
else:
    print(f"\nLoading MERT model ({MERT_MODEL})…")
    processor = AutoProcessor.from_pretrained(MERT_MODEL, trust_remote_code=True)
    model     = AutoModel.from_pretrained(MERT_MODEL, trust_remote_code=True).to(DEVICE)

    emb_matrix = np.zeros((len(all_ordered), MERT_DIM), dtype=np.float32)
    names      = []
    for i, f in enumerate(all_ordered):
        tag = "real" if f.name.startswith("real_") else "suno"
        print(f"  [{i+1:2d}/{len(all_ordered)}] [{tag}] {f.name}")
        emb_matrix[i] = embed_file(f, processor, model)
        names.append(f.name)

    np.savez(str(CACHE_FILE), embeddings=emb_matrix, names=np.array(names))
    print(f"Embeddings saved → {CACHE_FILE}")
    name_to_idx = {n: i for i, n in enumerate(names)}

# L2-normalise for cosine similarity
norms      = np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-12
emb_normed = emb_matrix / norms

def cosine(a_name, b_name):
    i = name_to_idx[a_name]
    j = name_to_idx[b_name]
    return float(np.dot(emb_normed[i], emb_normed[j]))

# ── Compute similarity groups ─────────────────────────────────────────────────

# 1. Same-prompt Suno pairs
same_prompt_sims = [cosine(a.name, b.name) for a, b in suno_pairs]

# 2. Real vs real (all C(5,2) = 10 pairs)
real_vs_real_sims = [
    cosine(a.name, b.name)
    for a, b in itertools.combinations(real_files, 2)
]

# 3. Different-prompt Suno pairs (all pairs with different canonical titles)
diff_prompt_sims = []
suno_list = list(suno_files)
for i in range(len(suno_list)):
    for j in range(i + 1, len(suno_list)):
        if canonical(suno_list[i]) != canonical(suno_list[j]):
            diff_prompt_sims.append(cosine(suno_list[i].name, suno_list[j].name))

# 4. Cross-group: Suno vs real
suno_vs_real_sims = [
    cosine(s.name, r.name)
    for s in suno_files for r in real_files
]

print(f"\nSame-prompt pairs       : n={len(same_prompt_sims)}")
print(f"Diff-prompt Suno pairs  : n={len(diff_prompt_sims)}")
print(f"Real vs real pairs      : n={len(real_vs_real_sims)}")
print(f"Suno vs real pairs      : n={len(suno_vs_real_sims)}")

# ── Stats helper ──────────────────────────────────────────────────────────────

def fmt(arr):
    a = np.array(arr)
    return f"{np.mean(a):.4f} ± {np.std(a):.4f}  [min {np.min(a):.4f}, max {np.max(a):.4f}]"

# ── Plot ───────────────────────────────────────────────────────────────────────
SAME_C  = "#E74C3C"   # red
DIFF_C  = "#F39C12"   # orange
REAL_C  = "#2980B9"   # blue
CROSS_C = "#8E44AD"   # purple

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle(
    "MERT Latent-Space Geometry — Suno AI vs Real Folk\n"
    "Model: m-a-p/MERT-v1-95M  |  768-dim  |  last hidden state mean-pooled",
    fontsize=13, fontweight="bold"
)

# ── Panel 1: KDE + rug plot of all three core groups ─────────────────────────
ax = axes[0, 0]
bins = np.linspace(0.4, 1.0, 50)

groups = [
    (same_prompt_sims,  SAME_C,  f"Same-prompt Suno pairs (n={len(same_prompt_sims)})"),
    (diff_prompt_sims,  DIFF_C,  f"Diff-prompt Suno pairs (n={len(diff_prompt_sims)})"),
    (real_vs_real_sims, REAL_C,  f"Real vs Real (n={len(real_vs_real_sims)})"),
]

for sims, col, label in groups:
    a = np.array(sims)
    kde_x = np.linspace(0.3, 1.01, 500)
    kde   = stats.gaussian_kde(a, bw_method=0.15)
    ax.plot(kde_x, kde(kde_x), color=col, lw=2.2, label=label)
    ax.fill_between(kde_x, kde(kde_x), alpha=0.12, color=col)
    ax.plot(a, np.full_like(a, -0.3 - groups.index((sims, col, label)) * 0.3),
            "|", color=col, alpha=0.6, ms=8)

ax.axvline(np.mean(same_prompt_sims), color=SAME_C, ls="--", lw=1.2, alpha=0.8)
ax.axvline(np.mean(real_vs_real_sims),  color=REAL_C,  ls="--", lw=1.2, alpha=0.8)
ax.set_xlabel("Cosine similarity", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title("Similarity distributions (KDE + rug)", fontsize=10)
ax.legend(fontsize=8.5)
ax.set_xlim(0.3, 1.02)
ax.grid(True, alpha=0.25)

# ── Panel 2: Per-pair dot plot for same-prompt pairs ─────────────────────────
ax = axes[0, 1]
pair_labels = [canonical(a) for a, b in suno_pairs]
y_pos = np.arange(len(same_prompt_sims))
ax.barh(y_pos, same_prompt_sims, color=SAME_C, alpha=0.75, height=0.6)
ax.axvline(np.mean(real_vs_real_sims),  color=REAL_C,  ls="--", lw=1.5,
           label=f"Real mean ({np.mean(real_vs_real_sims):.3f})")
ax.axvline(np.mean(diff_prompt_sims),   color=DIFF_C,  ls="--", lw=1.5,
           label=f"Diff-prompt mean ({np.mean(diff_prompt_sims):.3f})")
ax.set_yticks(y_pos)
ax.set_yticklabels(pair_labels, fontsize=7.5)
ax.set_xlabel("Cosine similarity", fontsize=10)
ax.set_title("Same-prompt Suno pair similarities\n(each bar = one prompt pair)", fontsize=10)
ax.legend(fontsize=8)
ax.set_xlim(0.4, 1.0)
ax.grid(True, axis="x", alpha=0.3)

# ── Panel 3: Violin + strip ───────────────────────────────────────────────────
ax = axes[1, 0]
all_groups  = [same_prompt_sims, diff_prompt_sims, real_vs_real_sims, suno_vs_real_sims]
group_cols  = [SAME_C, DIFF_C, REAL_C, CROSS_C]
group_xlabs = ["Same-prompt\nSuno", "Diff-prompt\nSuno", "Real\nvs Real", "Suno\nvs Real"]

rng = np.random.default_rng(42)
for xi, (sims, col, xlab) in enumerate(zip(all_groups, group_cols, group_xlabs)):
    a = np.array(sims)
    parts = ax.violinplot([a], positions=[xi], widths=0.6,
                          showmedians=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_facecolor(col); pc.set_alpha(0.45)
    for part in ["cmedians", "cmaxes", "cmins", "cbars"]:
        parts[part].set_color(col)
    jitter = rng.uniform(-0.12, 0.12, len(a))
    ax.scatter(xi + jitter, a, color=col, alpha=0.55, s=15, zorder=3)

ax.set_xticks(range(4))
ax.set_xticklabels(group_xlabs, fontsize=9)
ax.set_ylabel("Cosine similarity", fontsize=10)
ax.set_title("Violin plot — all four comparison groups", fontsize=10)
ax.grid(True, axis="y", alpha=0.3)

# ── Panel 4: CDF ─────────────────────────────────────────────────────────────
ax = axes[1, 1]
for sims, col, label in groups:
    a = np.sort(np.array(sims))
    cdf = np.arange(1, len(a) + 1) / len(a)
    ax.step(a, cdf, color=col, lw=2.0, label=label, where="post")

# Shade the "hypothesis zone": same-prompt Suno > real max
real_max  = np.max(real_vs_real_sims)
same_min  = np.min(same_prompt_sims)
if same_min > real_max:
    ax.axvspan(real_max, same_min, color="green", alpha=0.12,
               label="Clean separation zone")
ax.axvline(np.mean(same_prompt_sims),  color=SAME_C, ls="--", lw=1.2, alpha=0.7)
ax.axvline(np.mean(real_vs_real_sims), color=REAL_C,  ls="--", lw=1.2, alpha=0.7)

ax.set_xlabel("Cosine similarity", fontsize=11)
ax.set_ylabel("Cumulative probability", fontsize=11)
ax.set_title("CDF — do same-prompt pairs stochastically dominate?", fontsize=10)
ax.legend(fontsize=8.5)
ax.set_xlim(0.3, 1.02)
ax.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig(str(OUT_PNG), dpi=150, bbox_inches="tight")
print(f"\nFigure saved → {OUT_PNG}")

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  MERT COSINE SIMILARITY SUMMARY")
print("=" * 72)
print(f"{'Group':<30} {'n':>5}  mean ± std   [min, max]")
print("-" * 72)
for sims, label in [
    (same_prompt_sims,  "Same-prompt Suno pairs"),
    (diff_prompt_sims,  "Diff-prompt Suno pairs"),
    (real_vs_real_sims, "Real vs Real"),
    (suno_vs_real_sims, "Suno vs Real (cross)"),
]:
    print(f"  {label:<28} {len(sims):>5}  {fmt(sims)}")

print()
print("── Hypothesis test: same-prompt > real-vs-real ──")
t_stat, p_val = stats.ttest_ind(same_prompt_sims, real_vs_real_sims,
                                 alternative="greater")
mw_stat, mw_p = stats.mannwhitneyu(same_prompt_sims, real_vs_real_sims,
                                    alternative="greater")
print(f"  Welch t-test (one-sided):      t={t_stat:.3f}, p={p_val:.4f}")
print(f"  Mann-Whitney U (one-sided):    U={mw_stat:.0f}, p={mw_p:.4f}")

print()
print("── Per-pair breakdown ──")
print(f"  {'Prompt title':<38} {'Similarity':>10}")
print(f"  {'─'*50}")
for (a, b), s in sorted(zip(suno_pairs, same_prompt_sims), key=lambda x: -x[1]):
    print(f"  {canonical(a):<38} {s:>10.4f}")

print()
print("── Real vs Real breakdown ──")
for (a, b), s in sorted(
        zip(itertools.combinations(real_files, 2), real_vs_real_sims),
        key=lambda x: -x[1]):
    print(f"  {a.stem[:30]:<30}  ↔  {b.stem[:30]:<30}  {s:.4f}")

print()
clean_sep = np.min(same_prompt_sims) > np.max(real_vs_real_sims)
print(f"  Clean separation (all same-prompt > all real pairs): {clean_sep}")
print(f"  Same-prompt mean vs real mean gap: "
      f"{np.mean(same_prompt_sims) - np.mean(real_vs_real_sims):+.4f}")
print("=" * 72)
