"""
Part 2 — 1294-Dim Dual-Encoder Feature Pipeline
================================================
Build a richer 1294-dim feature set combining MERT, CLAP, Chroma, and
two auxiliary descriptors. Augment MIPPIA training pairs with SONICS
AI-generated music data and construct the full training pair table.

Feature layout (per 10-second segment):
    dims   0 :  768  → MERT-v1-95M   (music transformer)
    dims 768 : 1280  → CLAP           (audio-language contrastive model)
    dims 1280: 1292  → Chroma CQT     (pitch-class, key-invariant)
    dim  1292        → HP ratio        (harmonic/percussive energy, dB)
    dim  1293        → Spectral flatness (dB; higher for AI-generated audio)
    Total            → 1294 dims

Key finding: richer features did NOT beat the 782-dim baseline.
Primary cause: SONICS training pairs teach a different similarity
notion (generative provenance) than MIPPIA evaluation tests (compositional
plagiarism).

Dependencies:
    pip install numpy pandas torch torchaudio transformers librosa
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset

# ── Feature dimensions ────────────────────────────────────────────────────────

MERT_DIM    = 768
CLAP_DIM    = 512
CHROMA_DIM  = 12
HARM_DIM    = 1
FLAT_DIM    = 1
FEATURE_DIM = MERT_DIM + CLAP_DIM + CHROMA_DIM + HARM_DIM + FLAT_DIM  # 1294

WINDOW_SEC  = 10   # segment length
HOP_SEC     = 5    # hop between segment starts
MIN_SEG_SEC = 3.0  # minimum tail segment to include


# ── Cache validation ──────────────────────────────────────────────────────────

def validate_feature_cache(
    feature_dir: Path, expected_dim: int = FEATURE_DIM
) -> dict[str, np.ndarray]:
    """
    Load all .npy segment feature arrays and assert shape.

    Returns stem → array of shape [N_segments, expected_dim].

    This check caught the silent 782→1294 dimension mismatch in the
    repository's default training loader.
    """
    cache = {}
    for f in sorted(feature_dir.glob("*.npy")):
        arr = np.load(f).astype(np.float32)
        assert arr.ndim == 2, f"{f.name}: expected 2-D array, got {arr.shape}"
        assert arr.shape[1] == expected_dim, (
            f"{f.name}: expected dim {expected_dim}, got {arr.shape[1]}"
        )
        cache[f.stem] = arr
    return cache


# ── Dataset ───────────────────────────────────────────────────────────────────

class FeaturePairDataset(Dataset):
    """
    Load (feat_path_a, feat_path_b, label) rows from the feature-pair CSV.

    Each .npy file is a [N_segments, 1294] array. Mean-pooled to [1294]
    here to match the training regime of the Siamese MLP.

    Replaces the broken repository loader that fed 782-dim tensors into
    a 1294-dim model (see data/manifests/unified_1294_freeze_manifest.json).
    """

    def __init__(self, csv_path: str) -> None:
        self.df = pd.read_csv(csv_path)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple:
        row = self.df.iloc[idx]
        # [N_segments, 1294] → mean-pool → [1294]
        feat_a = np.load(row["feat_path_a"]).mean(axis=0).astype(np.float32)
        feat_b = np.load(row["feat_path_b"]).mean(axis=0).astype(np.float32)
        label  = float(row["label"])
        return (
            torch.from_numpy(feat_a),
            torch.from_numpy(feat_b),
            torch.tensor(label),
        )


# ── Training pair composition ─────────────────────────────────────────────────
# Source: data/manifests/complex_mippia_sonics_baseaware_manifest.json
#         data/pairs/complex_mippia_sonics_feature_pairs_1294.csv

PAIR_COMPOSITION = {
    "mippia_positive":          39,   # label 1 — compositional plagiarism/remake
    "mippia_negative":         117,   # label 0 — unrelated MIPPIA pairs
    "sonics_same_base_positive": 80,  # label 1 — same AI generative base ID
    "sonics_diff_base_negative": 80,  # label 0 — different AI base ID
    "mippia_sonics_negative":  156,   # label 0 — MIPPIA track vs SONICS track
    # Total: 472 pairs (119 pos / 353 neg → 3:1 imbalance)
}

# ── Recall collapse ───────────────────────────────────────────────────────────
# The unweighted model trained on these 472 pairs converged (43 epochs,
# val loss 0.477) but defaulted to predicting all negatives due to the
# 3:1 class imbalance. Recall was near zero. Fix: Part 3.
#
# SONICS distributional mismatch:
# - SONICS positives: two variants from the same generative base ID
# - MIPPIA positives: compositional borrowing between real tracks
# These are different similarity relations. Training on SONICS teaches
# the model to detect AI provenance, not musical plagiarism.
