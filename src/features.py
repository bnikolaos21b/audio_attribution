"""
Audio feature extraction pipeline for music attribution.

Design rationale
----------------
MERT-v1-95M (768-dim):
    Music-specific transformer trained on 1M hours of audio.
    Outperforms CLAP on music understanding tasks because it was trained
    exclusively on music rather than general audio-text pairs.
    We use the final hidden state mean-pooled over time.

CLAP / laion-clap-htsat-unfused (512-dim):
    Contrastive Language-Audio Pre-training model.  Its audio embeddings
    capture high-level semantic similarity across different renditions of the
    same musical material, complementing the low-level pattern matching of
    MERT.  Segments are resampled to 48 kHz (CLAP requirement) before
    processing.  Embeddings are L2-normalised to unit vectors.

Chroma CQT (12-dim):
    Captures tonal / harmonic content (pitch-class distribution).
    Mean-centered across the 12 bins so that transposed covers yield
    a similar profile — removes key bias without losing melodic shape.

HPSS Harmonic/Percussive Ratio (1-dim):
    HPSS-based proxy for how strongly harmonic vs. percussive a segment is.
    This is not classical Harmonic-to-Noise Ratio (HNR). It is computed as:
        10 * log10(E_harmonic / E_percussive)

Spectral Flatness (1-dim, log-scale):
    Measures how "noise-like" vs. "tone-like" a spectrum is.
    AI generators tend to produce more spectrally uniform output (higher
    flatness) than acoustic instruments.

Full feature vector layout (1294-dim per segment):
    dims   0 :  768  →  MERT
    dims 768 : 1280  →  CLAP
    dims 1280: 1292  →  Chroma CQT
    dim  1292        →  Harmonic/Percussive ratio
    dim  1293        →  Spectral Flatness

Windowing strategy:
    Full tracks are sliced into overlapping 10-second windows (5-second hop).
    This avoids the window-bias problem (only comparing first N seconds)
    and returns a per-segment feature matrix that downstream comparators
    can aggregate intelligently (e.g., top-K segment similarity).

Cache files:
    {stem}.npy      — full 1294-dim feature matrix (n_segments × 1294)
    {stem}_clap.npy — CLAP-only feature matrix (n_segments × 512)
                      Derived from {stem}.npy when available, so CLAP is
                      never computed twice for the same track.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import librosa
import numpy as np
import torch
import torchaudio

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

MERT_MODEL_NAME: str = "m-a-p/MERT-v1-95M"
MERT_SAMPLE_RATE: int = 24_000
CLAP_MODEL_NAME: str = "laion/clap-htsat-unfused"
CLAP_SAMPLE_RATE: int = 48_000
WINDOW_SEC: int = 10
HOP_SEC: int = 5
MIN_SEG_SEC: float = 3.0

MERT_DIM: int = 768
CLAP_DIM: int = 512
CHROMA_DIM: int = 12
HARM_PERC_RATIO_DIM: int = 1
FLATNESS_DIM: int = 1
FEATURE_DIM: int = (
    MERT_DIM + CLAP_DIM + CHROMA_DIM + HARM_PERC_RATIO_DIM + FLATNESS_DIM
)  # 1294


# ─── Low-level helpers ────────────────────────────────────────────────────────

def load_audio(path: str, target_sr: int = MERT_SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    """Load an audio file, convert to mono, and resample."""
    waveform, sr = torchaudio.load(path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    return waveform.squeeze(0).numpy().astype(np.float32), target_sr


def segment_audio(
    audio: np.ndarray,
    sr: int,
    window_sec: int = WINDOW_SEC,
    hop_sec: int = HOP_SEC,
    min_seg_sec: float = MIN_SEG_SEC,
) -> List[np.ndarray]:
    """Slice a waveform into overlapping fixed-length windows.

    Shorter tracks are zero-padded to one full window so every track yields
    at least one segment. A padded tail is added only if there is true
    uncovered remainder after the last full window and that remainder is
    at least `min_seg_sec` long.
    """
    win_samples = window_sec * sr
    hop_samples = hop_sec * sr
    min_samples = int(min_seg_sec * sr)

    n = len(audio)
    segments: List[np.ndarray] = []

    if n <= win_samples:
        padded = np.zeros(win_samples, dtype=np.float32)
        padded[:n] = audio[:n]
        return [padded]

    starts = list(range(0, n - win_samples + 1, hop_samples))
    for start in starts:
        segments.append(audio[start:start + win_samples])

    last_full_end = starts[-1] + win_samples
    remainder = n - last_full_end

    if remainder >= min_samples:
        padded = np.zeros(win_samples, dtype=np.float32)
        padded[:remainder] = audio[last_full_end:]
        segments.append(padded)

    return segments


# ─── Per-segment extractors ───────────────────────────────────────────────────

def extract_mert_embeddings(
    segments: List[np.ndarray],
    sr: int,
    processor,
    model,
    device: str = "cpu",
) -> np.ndarray:
    """Extract MERT embeddings for a list of audio segments."""
    assert sr == MERT_SAMPLE_RATE, f"Expected sample rate {MERT_SAMPLE_RATE}, got {sr}"

    embeddings: List[np.ndarray] = []
    model.eval()

    with torch.no_grad():
        for seg in segments:
            inputs = processor(
                seg,
                sampling_rate=sr,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            hidden = outputs.last_hidden_state

            assert hidden.ndim == 3, f"Unexpected MERT hidden shape: {hidden.shape}"
            assert hidden.shape[-1] == MERT_DIM, (
                f"Expected MERT dim {MERT_DIM}, got {hidden.shape[-1]}"
            )

            emb = hidden.mean(dim=1).squeeze(0).cpu().numpy()
            assert emb.shape == (MERT_DIM,), f"Unexpected pooled embedding shape: {emb.shape}"
            embeddings.append(emb.astype(np.float32))

    arr = np.array(embeddings, dtype=np.float32)
    assert arr.ndim == 2 and arr.shape[1] == MERT_DIM, f"Bad MERT array shape: {arr.shape}"
    return arr


def extract_chroma_cqt(
    segments: List[np.ndarray],
    sr: int,
) -> np.ndarray:
    """Compute mean-centered chroma CQT for each segment."""
    chromas: List[np.ndarray] = []

    for seg in segments:
        chroma = librosa.feature.chroma_cqt(y=seg, sr=sr, n_chroma=12)
        mean_chroma = chroma.mean(axis=1)
        mean_chroma -= mean_chroma.mean()
        chromas.append(mean_chroma.astype(np.float32))

    arr = np.array(chromas, dtype=np.float32)
    assert arr.ndim == 2 and arr.shape[1] == CHROMA_DIM, f"Bad chroma shape: {arr.shape}"
    return arr


def extract_harmonic_percussive_ratio(
    segments: List[np.ndarray],
    sr: int,
) -> np.ndarray:
    """Compute HPSS harmonic/percussive energy ratio (dB) for each segment.

    This is an HPSS-based proxy for harmonic-vs-percussive content.
    It is not classical Harmonic-to-Noise Ratio (HNR).
    """
    _ = sr
    ratios: List[float] = []
    eps = 1e-10

    for seg in segments:
        rms = float(np.sqrt(np.mean(seg ** 2)))
        if rms < 1e-4:
            ratios.append(0.0)
            continue

        harmonic, percussive = librosa.effects.hpss(seg)
        e_harm = float(np.mean(harmonic ** 2))
        e_perc = float(np.mean(percussive ** 2))
        ratio_db = 10.0 * np.log10((e_harm + eps) / (e_perc + eps))
        ratios.append(ratio_db)

    arr = np.array(ratios, dtype=np.float32).reshape(-1, 1)
    assert arr.ndim == 2 and arr.shape[1] == HARM_PERC_RATIO_DIM, (
        f"Bad harmonic/percussive ratio shape: {arr.shape}"
    )
    return arr


def extract_spectral_flatness(
    segments: List[np.ndarray],
    sr: int,
) -> np.ndarray:
    """Compute mean spectral flatness (dB) for each segment."""
    _ = sr
    flatnesses: List[float] = []

    for seg in segments:
        flat = librosa.feature.spectral_flatness(y=seg)
        mean_flat = float(flat.mean())
        flat_db = 10.0 * np.log10(mean_flat + 1e-10)
        flatnesses.append(flat_db)

    arr = np.array(flatnesses, dtype=np.float32).reshape(-1, 1)
    assert arr.ndim == 2 and arr.shape[1] == FLATNESS_DIM, f"Bad flatness shape: {arr.shape}"
    return arr


# ─── High-level extractor class ───────────────────────────────────────────────

class FeatureExtractor:
    """Full feature extraction pipeline."""

    def __init__(
        self,
        device: Optional[str] = None,
        mert_model_name: str = MERT_MODEL_NAME,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.mert_model_name = mert_model_name
        self._processor = None
        self._model = None
        self._clap_processor = None
        self._clap_model = None
        logger.info("FeatureExtractor initialised on device=%s", self.device)

    def _ensure_mert_loaded(self) -> None:
        """Lazily load MERT processor and model."""
        if self._processor is not None:
            return

        from transformers import AutoModel, AutoProcessor

        logger.info("Loading MERT model: %s", self.mert_model_name)
        self._processor = AutoProcessor.from_pretrained(
            self.mert_model_name, trust_remote_code=True
        )
        self._model = AutoModel.from_pretrained(
            self.mert_model_name, trust_remote_code=True
        ).to(self.device)
        self._model.eval()
        logger.info("MERT loaded successfully.")

    def _ensure_clap_loaded(self) -> None:
        """Lazily load CLAP processor and model."""
        if self._clap_processor is not None:
            return

        from transformers import ClapModel, ClapProcessor  # type: ignore

        logger.info("Loading CLAP model: %s", CLAP_MODEL_NAME)
        self._clap_processor = ClapProcessor.from_pretrained(CLAP_MODEL_NAME)
        self._clap_model = ClapModel.from_pretrained(CLAP_MODEL_NAME).to(self.device)
        self._clap_model.eval()
        logger.info("CLAP loaded successfully.")

    def _get_clap_embedding(self, segment_audio: np.ndarray) -> np.ndarray:
        """Return a 512-dim L2-normalised CLAP embedding for one audio segment.

        Parameters
        ----------
        segment_audio : np.ndarray
            1-D float32 waveform at MERT_SAMPLE_RATE (24 kHz).

        Returns
        -------
        np.ndarray
            Shape (512,), unit-normalised float32.
        """
        self._ensure_clap_loaded()

        # Resample 24 kHz → 48 kHz (CLAP requirement).
        wav_tensor = torch.tensor(segment_audio, dtype=torch.float32).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(
            orig_freq=MERT_SAMPLE_RATE, new_freq=CLAP_SAMPLE_RATE
        )
        audio_48k = resampler(wav_tensor).squeeze(0).numpy()

        with torch.no_grad():
            inputs = self._clap_processor(
                audio=[audio_48k],
                sampling_rate=CLAP_SAMPLE_RATE,
                return_tensors="pt",
            )
            input_features = inputs["input_features"].to(self.device)
            is_longer = inputs.get("is_longer")
            if is_longer is not None:
                is_longer = is_longer.to(self.device)

            audio_embed = self._clap_model.audio_model(
                input_features=input_features,
                is_longer=is_longer,
            ).pooler_output  # (1, audio_hidden_size)

            projected = self._clap_model.audio_projection(audio_embed)  # (1, 512)
            normalized = projected / (projected.norm(dim=-1, keepdim=True) + 1e-8)

        emb = normalized.squeeze(0).cpu().numpy().astype(np.float32)
        assert emb.shape == (CLAP_DIM,), f"Unexpected CLAP embedding shape: {emb.shape}"
        return emb

    def extract(
        self,
        path: str,
        window_sec: int = WINDOW_SEC,
        hop_sec: int = HOP_SEC,
    ) -> np.ndarray:
        """Extract the full feature matrix for a single audio file."""
        self._ensure_mert_loaded()
        self._ensure_clap_loaded()

        audio, sr = load_audio(path, target_sr=MERT_SAMPLE_RATE)
        assert sr == MERT_SAMPLE_RATE, f"Expected {MERT_SAMPLE_RATE}, got {sr}"

        segments = segment_audio(audio, sr, window_sec=window_sec, hop_sec=hop_sec)
        n_seg = len(segments)
        logger.debug("Extracted %d segments from %s", n_seg, Path(path).name)

        mert = extract_mert_embeddings(
            segments, sr, self._processor, self._model, self.device
        )
        clap = np.array(
            [self._get_clap_embedding(seg) for seg in segments],
            dtype=np.float32,
        )
        chroma = extract_chroma_cqt(segments, sr)
        harm_perc_ratio = extract_harmonic_percussive_ratio(segments, sr)
        flatness = extract_spectral_flatness(segments, sr)

        assert mert.ndim == 2 and mert.shape[1] == MERT_DIM
        assert clap.ndim == 2 and clap.shape[1] == CLAP_DIM
        assert chroma.ndim == 2 and chroma.shape[1] == CHROMA_DIM
        assert harm_perc_ratio.ndim == 2 and harm_perc_ratio.shape[1] == HARM_PERC_RATIO_DIM
        assert flatness.ndim == 2 and flatness.shape[1] == FLATNESS_DIM

        n_rows = {
            mert.shape[0], clap.shape[0],
            chroma.shape[0], harm_perc_ratio.shape[0], flatness.shape[0],
        }
        assert len(n_rows) == 1, f"Inconsistent segment counts: {n_rows}"

        features = np.concatenate([mert, clap, chroma, harm_perc_ratio, flatness], axis=1)
        assert features.ndim == 2 and features.shape[1] == FEATURE_DIM == 1294, (
            f"Expected feature dim 1294, got {features.shape}"
        )
        return features.astype(np.float32)

    def extract_and_cache(
        self,
        path: str,
        cache_path: str,
        force: bool = False,
    ) -> np.ndarray:
        """Extract features and save/load from a .npy cache file."""
        cache = Path(cache_path)
        if cache.exists() and not force:
            logger.debug("Loading cached features from %s", cache_path)
            return np.load(cache_path)

        features = self.extract(path)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, features)
        logger.debug("Saved features to %s", cache_path)
        return features

    def extract_clap_and_cache(
        self,
        path: str,
        cache_path: str,
        force: bool = False,
    ) -> np.ndarray:
        """Return per-segment CLAP features (n_segments × 512), cached to disk.

        If a full 1294-dim feature cache for this track already exists in the
        same directory, the CLAP slice (dims MERT_DIM : MERT_DIM+CLAP_DIM) is
        extracted from it directly to avoid recomputing CLAP embeddings.

        Parameters
        ----------
        path : str
            Audio file path.
        cache_path : str
            Destination for the ``_clap.npy`` cache file.
        force : bool
            Recompute even if the cache file exists.

        Returns
        -------
        np.ndarray
            Shape (n_segments, 512), float32.
        """
        cache = Path(cache_path)
        if cache.exists() and not force:
            logger.debug("Loading cached CLAP features from %s", cache_path)
            return np.load(cache_path)

        # If the full feature cache exists for this track, slice CLAP from it
        # rather than running the CLAP model again.
        stem = Path(path).stem
        full_cache = cache.parent / f"{stem}.npy"
        if full_cache.exists() and not force:
            full = np.load(str(full_cache))
            if full.ndim == 2 and full.shape[1] == FEATURE_DIM:
                clap_embs = full[:, MERT_DIM: MERT_DIM + CLAP_DIM].astype(np.float32)
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_path, clap_embs)
                logger.debug("Sliced CLAP from full cache → %s", cache_path)
                return clap_embs

        # Fall back to computing CLAP embeddings from scratch.
        self._ensure_clap_loaded()
        audio, sr = load_audio(path, target_sr=MERT_SAMPLE_RATE)
        segments = segment_audio(audio, sr)
        clap_embs = np.array(
            [self._get_clap_embedding(seg) for seg in segments],
            dtype=np.float32,
        )
        assert clap_embs.ndim == 2 and clap_embs.shape[1] == CLAP_DIM, (
            f"Bad CLAP array shape: {clap_embs.shape}"
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, clap_embs)
        logger.debug("Saved CLAP features to %s", cache_path)
        return clap_embs

    def extract_clap_only(
        self,
        path: str,
        cache_path: Optional[str] = None,
    ) -> np.ndarray:
        """Return the mean-pooled 512-dim CLAP embedding for a track.

        Useful for ensemble approaches that operate at the track level rather
        than the segment level.

        Parameters
        ----------
        path : str
            Audio file path.
        cache_path : str, optional
            If provided, per-segment CLAP features are cached at this path
            (``{stem}_clap.npy``) before pooling.

        Returns
        -------
        np.ndarray
            Shape (512,), float32 — mean-pooled over all segments.
        """
        if cache_path is not None:
            clap_segments = self.extract_clap_and_cache(path, cache_path)
        else:
            self._ensure_clap_loaded()
            audio, sr = load_audio(path, target_sr=MERT_SAMPLE_RATE)
            segments = segment_audio(audio, sr)
            clap_segments = np.array(
                [self._get_clap_embedding(seg) for seg in segments],
                dtype=np.float32,
            )

        pooled = clap_segments.mean(axis=0).astype(np.float32)
        assert pooled.shape == (CLAP_DIM,), f"Unexpected pooled CLAP shape: {pooled.shape}"
        return pooled
