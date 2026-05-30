"""Reference sample selection and multi-reference audio helpers."""

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from sqlalchemy.orm import Session

from app.config import settings
from app.models import VoiceSample


def score_sample(sample: VoiceSample) -> float:
    """Higher score = better reference candidate."""
    duration = sample.duration_seconds
    score = 0.0
    if sample.transcript and sample.transcript.strip():
        score += 50.0
    if 6 <= duration <= 30:
        score += 40.0
    elif 3 <= duration < 6:
        score += 20.0
    elif 30 < duration <= 60:
        score += 25.0
    elif duration > 60:
        score += 10.0
    return score


def select_reference_samples(db: Session, voice_id: str, limit: int = 3) -> list[VoiceSample]:
    samples = (
        db.query(VoiceSample)
        .filter(VoiceSample.voice_id == voice_id)
        .order_by(VoiceSample.created_at.desc())
        .all()
    )
    if not samples:
        return []
    ranked = sorted(samples, key=score_sample, reverse=True)
    return ranked[:limit]


def build_combined_reference_wav(sample_paths: list[Path], output_path: Path) -> Path:
    """Concatenate short reference clips into one conditioning WAV."""
    if len(sample_paths) == 1:
        return sample_paths[0]

    chunks: list[np.ndarray] = []
    target_sr = settings.default_sample_rate
    for path in sample_paths:
        audio, _ = librosa.load(path, sr=target_sr, mono=True)
        if audio.size == 0:
            continue
        chunks.append(audio)
        # Short silence between clips.
        chunks.append(np.zeros(int(target_sr * 0.15), dtype=np.float32))

    if not chunks:
        raise ValueError("No valid audio in reference samples")

    combined = np.concatenate(chunks)
    max_samples = int(target_sr * 45)
    if combined.size > max_samples:
        combined = combined[:max_samples]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, combined, target_sr, subtype="PCM_16")
    return output_path
