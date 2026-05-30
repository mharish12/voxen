"""Voice training: build speaker conditioning latents from labeled samples (CPU-friendly)."""

import json
import logging
import time
from pathlib import Path

import torch
from sqlalchemy.orm import Session

from app.config import settings
from app.models import VoiceSample
from app.storage import StorageService

logger = logging.getLogger(__name__)


def count_training_ready_samples(db: Session, voice_id: str) -> tuple[int, int]:
    samples = db.query(VoiceSample).filter(VoiceSample.voice_id == voice_id).all()
    total = len(samples)
    with_transcript = sum(1 for s in samples if s.transcript and s.transcript.strip())
    return total, with_transcript


def build_metadata_csv(db: Session, storage: StorageService, user_id: str, voice_id: str) -> Path:
    rows: list[str] = []
    for sample in db.query(VoiceSample).filter(VoiceSample.voice_id == voice_id).all():
        if not sample.transcript or not sample.transcript.strip():
            continue
        filename = Path(sample.file_path).name
        rows.append(f"{filename}|{sample.transcript.strip()}")
    metadata_path = storage.processed_dir(user_id, voice_id) / "metadata.csv"
    metadata_path.write_text("\n".join(rows), encoding="utf-8")
    return metadata_path


def train_speaker_latents(
    tts_model,
    db: Session,
    storage: StorageService,
    user_id: str,
    voice_id: str,
    model_version: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> dict:
    """
    CPU-friendly personalization: aggregate XTTS conditioning latents from all
  labeled samples. This produces a real checkpoint usable at inference time.
    """
    samples = [
        s
        for s in db.query(VoiceSample).filter(VoiceSample.voice_id == voice_id).all()
        if s.transcript and s.transcript.strip() and Path(s.file_path).exists()
    ]
    if len(samples) < settings.min_training_samples:
        raise ValueError(
            f"Need at least {settings.min_training_samples} samples with transcripts; found {len(samples)}"
        )

    audio_paths = [str(Path(s.file_path)) for s in samples]
    logger.info(
        "training_latents_start voice_id=%s sample_count=%s paths=%s",
        voice_id,
        len(audio_paths),
        audio_paths,
    )

    started = time.perf_counter()
    xtts = tts_model.synthesizer.tts_model
    # Coqui XTTS v2: conditioning params are method defaults, not instance attrs.
    gpt_cond_latent, speaker_embedding = xtts.get_conditioning_latents(audio_path=audio_paths)

    model_dir = storage.models_dir(user_id, voice_id, model_version)
    latents_path = model_dir / "speaker_latents.pth"
    config_path = model_dir / "config.json"

    torch.save(
        {
            "gpt_cond_latent": gpt_cond_latent.cpu(),
            "speaker_embedding": speaker_embedding.cpu(),
            "sample_count": len(audio_paths),
            "audio_paths": audio_paths,
        },
        latents_path,
    )
    config_path.write_text(
        json.dumps(
            {
                "type": "xtts_speaker_latents",
                "model": "xtts_v2",
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "sample_count": len(audio_paths),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    duration = int(time.perf_counter() - started)
    logger.info(
        "training_latents_done voice_id=%s model_version=%s duration_s=%s latents_path=%s",
        voice_id,
        model_version,
        duration,
        latents_path,
    )
    return {
        "model_path": str(latents_path),
        "config_path": str(config_path),
        "training_duration_seconds": duration,
        "sample_count": len(audio_paths),
    }
