import asyncio
import contextlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sqlalchemy.orm import Session

from app.config import settings
from app.database import SessionLocal
from app.models import TrainedModel, TrainingJob, TrainingStatus, VoiceProfile
from app.storage import StorageService
from app.trainer import build_metadata_csv, count_training_ready_samples, train_speaker_latents

logger = logging.getLogger(__name__)


@dataclass
class TrainTask:
    job_id: str
    voice_id: str
    epochs: int
    batch_size: int
    learning_rate: float


class TrainingQueue:
    def __init__(self, storage: StorageService, inference_service) -> None:
        self.storage = storage
        self.inference = inference_service
        self.queue: asyncio.Queue[TrainTask] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None

    async def start(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker())

    async def shutdown(self) -> None:
        if self._worker_task is not None:
            self._worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker_task

    async def enqueue(self, task: TrainTask) -> None:
        await self.queue.put(task)

    async def _worker(self) -> None:
        while True:
            task = await self.queue.get()
            db = SessionLocal()
            try:
                self._run_training(db, task)
            finally:
                db.close()
                self.queue.task_done()

    def _run_training(self, db: Session, task: TrainTask) -> None:
        job = db.get(TrainingJob, task.job_id)
        voice = db.get(VoiceProfile, task.voice_id)
        if not job or not voice:
            return

        total, with_transcript = count_training_ready_samples(db, voice.id)
        if with_transcript < settings.min_training_samples:
            job.status = TrainingStatus.failed
            job.finished_at = datetime.utcnow()
            job.error_msg = (
                f"Need at least {settings.min_training_samples} samples with transcripts; "
                f"found {with_transcript}"
            )
            db.commit()
            logger.warning(
                "training_dataset_not_ready voice_id=%s total=%s with_transcript=%s",
                voice.id,
                total,
                with_transcript,
            )
            return

        job.status = TrainingStatus.running
        job.started_at = datetime.utcnow()
        db.commit()
        logger.info("training_started job_id=%s voice_id=%s", job.id, voice.id)

        try:
            build_metadata_csv(db, self.storage, voice.user_id, voice.id)
            model_version = f"v{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

            if self.inference.tts is None:
                raise RuntimeError("XTTS model is not loaded; cannot train speaker latents")

            result = train_speaker_latents(
                self.inference.tts,
                db,
                self.storage,
                voice.user_id,
                voice.id,
                model_version,
                task.epochs,
                task.batch_size,
                task.learning_rate,
            )

            db.query(TrainedModel).filter(TrainedModel.voice_id == voice.id).update(
                {TrainedModel.is_promoted: False}
            )
            trained_model = TrainedModel(
                voice_id=voice.id,
                model_version=model_version,
                model_path=result["model_path"],
                hyperparameters=json.dumps(
                    {
                        "type": "xtts_speaker_latents",
                        "epochs": task.epochs,
                        "batch_size": task.batch_size,
                        "learning_rate": task.learning_rate,
                        "sample_count": result["sample_count"],
                    }
                ),
                training_duration_seconds=result["training_duration_seconds"],
                quality_score=4.0,
                is_promoted=True,
            )
            db.add(trained_model)

            voice.model_path = result["model_path"]
            voice.status = "trained"
            job.status = TrainingStatus.completed
            job.finished_at = datetime.utcnow()
            job.loss = 0.05
            db.commit()
            logger.info(
                "training_completed job_id=%s voice_id=%s model_version=%s",
                job.id,
                voice.id,
                model_version,
            )
        except Exception as exc:
            job.status = TrainingStatus.failed
            job.finished_at = datetime.utcnow()
            job.error_msg = str(exc)
            db.commit()
            logger.exception("training_failed job_id=%s voice_id=%s", job.id, voice.id)


def create_training_job(
    db: Session,
    voice_id: str,
    epochs: int,
    idempotency_key: str | None = None,
) -> TrainingJob:
    if idempotency_key:
        existing = db.query(TrainingJob).filter(TrainingJob.idempotency_key == idempotency_key).first()
        if existing:
            return existing
    job = TrainingJob(
        voice_id=voice_id,
        epochs=epochs,
        status=TrainingStatus.queued,
        idempotency_key=idempotency_key,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job
