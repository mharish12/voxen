import shutil
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.audio import preprocess_audio
from app.config import settings
from app.database import Base, engine, get_db
from app.inference import InferenceService
from app.logging_setup import RequestContextMiddleware, RequestIdFilter, setup_logging
from app.migrations import run_migrations
from app.models import SynthesisRequest, TrainedModel, User, VoiceProfile, VoiceSample
from app.schemas import (
    PromoteModelIn,
    SynthesisRequestIn,
    TrainedModelRead,
    TrainingJobRead,
    TrainingRequestIn,
    UserCreate,
    UserRead,
    VoiceProfileCreate,
    VoiceProfileRead,
)
from app.storage import StorageService
from app.training import TrainTask, TrainingQueue, create_training_job

storage = StorageService()
inference = InferenceService()
training_queue = TrainingQueue(storage=storage)


@asynccontextmanager
async def lifespan(_: FastAPI):
    setup_logging()
    import logging

    logging.getLogger().addFilter(RequestIdFilter())
    run_migrations()
    Base.metadata.create_all(bind=engine)
    inference.warmup()
    await training_queue.start()
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.add_middleware(RequestContextMiddleware)


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/users", response_model=UserRead)
def create_user(payload: UserCreate, db: Session = Depends(get_db)):
    exists = db.query(User).filter(User.email == payload.email).first()
    if exists is not None:
        raise HTTPException(status_code=409, detail="Email already exists")
    user = User(email=payload.email, password_hash=payload.password_hash)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@app.post("/api/voices", response_model=VoiceProfileRead)
def create_voice_profile(payload: VoiceProfileCreate, db: Session = Depends(get_db)):
    user = db.get(User, payload.user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    voice = VoiceProfile(user_id=payload.user_id, name=payload.name, language=payload.language)
    db.add(voice)
    db.commit()
    db.refresh(voice)
    return voice


@app.get("/api/voices", response_model=list[VoiceProfileRead])
def list_voice_profiles(user_id: str | None = None, db: Session = Depends(get_db)):
    query = db.query(VoiceProfile)
    if user_id:
        query = query.filter(VoiceProfile.user_id == user_id)
    return query.order_by(VoiceProfile.created_at.desc()).all()


@app.delete("/api/voices/{voice_id}", status_code=204)
def delete_voice_profile(voice_id: str, db: Session = Depends(get_db)):
    voice = db.get(VoiceProfile, voice_id)
    if voice is None:
        raise HTTPException(status_code=404, detail="Voice not found")
    db.delete(voice)
    db.commit()


@app.post("/api/voices/{voice_id}/samples", response_model=VoiceProfileRead)
async def upload_reference_sample(
    voice_id: str,
    file: UploadFile = File(...),
    transcript: str | None = None,
    db: Session = Depends(get_db),
):
    voice = db.get(VoiceProfile, voice_id)
    if voice is None:
        raise HTTPException(status_code=404, detail="Voice not found")
    if not file.filename or not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported")

    sample_id = str(uuid4())
    samples_dir = storage.samples_dir(voice.user_id, voice.id)
    raw_path = samples_dir / f"{sample_id}.wav"
    with raw_path.open("wb") as fp:
        shutil.copyfileobj(file.file, fp)

    processed_dir = storage.processed_dir(voice.user_id, voice.id)
    processed_path = processed_dir / f"{sample_id}.wav"
    try:
        duration = preprocess_audio(raw_path, processed_path)
    except ValueError as exc:
        raw_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if duration < settings.min_reference_duration_seconds or duration > settings.max_reference_duration_seconds:
        raw_path.unlink(missing_ok=True)
        processed_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail=f"Audio duration must be between {settings.min_reference_duration_seconds}s and {settings.max_reference_duration_seconds}s",
        )

    sample = VoiceSample(
        id=sample_id,
        voice_id=voice.id,
        file_path=str(processed_path),
        transcript=transcript,
        duration_seconds=duration,
    )
    db.add(sample)
    voice.sample_count += 1
    voice.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(voice)
    return voice


@app.post("/api/synthesize")
def synthesize(payload: SynthesisRequestIn, db: Session = Depends(get_db)):
    voice = db.get(VoiceProfile, payload.voice_id)
    if voice is None:
        raise HTTPException(status_code=404, detail="Voice not found")
    if len(payload.text) > settings.max_text_length:
        raise HTTPException(status_code=400, detail=f"Text length exceeds {settings.max_text_length} characters")

    sample = (
        db.query(VoiceSample)
        .filter(VoiceSample.voice_id == voice.id)
        .order_by(VoiceSample.created_at.desc())
        .first()
    )
    if not sample:
        raise HTTPException(status_code=400, detail="No reference sample available for voice")

    model_version = payload.model_version
    if model_version:
        model = (
            db.query(TrainedModel)
            .filter(TrainedModel.voice_id == voice.id, TrainedModel.model_version == model_version)
            .first()
        )
        if model is None:
            raise HTTPException(status_code=404, detail="Requested model version not found")
    else:
        model = (
            db.query(TrainedModel)
            .filter(TrainedModel.voice_id == voice.id)
            .order_by(TrainedModel.is_promoted.desc(), TrainedModel.created_at.desc())
            .first()
        )
        model_version = model.model_version if model else None

    output_name = f"{uuid4()}.wav"
    output_path = storage.outputs_dir(voice.user_id, voice.id) / output_name
    started = time.perf_counter()
    inference.synthesize(payload.text, Path(sample.file_path), payload.language, output_path)
    latency_ms = int((time.perf_counter() - started) * 1000)

    req = SynthesisRequest(
        voice_id=voice.id,
        user_id=payload.user_id,
        input_text=payload.text,
        language=payload.language,
        output_path=str(output_path),
        model_version=model_version,
        latency_ms=latency_ms,
    )
    db.add(req)
    db.commit()
    return FileResponse(path=output_path, media_type="audio/wav", filename="speech.wav")


@app.post("/api/train", response_model=TrainingJobRead)
async def train_voice(
    payload: TrainingRequestIn,
    x_idempotency_key: str | None = Header(default=None),
    db: Session = Depends(get_db),
):
    voice = db.get(VoiceProfile, payload.voice_id)
    if voice is None:
        raise HTTPException(status_code=404, detail="Voice not found")

    key = payload.idempotency_key or x_idempotency_key
    job = create_training_job(db, payload.voice_id, payload.epochs, key)
    await training_queue.enqueue(
        TrainTask(
            job_id=job.id,
            voice_id=payload.voice_id,
            epochs=payload.epochs,
            batch_size=payload.batch_size,
            learning_rate=payload.learning_rate,
        )
    )
    db.refresh(job)
    return job


@app.get("/api/train/{job_id}", response_model=TrainingJobRead)
def get_training_job(job_id: str, db: Session = Depends(get_db)):
    from app.models import TrainingJob

    job = db.get(TrainingJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Training job not found")
    return job


@app.get("/api/voices/{voice_id}/models", response_model=list[TrainedModelRead])
def list_models(voice_id: str, db: Session = Depends(get_db)):
    return (
        db.query(TrainedModel)
        .filter(TrainedModel.voice_id == voice_id)
        .order_by(TrainedModel.created_at.desc())
        .all()
    )


@app.post("/api/voices/{voice_id}/models/promote", response_model=TrainedModelRead)
def promote_model(voice_id: str, payload: PromoteModelIn, db: Session = Depends(get_db)):
    target = (
        db.query(TrainedModel)
        .filter(TrainedModel.voice_id == voice_id, TrainedModel.model_version == payload.model_version)
        .first()
    )
    if target is None:
        raise HTTPException(status_code=404, detail="Model not found")

    db.query(TrainedModel).filter(TrainedModel.voice_id == voice_id).update({TrainedModel.is_promoted: False})
    target.is_promoted = True
    voice = db.get(VoiceProfile, voice_id)
    if voice is not None:
        voice.model_path = target.model_path
    db.commit()
    db.refresh(target)
    return target
