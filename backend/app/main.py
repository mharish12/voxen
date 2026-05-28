import shutil
import time
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from sqlalchemy.orm import Session

from app.audio import preprocess_audio
from app.config import settings
from app.database import Base, engine, get_db
from app.inference import InferenceService, InferenceUnavailableError
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
logger = logging.getLogger(__name__)


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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.get("/api/users", response_model=list[UserRead])
def list_users(email: str | None = None, db: Session = Depends(get_db)):
    query = db.query(User)
    if email:
        query = query.filter(User.email.ilike(f"%{email.strip()}%"))
    return query.order_by(User.created_at.desc()).limit(20).all()


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
    logger.info(
        "upload_sample_request voice_id=%s filename=%s content_type=%s",
        voice_id,
        file.filename,
        file.content_type,
    )
    voice = db.get(VoiceProfile, voice_id)
    if voice is None:
        raise HTTPException(status_code=404, detail="Voice not found")
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing file name")
    lower_name = file.filename.lower()
    if not (lower_name.endswith(".wav") or lower_name.endswith(".mp3")):
        raise HTTPException(status_code=400, detail="Only .wav and .mp3 files are supported")
    allowed_content_types = {
        "audio/wav",
        "audio/x-wav",
        "audio/wave",
        "audio/mpeg",
        "audio/mp3",
        "application/octet-stream",
    }
    if file.content_type and file.content_type.lower() not in allowed_content_types:
        raise HTTPException(status_code=400, detail="Unsupported audio content type. Upload a valid WAV or MP3 file.")

    sample_id = str(uuid4())
    samples_dir = storage.samples_dir(voice.user_id, voice.id)
    raw_ext = ".mp3" if lower_name.endswith(".mp3") else ".wav"
    raw_path = samples_dir / f"{sample_id}{raw_ext}"
    with raw_path.open("wb") as fp:
        shutil.copyfileobj(file.file, fp)

    processed_dir = storage.processed_dir(voice.user_id, voice.id)
    processed_path = processed_dir / f"{sample_id}.wav"
    try:
        duration = preprocess_audio(raw_path, processed_path)
    except ValueError as exc:
        logger.warning(
            "upload_sample_rejected voice_id=%s sample_id=%s filename=%s detail=%s",
            voice_id,
            sample_id,
            file.filename,
            str(exc),
        )
        raw_path.unlink(missing_ok=True)
        processed_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception(
            "upload_sample_failed voice_id=%s sample_id=%s filename=%s",
            voice_id,
            sample_id,
            file.filename,
        )
        raw_path.unlink(missing_ok=True)
        processed_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Failed to read audio. Please upload a valid WAV or MP3 file.") from exc

    if duration < settings.min_reference_duration_seconds or duration > settings.max_reference_duration_seconds:
        logger.warning(
            "upload_sample_duration_invalid voice_id=%s sample_id=%s duration=%s",
            voice_id,
            sample_id,
            duration,
        )
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
    logger.info(
        "upload_sample_success voice_id=%s sample_id=%s duration=%s processed_path=%s",
        voice_id,
        sample_id,
        duration,
        processed_path,
    )
    return voice


@app.post("/api/synthesize")
def synthesize(payload: SynthesisRequestIn, db: Session = Depends(get_db)):
    request_started = time.perf_counter()
    step_started = request_started
    step_index = 0

    def log_step(name: str, **fields):
        nonlocal step_started, step_index
        now = time.perf_counter()
        step_elapsed_ms = int((now - step_started) * 1000)
        total_elapsed_ms = int((now - request_started) * 1000)
        step_index += 1
        joined_fields = " ".join(f"{k}={v}" for k, v in fields.items())
        logger.info(
            "synthesize_step step=%s name=%s step_elapsed_ms=%s total_elapsed_ms=%s %s",
            step_index,
            name,
            step_elapsed_ms,
            total_elapsed_ms,
            joined_fields,
        )
        step_started = now

    logger.info(
        "synthesize_request user_id=%s voice_id=%s language=%s text_chars=%s model_version=%s",
        payload.user_id,
        payload.voice_id,
        payload.language,
        len(payload.text),
        payload.model_version,
    )
    voice = db.get(VoiceProfile, payload.voice_id)
    log_step("voice_lookup", found=voice is not None)
    if voice is None:
        logger.warning("synthesize_voice_not_found voice_id=%s", payload.voice_id)
        raise HTTPException(status_code=404, detail="Voice not found")
    if len(payload.text) > settings.max_text_length:
        logger.warning("synthesize_text_too_long voice_id=%s text_chars=%s", payload.voice_id, len(payload.text))
        raise HTTPException(status_code=400, detail=f"Text length exceeds {settings.max_text_length} characters")

    sample = (
        db.query(VoiceSample)
        .filter(VoiceSample.voice_id == voice.id)
        .order_by(VoiceSample.created_at.desc())
        .first()
    )
    log_step("sample_lookup", found=sample is not None)
    if not sample:
        logger.warning("synthesize_no_reference_sample voice_id=%s", voice.id)
        raise HTTPException(status_code=400, detail="No reference sample available for voice")

    model_version = payload.model_version
    if model_version:
        model = (
            db.query(TrainedModel)
            .filter(TrainedModel.voice_id == voice.id, TrainedModel.model_version == model_version)
            .first()
        )
        log_step("model_lookup_specific", found=model is not None, model_version=model_version)
        if model is None:
            logger.warning("synthesize_model_not_found voice_id=%s model_version=%s", voice.id, model_version)
            raise HTTPException(status_code=404, detail="Requested model version not found")
    else:
        model = (
            db.query(TrainedModel)
            .filter(TrainedModel.voice_id == voice.id)
            .order_by(TrainedModel.is_promoted.desc(), TrainedModel.created_at.desc())
            .first()
        )
        model_version = model.model_version if model else None
        log_step("model_lookup_default", found=model is not None, selected_model_version=model_version)

    output_wav_path = storage.outputs_dir(voice.user_id, voice.id) / f"{uuid4()}.wav"
    output_mp3_path = output_wav_path.with_suffix(".mp3")
    try:
        inference.synthesize(payload.text, Path(sample.file_path), payload.language, output_wav_path)
        log_step("xtts_generation", wav_path=output_wav_path)
    except InferenceUnavailableError as exc:
        logger.error("synthesize_inference_unavailable voice_id=%s reason=%s", voice.id, str(exc))
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ImportError as exc:
        logger.exception("synthesize_dependency_missing voice_id=%s error=%s", voice.id, str(exc))
        raise HTTPException(
            status_code=500,
            detail="Missing runtime dependency for synthesis. Install torchcodec and restart backend.",
        ) from exc
    except Exception as exc:
        logger.exception("synthesize_failed voice_id=%s error=%s", voice.id, str(exc))
        raise HTTPException(status_code=500, detail="Synthesis failed. Check backend logs for details.") from exc

    try:
        AudioSegment.from_wav(output_wav_path).export(output_mp3_path, format="mp3")
        log_step("mp3_export", mp3_path=output_mp3_path)
    except (CouldntDecodeError, FileNotFoundError) as exc:
        logger.exception("synthesize_mp3_export_failed voice_id=%s wav_path=%s", voice.id, output_wav_path)
        output_wav_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to encode MP3 output. Ensure ffmpeg is installed and reachable.",
        ) from exc
    latency_ms = int((time.perf_counter() - request_started) * 1000)
    output_wav_path.unlink(missing_ok=True)
    log_step("cleanup_wav")

    req = SynthesisRequest(
        voice_id=voice.id,
        user_id=payload.user_id,
        input_text=payload.text,
        language=payload.language,
        output_path=str(output_mp3_path),
        model_version=model_version,
        latency_ms=latency_ms,
    )
    db.add(req)
    db.commit()
    log_step("db_commit")
    logger.info(
        "synthesize_success voice_id=%s output_path=%s latency_ms=%s",
        voice.id,
        output_mp3_path,
        latency_ms,
    )
    return FileResponse(path=output_mp3_path, media_type="audio/mpeg", filename="speech.mp3")


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
