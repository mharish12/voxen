from datetime import datetime

from pydantic import BaseModel, Field

from app.models import TrainingStatus


class VoiceProfileCreate(BaseModel):
    user_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=2, max_length=120)
    language: str = Field(default="en", min_length=2, max_length=16)


class UserCreate(BaseModel):
    email: str = Field(..., min_length=3, max_length=255)
    password_hash: str = Field(..., min_length=1, max_length=255)


class UserRead(BaseModel):
    id: str
    email: str
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class VoiceProfileRead(BaseModel):
    id: str
    user_id: str
    name: str
    language: str
    sample_count: int
    model_path: str | None
    status: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class VoiceSampleRead(BaseModel):
    id: str
    voice_id: str
    file_path: str
    transcript: str | None
    duration_seconds: float
    created_at: datetime

    class Config:
        from_attributes = True


class TrainingReadinessRead(BaseModel):
    sample_count: int
    with_transcript_count: int
    min_required: int
    target_count: int
    ready: bool


class SynthesisRequestIn(BaseModel):
    user_id: str | None = None
    voice_id: str
    text: str = Field(..., min_length=1, max_length=1200)
    language: str = Field(default="en", min_length=2, max_length=16)
    model_version: str | None = Field(default=None, max_length=64)
    use_reference_only: bool = False


class TrainingRequestIn(BaseModel):
    voice_id: str
    epochs: int = Field(default=50, ge=1, le=500)
    batch_size: int = Field(default=2, ge=1, le=16)
    learning_rate: float = Field(default=0.000005, gt=0, le=0.001)
    idempotency_key: str | None = Field(default=None, max_length=128)


class TrainingJobRead(BaseModel):
    id: str
    voice_id: str
    status: TrainingStatus
    started_at: datetime | None
    finished_at: datetime | None
    epochs: int
    loss: float | None
    error_msg: str | None
    created_at: datetime

    class Config:
        from_attributes = True


class TrainedModelRead(BaseModel):
    id: str
    voice_id: str
    model_version: str
    model_path: str
    base_model: str
    hyperparameters: str | None
    training_duration_seconds: int | None
    quality_score: float | None
    is_promoted: bool
    created_at: datetime

    class Config:
        from_attributes = True


class PromoteModelIn(BaseModel):
    model_version: str = Field(..., min_length=1, max_length=64)
