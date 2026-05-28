from datetime import datetime
from enum import Enum
from uuid import uuid4

from sqlalchemy import DateTime, Enum as SqlEnum, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class TrainingStatus(str, Enum):
    queued = "queued"
    running = "running"
    failed = "failed"
    completed = "completed"


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    voices: Mapped[list["VoiceProfile"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class VoiceProfile(Base):
    __tablename__ = "voice_profiles"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    language: Mapped[str] = mapped_column(String(16), nullable=False, default="en")
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    model_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="ready")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    user: Mapped["User"] = relationship(back_populates="voices")
    samples: Mapped[list["VoiceSample"]] = relationship(back_populates="voice", cascade="all, delete-orphan")
    jobs: Mapped[list["TrainingJob"]] = relationship(back_populates="voice", cascade="all, delete-orphan")
    trained_models: Mapped[list["TrainedModel"]] = relationship(
        back_populates="voice",
        cascade="all, delete-orphan",
    )


class VoiceSample(Base):
    __tablename__ = "voice_samples"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    voice_id: Mapped[str] = mapped_column(String(36), ForeignKey("voice_profiles.id", ondelete="CASCADE"), index=True)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    transcript: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_seconds: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    voice: Mapped["VoiceProfile"] = relationship(back_populates="samples")


class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    voice_id: Mapped[str] = mapped_column(String(36), ForeignKey("voice_profiles.id", ondelete="CASCADE"), index=True)
    status: Mapped[TrainingStatus] = mapped_column(SqlEnum(TrainingStatus), nullable=False, default=TrainingStatus.queued)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    epochs: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    error_msg: Mapped[str | None] = mapped_column(Text, nullable=True)
    idempotency_key: Mapped[str | None] = mapped_column(String(128), nullable=True, unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    voice: Mapped["VoiceProfile"] = relationship(back_populates="jobs")


class TrainedModel(Base):
    __tablename__ = "trained_models"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    voice_id: Mapped[str] = mapped_column(String(36), ForeignKey("voice_profiles.id", ondelete="CASCADE"), index=True)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    model_path: Mapped[str] = mapped_column(Text, nullable=False)
    base_model: Mapped[str] = mapped_column(String(128), nullable=False, default="xtts_v2")
    hyperparameters: Mapped[str | None] = mapped_column(Text, nullable=True)
    training_duration_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_promoted: Mapped[bool] = mapped_column(default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    voice: Mapped["VoiceProfile"] = relationship(back_populates="trained_models")


class SynthesisRequest(Base):
    __tablename__ = "synthesis_requests"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    voice_id: Mapped[str] = mapped_column(String(36), ForeignKey("voice_profiles.id", ondelete="SET NULL"), nullable=True, index=True)
    user_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    input_text: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[str] = mapped_column(String(16), nullable=False, default="en")
    output_path: Mapped[str] = mapped_column(Text, nullable=False)
    model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
