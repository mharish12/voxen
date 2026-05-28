from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Voice Cloning TTS API"
    environment: str = "development"
    # Local default targets a host Postgres instance.
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/tts"
    api_prefix: str = "/api"
    voices_dir: Path = Path("voices")
    max_upload_size_mb: int = 25
    min_reference_duration_seconds: int = 3
    max_reference_duration_seconds: int = 300
    default_sample_rate: int = 24000
    max_text_length: int = 1200
    training_concurrency: int = 1

    model_config = SettingsConfigDict(env_file=".env", env_prefix="TTS_")


settings = Settings()
