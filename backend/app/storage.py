from pathlib import Path

from app.config import settings


class StorageService:
    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = (base_dir or settings.voices_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def voice_root(self, user_id: str, voice_id: str) -> Path:
        root = self.base_dir / user_id / voice_id
        root.mkdir(parents=True, exist_ok=True)
        return root

    def samples_dir(self, user_id: str, voice_id: str) -> Path:
        path = self.voice_root(user_id, voice_id) / "samples"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def processed_dir(self, user_id: str, voice_id: str) -> Path:
        path = self.voice_root(user_id, voice_id) / "processed"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def models_dir(self, user_id: str, voice_id: str, model_version: str | None = None) -> Path:
        root = self.voice_root(user_id, voice_id) / "models"
        root.mkdir(parents=True, exist_ok=True)
        if model_version is None:
            return root
        path = root / model_version
        path.mkdir(parents=True, exist_ok=True)
        return path

    def outputs_dir(self, user_id: str, voice_id: str) -> Path:
        path = self.voice_root(user_id, voice_id) / "outputs"
        path.mkdir(parents=True, exist_ok=True)
        return path
