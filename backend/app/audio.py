import tempfile
from pathlib import Path

import audioread
import librosa
import soundfile as sf

from app.config import settings


def preprocess_audio(input_path: Path, output_path: Path, target_sr: int | None = None) -> tuple[float, float]:
    """Return (trimmed_duration_seconds, original_duration_seconds)."""
    sample_rate = target_sr or settings.default_sample_rate
    try:
        audio, _ = librosa.load(input_path, sr=sample_rate, mono=True)
    except audioread.exceptions.NoBackendError as exc:
        if input_path.suffix.lower() == ".mp3":
            raise ValueError(
                "MP3 decoding backend is unavailable on the server. "
                "Install ffmpeg for MP3 support or upload WAV."
            ) from exc
        raise ValueError("Unsupported or invalid audio file. Please upload a valid WAV or MP3 recording.") from exc
    except (sf.LibsndfileError, EOFError) as exc:
        raise ValueError("Unsupported or invalid audio file. Please upload a valid WAV or MP3 recording.") from exc

    original_duration = float(librosa.get_duration(y=audio, sr=sample_rate))
    trimmed, _ = librosa.effects.trim(audio, top_db=25)
    if trimmed.size == 0:
        raise ValueError("Audio is silent after trimming")
    normalized = librosa.util.normalize(trimmed)
    sf.write(output_path, normalized, sample_rate, subtype="PCM_16")
    trimmed_duration = float(librosa.get_duration(y=normalized, sr=sample_rate))
    return trimmed_duration, original_duration


def probe_duration_seconds(input_path: Path) -> float:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "probe.wav"
        _, original_duration = preprocess_audio(input_path, tmp_path)
    return original_duration
