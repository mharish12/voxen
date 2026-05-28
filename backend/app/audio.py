import tempfile
from pathlib import Path

import librosa
import soundfile as sf

from app.config import settings


def preprocess_audio(input_path: Path, output_path: Path, target_sr: int | None = None) -> float:
    sample_rate = target_sr or settings.default_sample_rate
    audio, _ = librosa.load(input_path, sr=sample_rate, mono=True)
    trimmed, _ = librosa.effects.trim(audio, top_db=25)
    if trimmed.size == 0:
        raise ValueError("Audio is silent after trimming")
    normalized = librosa.util.normalize(trimmed)
    sf.write(output_path, normalized, sample_rate, subtype="PCM_16")
    return float(librosa.get_duration(y=normalized, sr=sample_rate))


def probe_duration_seconds(input_path: Path) -> float:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "probe.wav"
        duration = preprocess_audio(input_path, tmp_path)
    return duration
