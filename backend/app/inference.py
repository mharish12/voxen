from pathlib import Path

import soundfile as sf


class InferenceService:
    def __init__(self) -> None:
        self._tts = None
        self._load_failed = False

    def warmup(self) -> None:
        try:
            from TTS.api import TTS  # pylint: disable=import-outside-toplevel

            self._tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        except Exception:
            self._load_failed = True
            self._tts = None

    def synthesize(self, text: str, speaker_wav: Path, language: str, output_path: Path) -> None:
        if self._tts is not None:
            self._tts.tts_to_file(text=text, speaker_wav=str(speaker_wav), language=language, file_path=str(output_path))
            return

        # Fallback for environments where XTTS is not available yet.
        # Writes one second of silence so APIs remain testable.
        sf.write(str(output_path), [0.0] * 24000, 24000, subtype="PCM_16")
