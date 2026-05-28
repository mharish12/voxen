from pathlib import Path
import logging
import os
import time
import warnings


logger = logging.getLogger(__name__)


class InferenceUnavailableError(RuntimeError):
    pass


class InferenceService:
    def __init__(self) -> None:
        self._tts = None
        self._load_failed = False
        self._load_error: str | None = None

    def warmup(self) -> None:
        try:
            # XTTS on newer PyTorch versions can fail because torch.load now defaults
            # to weights_only=True. Coqui XTTS checkpoints need full object loading.
            # This setting should only be used for trusted model sources.
            os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
            logger.info("xtts_warmup_env TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=%s", os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"))
            warnings.filterwarnings(
                "ignore",
                message="pkg_resources is deprecated as an API.*",
                category=UserWarning,
                module="jieba._compat",
            )
            from TTS.api import TTS  # pylint: disable=import-outside-toplevel

            self._tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
            self._load_failed = False
            self._load_error = None
            logger.info("xtts_warmup_success model=xtts_v2")
        except Exception as exc:
            self._load_failed = True
            self._tts = None
            self._load_error = str(exc)
            logger.exception("xtts_warmup_failed model=xtts_v2 error=%s", self._load_error)

    def synthesize(self, text: str, speaker_wav: Path, language: str, output_path: Path) -> None:
        if self._tts is not None:
            started = time.perf_counter()
            logger.info(
                "xtts_inference_start speaker_wav=%s language=%s text_chars=%s output_path=%s",
                speaker_wav,
                language,
                len(text),
                output_path,
            )
            self._tts.tts_to_file(text=text, speaker_wav=str(speaker_wav), language=language, file_path=str(output_path))
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            logger.info("xtts_inference_done output_path=%s elapsed_ms=%s", output_path, elapsed_ms)
            return

        reason = self._load_error or "model not initialized"
        raise InferenceUnavailableError(
            f"XTTS model is unavailable. Warmup failed: {reason}"
        )
