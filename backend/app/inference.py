from pathlib import Path
import logging
import os
import time
import warnings

import torch

logger = logging.getLogger(__name__)


class InferenceUnavailableError(RuntimeError):
    pass


class InferenceService:
    def __init__(self) -> None:
        self._tts = None
        self._load_failed = False
        self._load_error: str | None = None
        self._latent_cache: dict[str, dict] = {}

    def warmup(self) -> None:
        try:
            os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
            logger.info(
                "xtts_warmup_env TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=%s",
                os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"),
            )
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

    @property
    def tts(self):
        return self._tts

    def _load_latents(self, latents_path: Path) -> dict:
        key = str(latents_path.resolve())
        if key not in self._latent_cache:
            data = torch.load(key, map_location="cpu", weights_only=False)
            self._latent_cache[key] = data
        return self._latent_cache[key]

    def synthesize(
        self,
        text: str,
        speaker_wav: Path | list[Path],
        language: str,
        output_path: Path,
        *,
        model_path: Path | None = None,
        synthesis_mode: str = "reference",
    ) -> None:
        if self._tts is None:
            reason = self._load_error or "model not initialized"
            raise InferenceUnavailableError(
                f"XTTS model is unavailable. Warmup failed: {reason}"
            )

        started = time.perf_counter()
        logger.info(
            "xtts_inference_start mode=%s speaker_wav=%s language=%s text_chars=%s model_path=%s output_path=%s",
            synthesis_mode,
            speaker_wav,
            language,
            len(text),
            model_path,
            output_path,
        )

        if synthesis_mode == "trained" and model_path and model_path.exists():
            self._synthesize_with_latents(text, language, output_path, model_path)
        else:
            paths = speaker_wav if isinstance(speaker_wav, list) else [speaker_wav]
            ref_arg = str(paths[0]) if len(paths) == 1 else [str(p) for p in paths]
            self._tts.tts_to_file(
                text=text,
                speaker_wav=ref_arg,
                language=language,
                file_path=str(output_path),
            )

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "xtts_inference_done mode=%s output_path=%s elapsed_ms=%s",
            synthesis_mode,
            output_path,
            elapsed_ms,
        )

    def _synthesize_with_latents(
        self, text: str, language: str, output_path: Path, latents_path: Path
    ) -> None:
        data = self._load_latents(latents_path)
        gpt_cond_latent = data["gpt_cond_latent"]
        speaker_embedding = data["speaker_embedding"]

        xtts = self._tts.synthesizer.tts_model
        device = next(xtts.parameters()).device
        gpt_cond_latent = gpt_cond_latent.to(device)
        speaker_embedding = speaker_embedding.to(device)

        out = xtts.inference(
            text=text,
            language=language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            enable_text_splitting=True,
        )
        wav = out["wav"]
        if hasattr(self._tts.synthesizer, "save_wav"):
            self._tts.synthesizer.save_wav(wav, str(output_path))
        else:
            import soundfile as sf

            sf.write(str(output_path), wav, 24000)
