# Copyright 2026 Dorsal Hub LTD
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from importlib.metadata import version
import math
from typing import ClassVar, Any

from dorsal import AnnotationModel
from dorsal.common.language import normalize_language_alpha3

try:
    import faster_whisper  # type: ignore[import-untyped]
    from faster_whisper import WhisperModel, BatchedInferencePipeline  # type: ignore[import-untyped]

    FASTER_WHISPER_VERSION = getattr(faster_whisper, "__version__", "unknown")

except ImportError:
    WhisperModel = None
    BatchedInferencePipeline = None
    FASTER_WHISPER_VERSION = "unknown"

MAX_TEXT_LENGTH = 524288

logger = logging.getLogger(__name__)


class FasterWhisperTranscriber(AnnotationModel):
    """
    Transcribes audio/video using faster-whisper (CTranslate2).

    Output Schema: open/audio-transcription (min v0.4.0)
    """

    id = "github:dorsalhub/dorsal-whisper"
    version = version("dorsal-whisper")
    variant = f"faster-whisper-{FASTER_WHISPER_VERSION}"
    default_model_size = "base"
    _active_model: ClassVar[tuple[str, Any] | None] = None

    def _load_model(
        self,
        model_size: str,
        device: str = "auto",
        compute_type: str = "default",
        cpu_threads: int = 0,
    ):
        effective_compute_type = compute_type
        if device == "cpu" and effective_compute_type == "default":
            effective_compute_type = "int8"
            logger.info(
                "Automatically selecting 'int8' compute_type for optimized CPU execution."
            )

        cache_key = f"{model_size}-{device}-{effective_compute_type}-{cpu_threads}"

        if self._active_model and self._active_model[0] == cache_key:
            logger.debug(f"Loading from cache: {cache_key}")
            return self._active_model[1]

        if self._active_model:
            logger.info(
                f"Evicting model '{self._active_model[0]}' from cache to load '{cache_key}'..."
            )
            del FasterWhisperTranscriber._active_model
            FasterWhisperTranscriber._active_model = None

        logger.info(
            f"Loading faster-whisper model '{model_size}' on '{device}' with compute_type '{effective_compute_type}'..."
        )

        try:
            model = WhisperModel(
                model_size_or_path=model_size,
                device=device,
                compute_type=effective_compute_type,
                cpu_threads=cpu_threads,
            )
        except (ValueError, RuntimeError) as e:
            logger.warning(
                f"Failed to load model on {device} with {effective_compute_type}, falling back to CPU (int8): {e}"
            )
            model = WhisperModel(
                model_size_or_path=model_size,
                device="cpu",
                compute_type="int8",
                cpu_threads=cpu_threads,
            )

        logger.info(f"Loaded faster-whisper model '{model_size}' successfully.")

        FasterWhisperTranscriber._active_model = (cache_key, model)
        return model

    def main(
        self,
        model_size: str | None = None,
        device: str = "auto",
        compute_type: str = "default",
        cpu_threads: int = 0,
        beam_size: int = 5,
        vad_filter: bool = True,
        force: bool = False,
        batch_size: int | None = None,
        **kwargs,
    ) -> dict | None:
        """
        Transcribe a media file using faster-whisper.

        Args:
            model_size: The model size to use (e.g., "base", "large-v3").
            device: Device to use ("auto", "cuda", "cpu"). Defaults to "auto".
            compute_type: Quantization type (e.g., "int8", "float16", "default").
            cpu_threads: Number of threads for CPU execution. 0 uses default.
            beam_size: Beam size for decoding. Defaults to 5.
            vad_filter: Whether to apply Voice Activity Detection. Defaults to True.
            force: If True, allows output text to exceed the schema limit.
            batch_size: If provided, uses BatchedInferencePipeline for faster inference.
            **kwargs: Additional arguments passed to model.transcribe.
        Returns:
            A dictionary matching the open/audio-transcription schema.
        """
        if WhisperModel is None:
            self.set_error("Missing dependency: 'faster-whisper'. Install via pip.")
            return None

        target_size = model_size or self.default_model_size

        try:
            model = self._load_model(
                target_size,
                device=device,
                compute_type=compute_type,
                cpu_threads=cpu_threads,
            )
        except Exception as e:
            self.set_error(f"Failed to load model '{target_size}': {e}")
            return None

        try:
            if batch_size is not None:
                logger.info(
                    f"Using BatchedInferencePipeline with batch_size={batch_size}"
                )
                inference_model = BatchedInferencePipeline(model=model)
            else:
                inference_model = model

            logger.debug(
                f"Transcribing {self.name} with {target_size} (beam={beam_size}, vad={vad_filter}, kwargs={kwargs})..."
            )

            transcribe_kwargs = kwargs.copy()
            if batch_size is not None:
                transcribe_kwargs["batch_size"] = batch_size

            segments_generator, info = inference_model.transcribe(
                self.file_path,
                beam_size=beam_size,
                vad_filter=vad_filter,
                **transcribe_kwargs,
            )

            segments = []
            total_duration = round(info.duration, 2)

            for seg in segments_generator:
                segments.append(seg)
                self.update_progress(current=seg.end, total=total_duration)

        except Exception as e:
            self.set_error(f"Transcription failed: {e}")
            return None

        lang_3_letter = normalize_language_alpha3(info.language)

        schema_segments = []
        full_text_parts = []
        use_word_timing = kwargs.get("word_timestamps", False)

        for seg in segments:
            text_clean = seg.text.strip()
            full_text_parts.append(text_clean)

            start_time = float(seg.start)
            end_time = float(seg.end)

            if use_word_timing and hasattr(seg, "words") and seg.words:
                start_time = float(seg.words[0].start)
                end_time = float(seg.words[-1].end)

            schema_segments.append(
                {
                    "text": text_clean,
                    "start_time": round(start_time, 3),
                    "end_time": round(end_time, 3),
                    "score": round(math.exp(float(seg.avg_logprob)), 4),
                }
            )

        full_text = " ".join(full_text_parts)

        if len(full_text) > MAX_TEXT_LENGTH:
            if force:
                logger.warning(
                    f"Transcription length ({len(full_text)}) exceeds schema limit ({MAX_TEXT_LENGTH}), "
                    "but force=True. Schema validation will fail."
                )
            else:
                logger.warning(
                    f"Transcription length ({len(full_text)}) exceeds schema limit ({MAX_TEXT_LENGTH}). "
                    "Truncating text."
                )
                full_text = full_text[:MAX_TEXT_LENGTH]

        return {
            "producer": f"faster-whisper-{target_size}",
            "text": full_text,
            "language": lang_3_letter,
            "duration": info.duration,
            "score_explanation": "Probability derived from avg_logprob (exp)",
            "segments": schema_segments,
            "attributes": {"language_probability": round(info.language_probability, 4)},
        }
