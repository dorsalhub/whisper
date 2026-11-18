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
import math
import os
import re
from collections.abc import Iterable
from importlib.metadata import version
from pathlib import Path
from typing import Any, ClassVar

from dorsal import AnnotationModel
from dorsal.common.language import normalize_language_alpha3

try:
    import faster_whisper  # type: ignore[import-untyped]
    from faster_whisper import BatchedInferencePipeline, WhisperModel  # type: ignore[import-untyped]

    FASTER_WHISPER_VERSION = getattr(faster_whisper, "__version__", "unknown")

except ImportError:  # pragma: no cover
    WhisperModel = None
    BatchedInferencePipeline = None
    FASTER_WHISPER_VERSION = "unknown"

MAX_SEGMENT_TEXT_LENGTH = 4096
MAX_LABEL_LENGTH = 128
TRANSLATION_LANGUAGE = "eng"
DEFAULT_TASK = "transcribe"

_LANGUAGE_PATTERN = re.compile(r"^[a-z]{3}$")
_TRUE_VALUES = frozenset({"true", "yes", "1", "t", "y"})
_FALSE_VALUES = frozenset({"false", "no", "0", "f", "n"})

logger = logging.getLogger(__name__)


def model_label(*, model_size: str) -> str:
    """Return the name used for `model_size` in records.

    A local model path gives its folder name, and a Hugging Face repo id has each `/` written as `--`.
    """
    label = (
        Path(model_size).name
        if os.path.isdir(model_size)
        else model_size.replace("/", "--")
    )
    return label[:MAX_LABEL_LENGTH] or "custom"


def option_bool(*, name: str, value: Any) -> bool:
    """Return a boolean option's value, reading a string such as "false" or "1" as the CLI writes it."""
    if not isinstance(value, str):
        return bool(value)
    text = value.strip().lower()
    if text in _TRUE_VALUES:
        return True
    if text in _FALSE_VALUES:
        return False
    raise ValueError(f"Option '{name}' must be true or false, not {value!r}.")


def option_int(*, name: str, value: Any) -> int:
    """Return an integer option's value, reading a string such as "3" as the CLI writes it."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            pass
    raise ValueError(f"Option '{name}' must be an integer, not {value!r}.")


def language_code(*, language: str | None) -> str | None:
    """Return the ISO 639-3 code for a faster-whisper language, or None when it is unknown."""
    code = normalize_language_alpha3(language)
    if code is None or not _LANGUAGE_PATTERN.match(code):
        return None
    return code


def segment_score(*, avg_logprob: float) -> float | None:
    """Return a 0-1 confidence from a segment's average log probability, or None when it is not a number."""
    value = float(avg_logprob)
    if math.isnan(value):
        return None
    return round(min(1.0, max(0.0, math.exp(min(value, 0.0)))), 4)


def build_segments(
    *, segments: Iterable[Any], use_word_timing: bool, force: bool
) -> tuple[list[dict], list[str]]:
    """Convert faster-whisper segments to schema segments, returning them with the text of every non-empty segment."""
    schema_segments: list[dict] = []
    texts: list[str] = []

    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        texts.append(text)

        start_time = float(seg.start)
        end_time = float(seg.end)
        words = getattr(seg, "words", None)
        if use_word_timing and words:
            start_time = float(words[0].start)
            end_time = float(words[-1].end)
        start_time = max(0.0, start_time)
        end_time = max(start_time, end_time)

        if len(text) > MAX_SEGMENT_TEXT_LENGTH and not force:
            logger.warning(
                f"Segment text length ({len(text)}) exceeds schema limit ({MAX_SEGMENT_TEXT_LENGTH}). Truncating segment."
            )
            text = text[:MAX_SEGMENT_TEXT_LENGTH]

        entry: dict[str, Any] = {
            "text": text,
            "start_time": round(start_time, 3),
            "end_time": round(end_time, 3),
        }
        score = segment_score(avg_logprob=seg.avg_logprob)
        if score is not None:
            entry["score"] = score
        schema_segments.append(entry)

    return schema_segments, texts


class FasterWhisperTranscriber(AnnotationModel):
    """
    Transcribes audio/video using faster-whisper (CTranslate2).

    Output Schema: open/audio-transcription (0.8.0)
    """

    id = "github:dorsalhub/whisper"
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
            if device != "auto":
                raise
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
            model_size: The model size, Hugging Face repo id or local model path (e.g., "base", "large-v3").
            device: Device to use ("auto", "cuda", "cpu"). Only "auto" falls back to CPU if loading fails.
            compute_type: Quantization type (e.g., "int8", "float16", "default").
            cpu_threads: Number of threads for CPU execution. 0 uses default.
            beam_size: Beam size for decoding. Defaults to 5.
            vad_filter: Whether to apply Voice Activity Detection. Defaults to True.
            force: If True, a segment's text is not cut to the schema's 4096-character limit, so a longer segment fails
                validation and the run gives an error and no record. Over-long transcripts and segment lists are never cut:
                Dorsal splits them into several records.
            batch_size: If provided, uses BatchedInferencePipeline for faster inference.
            **kwargs: Additional arguments passed to model.transcribe (e.g., task, language, word_timestamps).
        Returns:
            A dictionary matching the open/audio-transcription schema.
        """
        if WhisperModel is None:
            self.set_error("Missing dependency: 'faster-whisper'. Install via pip.")
            return None

        try:
            force = option_bool(name="force", value=force)
            vad_filter = option_bool(name="vad_filter", value=vad_filter)
            beam_size = option_int(name="beam_size", value=beam_size)
            cpu_threads = option_int(name="cpu_threads", value=cpu_threads)
            if batch_size is not None:
                batch_size = option_int(name="batch_size", value=batch_size)
            if "word_timestamps" in kwargs:
                kwargs["word_timestamps"] = option_bool(
                    name="word_timestamps", value=kwargs["word_timestamps"]
                )
        except ValueError as e:
            self.set_error(str(e))
            return None

        target_size = model_size or self.default_model_size
        label = model_label(model_size=target_size)
        task = str(kwargs.get("task") or DEFAULT_TASK)
        self.variant = f"faster-whisper-{FASTER_WHISPER_VERSION}/{label}/{task[:64]}"

        try:
            model = self._load_model(
                target_size,
                device=device,
                compute_type=compute_type,
                cpu_threads=cpu_threads,
            )
        except Exception as e:  # noqa: BLE001
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

        except Exception as e:  # noqa: BLE001
            self.set_error(f"Transcription failed: {e}")
            return None

        schema_segments, texts = build_segments(
            segments=segments,
            use_word_timing=bool(kwargs.get("word_timestamps", False)),
            force=force,
        )

        full_text = " ".join(texts)

        attributes: dict[str, Any] = {}
        language_probability = float(info.language_probability)
        if not math.isnan(language_probability):
            attributes["language_probability"] = round(language_probability, 4)

        detected_language = language_code(language=info.language)
        record_language = detected_language
        if task == "translate":
            record_language = TRANSLATION_LANGUAGE
            if detected_language is not None:
                attributes["source_language"] = detected_language

        record: dict[str, Any] = {
            "producer": f"faster-whisper-{label}",
            "text": full_text,
        }
        if record_language is not None:
            record["language"] = record_language
        record["duration"] = max(0.0, float(info.duration))
        record["score_explanation"] = "Probability derived from avg_logprob (exp)"
        if schema_segments:
            record["segments"] = schema_segments
        if attributes:
            record["attributes"] = attributes
        return record
