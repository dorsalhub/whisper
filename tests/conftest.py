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

import pathlib
import tomllib
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import pytest

from dorsal_whisper import model as model_module
from dorsal_whisper.model import FasterWhisperTranscriber

ROOT = pathlib.Path(__file__).parent.parent
TEST_ASSETS = pathlib.Path(__file__).parent / "assets"
AUDIO_FILE = TEST_ASSETS / "OSR_uk_000_0020_8k.wav"

with open(ROOT / "model_config.toml", "rb") as f:
    CONFIG = tomllib.load(f)

TRANSCRIBE_KEYWORDS = frozenset(
    {
        "beam_size",
        "vad_filter",
        "vad_parameters",
        "language",
        "task",
        "word_timestamps",
        "batch_size",
        "initial_prompt",
        "temperature",
    }
)


@dataclass
class FakeWord:
    start: float
    end: float
    word: str = "word"


@dataclass
class FakeSegment:
    text: str
    start: float = 0.0
    end: float = 1.0
    avg_logprob: float = -0.2
    words: list[FakeWord] | None = None


@dataclass
class FakeInfo:
    language: str | None = "en"
    language_probability: float = 0.98
    duration: float = 10.0


@dataclass
class FakeWhisper:
    """Stands in for faster-whisper's `WhisperModel` and `BatchedInferencePipeline`."""

    segments: list[FakeSegment] = field(
        default_factory=lambda: [FakeSegment(text=" Hello world. ")]
    )
    info: FakeInfo = field(default_factory=FakeInfo)
    failing_devices: set[str] = field(default_factory=set)
    transcribe_error: Exception | None = None
    loads: list[dict[str, Any]] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)
    batched: list[Any] = field(default_factory=list)

    def model_class(self) -> type:
        fake = self

        class WhisperModel:
            def __init__(
                self,
                *,
                model_size_or_path: str,
                device: str,
                compute_type: str,
                cpu_threads: int,
            ):
                load = {
                    "model_size_or_path": model_size_or_path,
                    "device": device,
                    "compute_type": compute_type,
                    "cpu_threads": cpu_threads,
                }
                fake.loads.append(load)
                if device in fake.failing_devices:
                    raise RuntimeError(f"cannot load on {device}")
                self.load = load

            def transcribe(
                self, audio: str, **kwargs: Any
            ) -> tuple[Iterator[FakeSegment], FakeInfo]:
                unknown = set(kwargs) - TRANSCRIBE_KEYWORDS
                if unknown:
                    raise TypeError(
                        f"transcribe() got an unexpected keyword argument '{min(unknown)}'"
                    )
                if fake.transcribe_error is not None:
                    raise fake.transcribe_error
                fake.calls.append({"audio": audio, "model": self, **kwargs})
                return iter(fake.segments), fake.info

        return WhisperModel

    def pipeline_class(self) -> type:
        fake = self

        class BatchedInferencePipeline:
            def __init__(self, *, model: Any):
                self.model = model
                fake.batched.append(self)

            def transcribe(
                self, audio: str, **kwargs: Any
            ) -> tuple[Iterator[FakeSegment], FakeInfo]:
                return self.model.transcribe(audio, **kwargs)

        return BatchedInferencePipeline


@pytest.fixture
def whisper(monkeypatch: pytest.MonkeyPatch) -> FakeWhisper:
    fake = FakeWhisper()
    monkeypatch.setattr(model_module, "WhisperModel", fake.model_class())
    monkeypatch.setattr(model_module, "BatchedInferencePipeline", fake.pipeline_class())
    monkeypatch.setattr(FasterWhisperTranscriber, "_active_model", None)
    return fake
