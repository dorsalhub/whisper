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

from typing import Any

import oatschema
import pytest
from dorsal.testing import model_config_options, run_model

from dorsal_whisper.model import FasterWhisperTranscriber

from .conftest import AUDIO_FILE, CONFIG

pytestmark = pytest.mark.weights


def run(**overrides: Any) -> dict[str, Any]:
    (result,) = run_model(
        annotation_model=FasterWhisperTranscriber,
        file_path=str(AUDIO_FILE),
        schema_id=CONFIG["schema_id"],
        schema_version=CONFIG["schema_version"],
        dependencies=CONFIG.get("dependencies"),
        options={
            **model_config_options(config=CONFIG),
            "model_size": "tiny",
            **overrides,
        },
    )
    assert result.error is None, result.error
    assert result.schema_version == CONFIG["schema_version"]
    assert result.records
    (record,) = result.records
    oatschema.validate(record, "audio-transcription", CONFIG["schema_version"])
    return record


def test_transcribe() -> None:
    record = run()

    assert record["producer"] == "faster-whisper-tiny"
    assert record["language"] == "eng"
    assert "birch canoe" in record["text"]
    assert record["duration"] > 0
    assert record["segments"]


def test_translate() -> None:
    record = run(task="translate")

    assert record["language"] == "eng"
    assert record["attributes"]["source_language"] == "eng"


def test_word_timestamps() -> None:
    record = run(word_timestamps=True)

    for segment in record["segments"]:
        assert segment["end_time"] >= segment["start_time"]


def test_batched_inference() -> None:
    record = run(batch_size=4)

    assert record["text"]


def test_vad_parameters() -> None:
    record = run(vad_parameters={"min_silence_duration_ms": 500})

    assert record["segments"]
