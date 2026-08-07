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
from dorsal.testing import run_model
from dorsal_whisper.model import FasterWhisperTranscriber

TEST_ASSETS = pathlib.Path(__file__).parent / "assets"

root = pathlib.Path(__file__).parent.parent
with open(root / "model_config.toml", "rb") as f:
    config = tomllib.load(f)


def get_parsed_options(config_dict: dict) -> dict:
    """Extracts default values from the TOML options schema, ignoring help-only keys."""
    raw_options = config_dict.get("options", {})
    parsed_options = {}

    for key, value in raw_options.items():
        if isinstance(value, dict):
            if "default" in value:
                parsed_options[key] = value["default"]
        else:
            parsed_options[key] = value

    return parsed_options


BASE_TEST_OPTIONS = get_parsed_options(config)


def test_model_integration():
    """Tests the Whisper model running inside the Dorsal harness."""
    audio_file = TEST_ASSETS / "OSR_uk_000_0020_8k.wav"

    test_options = BASE_TEST_OPTIONS.copy()

    results = run_model(
        annotation_model=FasterWhisperTranscriber,
        file_path=str(audio_file),
        schema_id=config["schema_id"],
        validation_model=config.get("validation_model"),
        dependencies=config.get("dependencies"),
        options=test_options,
    )

    assert isinstance(results, list) and len(results) > 0, (
        "Expected a non-empty list of results"
    )
    result = results[0]

    assert result.error is None, f"Model execution failed: {result.error}"
    assert result.record is not None, "Model returned no data"

    output = result.record

    assert "faster-whisper" in output["producer"]
    assert "text" in output
    assert len(output["text"]) > 0, "Transcription should not be empty"
    assert len(output["segments"]) > 0
    assert "duration" in output
    assert output["duration"] > 0

    first_segment = output["segments"][0]
    assert "start_time" in first_segment
    assert "end_time" in first_segment


def test_model_caching():
    """Tests that the _load_model method properly caches and evicts models to save VRAM."""

    transcriber = FasterWhisperTranscriber(file_path="dummy.wav")

    FasterWhisperTranscriber._active_model = None

    model_1 = transcriber._load_model("tiny", compute_type="default")
    assert FasterWhisperTranscriber._active_model is not None
    assert FasterWhisperTranscriber._active_model[0] == "tiny-auto-default-0"

    model_2 = transcriber._load_model("tiny", compute_type="default")
    assert model_1 is model_2, "Model was not loaded from cache!"

    model_3 = transcriber._load_model("base", compute_type="default")
    assert FasterWhisperTranscriber._active_model[0] == "base-auto-default-0"
    assert model_3 is not model_1, "Cache did not evict the old model!"


def test_word_timestamps():
    """Tests transcription with word-level timestamps enabled to ensure the sharpening logic works."""
    audio_file = TEST_ASSETS / "OSR_uk_000_0020_8k.wav"

    test_options = BASE_TEST_OPTIONS.copy()
    test_options["word_timestamps"] = True
    test_options["model_size"] = "tiny"

    results = run_model(
        annotation_model=FasterWhisperTranscriber,
        file_path=str(audio_file),
        schema_id=config["schema_id"],
        validation_model=config.get("validation_model"),
        dependencies=config.get("dependencies"),
        options=test_options,
    )

    assert isinstance(results, list) and len(results) > 0, (
        "Expected a non-empty list of results"
    )
    result = results[0]

    assert result.error is None, f"Word timestamps run failed: {result.error}"

    output = result.record
    assert len(output["segments"]) > 0

    first_segment = output["segments"][0]
    assert isinstance(first_segment["start_time"], float)
    assert isinstance(first_segment["end_time"], float)
    assert first_segment["end_time"] >= first_segment["start_time"]


def test_batched_inference():
    """Tests that the BatchedInferencePipeline wrapper works without throwing kwarg errors."""
    audio_file = TEST_ASSETS / "OSR_uk_000_0020_8k.wav"

    test_options = BASE_TEST_OPTIONS.copy()
    test_options["batch_size"] = 4
    test_options["model_size"] = "tiny"

    results = run_model(
        annotation_model=FasterWhisperTranscriber,
        file_path=str(audio_file),
        schema_id=config["schema_id"],
        validation_model=config.get("validation_model"),
        dependencies=config.get("dependencies"),
        options=test_options,
    )

    assert isinstance(results, list) and len(results) > 0, (
        "Expected a non-empty list of results"
    )
    result = results[0]

    assert result.error is None, f"Batched inference failed: {result.error}"
    assert "text" in result.record
    assert len(result.record["text"]) > 0
