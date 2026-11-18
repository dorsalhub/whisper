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

import math
import pathlib
from typing import Any, Final

import oatschema
import pytest
from dorsal.api.adapters import export_record
from dorsal.file.chunking import merge_chunked_records
from dorsal.testing import model_config_options, run_model

from dorsal_whisper import model as model_module
from dorsal_whisper.model import (
    MAX_SEGMENT_TEXT_LENGTH,
    FasterWhisperTranscriber,
    language_code,
    model_label,
    option_bool,
    option_int,
    segment_score,
)

from .conftest import AUDIO_FILE, CONFIG, FakeInfo, FakeSegment, FakeWhisper, FakeWord

SCHEMA_ID = CONFIG["schema_id"]
SCHEMA_VERSION = CONFIG["schema_version"]
SCHEMA_NAME: Final = "audio-transcription"


def run(**overrides: Any) -> list[Any]:
    options = {**model_config_options(config=CONFIG), **overrides}
    return run_model(
        annotation_model=FasterWhisperTranscriber,
        file_path=str(AUDIO_FILE),
        schema_id=SCHEMA_ID,
        schema_version=SCHEMA_VERSION,
        dependencies=CONFIG.get("dependencies"),
        options=options,
    )


def assert_no_nulls(*, value: Any) -> None:
    if isinstance(value, dict):
        for item in value.values():
            assert_no_nulls(value=item)
    elif isinstance(value, list):
        for item in value:
            assert_no_nulls(value=item)
    else:
        assert value is not None


def valid_records(*, results: list[Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for result in results:
        assert result.error is None, result.error
        assert result.schema_version == SCHEMA_VERSION
        assert result.source.id == "github:dorsalhub/whisper"
        assert result.records
        for record in result.records:
            oatschema.validate(record, SCHEMA_NAME, SCHEMA_VERSION)
            assert_no_nulls(value=record)
            records.append(record)
    return records


def only_record(*, results: list[Any]) -> dict[str, Any]:
    (record,) = valid_records(results=results)
    return record


def test_pin_is_oats_0_8_0() -> None:
    assert SCHEMA_ID == "open/audio-transcription"
    assert SCHEMA_VERSION == "0.8.0"


def test_every_config_option_is_documented() -> None:
    for name, option in CONFIG["options"].items():
        assert isinstance(option, dict), name
        assert {"type", "help"} <= set(option), name


def test_speech_with_known_language(whisper: FakeWhisper) -> None:
    whisper.segments = [
        FakeSegment(text=" Hello there. ", start=0.5, end=2.25, avg_logprob=-0.1),
        FakeSegment(text="General Kenobi.", start=2.5, end=4.0, avg_logprob=-0.3),
    ]
    (result,) = run()
    record = only_record(results=[result])

    assert (
        result.source.variant
        == f"faster-whisper-{model_module.FASTER_WHISPER_VERSION}/base/transcribe"
    )
    assert record == {
        "producer": "faster-whisper-base",
        "text": "Hello there. General Kenobi.",
        "language": "eng",
        "duration": 10.0,
        "score_explanation": "Probability derived from avg_logprob (exp)",
        "segments": [
            {
                "text": "Hello there.",
                "start_time": 0.5,
                "end_time": 2.25,
                "score": round(math.exp(-0.1), 4),
            },
            {
                "text": "General Kenobi.",
                "start_time": 2.5,
                "end_time": 4.0,
                "score": round(math.exp(-0.3), 4),
            },
        ],
        "attributes": {"language_probability": 0.98},
    }


def test_config_options_reach_the_model(whisper: FakeWhisper) -> None:
    only_record(results=run())

    (load,) = whisper.loads
    assert load == {
        "model_size_or_path": "base",
        "device": "auto",
        "compute_type": "default",
        "cpu_threads": 0,
    }
    (call,) = whisper.calls
    assert call["audio"] == str(AUDIO_FILE)
    assert {
        key: value for key, value in call.items() if key not in {"audio", "model"}
    } == {
        "beam_size": 5,
        "vad_filter": True,
    }


def test_vad_parameters_reach_the_model_as_a_dict(whisper: FakeWhisper) -> None:
    only_record(results=run(vad_parameters={"min_silence_duration_ms": 500}))

    (call,) = whisper.calls
    assert call["vad_parameters"] == {"min_silence_duration_ms": 500}


@pytest.mark.parametrize(
    "segments", [[], [FakeSegment(text="   "), FakeSegment(text="\n\t")]]
)
def test_no_speech(whisper: FakeWhisper, segments: list[FakeSegment]) -> None:
    whisper.segments = segments
    record = only_record(results=run())

    assert record["text"] == ""
    assert "segments" not in record
    assert record["language"] == "eng"


@pytest.mark.parametrize("language", [None, "", "xx"])
def test_unknown_language_is_left_out(
    whisper: FakeWhisper, language: str | None
) -> None:
    whisper.info = FakeInfo(language=language)
    record = only_record(results=run())

    assert "language" not in record
    assert record["text"] == "Hello world."


def test_no_speech_and_unknown_language(whisper: FakeWhisper) -> None:
    whisper.segments = []
    whisper.info = FakeInfo(
        language=None, language_probability=float("nan"), duration=0.0
    )
    record = only_record(results=run())

    assert record == {
        "producer": "faster-whisper-base",
        "text": "",
        "duration": 0.0,
        "score_explanation": "Probability derived from avg_logprob (exp)",
    }


def test_whitespace_only_segments_are_dropped(whisper: FakeWhisper) -> None:
    whisper.segments = [
        FakeSegment(text=" One. ", start=0.0, end=1.0),
        FakeSegment(text="  ", start=1.0, end=2.0),
        FakeSegment(text="Two.", start=2.0, end=3.0),
    ]
    record = only_record(results=run())

    assert record["text"] == "One. Two."
    assert [segment["text"] for segment in record["segments"]] == ["One.", "Two."]


@pytest.mark.parametrize("language", ["fr", None])
def test_translate(whisper: FakeWhisper, language: str | None) -> None:
    whisper.info = FakeInfo(language=language)
    (result,) = run(task="translate")
    record = only_record(results=[result])

    assert result.source.variant.endswith("/base/translate")
    assert record["language"] == "eng"
    if language is None:
        assert "source_language" not in record["attributes"]
    else:
        assert record["attributes"]["source_language"] == "fra"
    (call,) = whisper.calls
    assert call["task"] == "translate"


def test_language_option_is_passed_through(whisper: FakeWhisper) -> None:
    whisper.info = FakeInfo(language="de")
    record = only_record(results=run(language="de"))

    assert record["language"] == "deu"
    (call,) = whisper.calls
    assert call["language"] == "de"


def test_long_segment_is_cut_to_the_schema_limit(whisper: FakeWhisper) -> None:
    long_text = "a" * (MAX_SEGMENT_TEXT_LENGTH + 500)
    whisper.segments = [FakeSegment(text=long_text)]
    record = only_record(results=run())

    assert record["segments"][0]["text"] == "a" * MAX_SEGMENT_TEXT_LENGTH
    assert record["text"] == long_text


def test_forced_long_segment_fails_its_pin(whisper: FakeWhisper) -> None:
    whisper.segments = [FakeSegment(text="a" * (MAX_SEGMENT_TEXT_LENGTH + 1))]
    (result,) = run(force=True)

    assert result.error is not None
    assert "maxLength" in result.error
    assert not result.records


def test_longest_text_is_split_by_dorsal(whisper: FakeWhisper) -> None:
    segment_text = "word " * 800
    whisper.segments = [
        FakeSegment(text=segment_text, start=float(index), end=float(index) + 1.0)
        for index in range(200)
    ]
    results = run()
    records = valid_records(results=results)

    assert len(records) > 1
    assert [result.source.execution_part for result in results] == list(
        range(1, len(records) + 1)
    )
    merged = merge_chunked_records(records, SCHEMA_ID)
    assert merged["text"] == " ".join([segment_text.strip()] * 200)
    assert len(merged["text"]) > 524288
    assert len(merged["segments"]) == 200


def test_most_segments_are_split_by_dorsal(whisper: FakeWhisper) -> None:
    count = 100_001
    whisper.segments = [
        FakeSegment(text="Hi.", start=float(index), end=float(index) + 0.5)
        for index in range(count)
    ]
    whisper.info = FakeInfo(duration=float(count))
    records = valid_records(results=run())

    assert [len(record["segments"]) for record in records] == [100_000, 1]


def test_word_timestamps_tighten_segment_times(whisper: FakeWhisper) -> None:
    whisper.segments = [
        FakeSegment(
            text="Tight.",
            start=0.0,
            end=3.0,
            words=[FakeWord(start=0.4, end=0.9), FakeWord(start=1.0, end=1.6)],
        ),
        FakeSegment(text="No words.", start=3.0, end=4.0, words=[]),
    ]
    record = only_record(results=run(word_timestamps=True))

    assert [(s["start_time"], s["end_time"]) for s in record["segments"]] == [
        (0.4, 1.6),
        (3.0, 4.0),
    ]
    (call,) = whisper.calls
    assert call["word_timestamps"] is True


def test_word_times_are_ignored_without_word_timestamps(whisper: FakeWhisper) -> None:
    whisper.segments = [
        FakeSegment(
            text="Loose.", start=0.0, end=3.0, words=[FakeWord(start=0.4, end=0.9)]
        )
    ]
    record = only_record(results=run())

    assert (record["segments"][0]["start_time"], record["segments"][0]["end_time"]) == (
        0.0,
        3.0,
    )


def test_out_of_range_numbers_are_made_valid(whisper: FakeWhisper) -> None:
    whisper.segments = [
        FakeSegment(text="Early.", start=-0.2, end=-0.1, avg_logprob=0.3),
        FakeSegment(text="Backwards.", start=5.0, end=4.0, avg_logprob=-50.0),
        FakeSegment(text="Unscored.", start=6.0, end=7.0, avg_logprob=float("nan")),
    ]
    whisper.info = FakeInfo(duration=-1.0, language_probability=float("nan"))
    record = only_record(results=run())

    assert record["segments"] == [
        {"text": "Early.", "start_time": 0.0, "end_time": 0.0, "score": 1.0},
        {"text": "Backwards.", "start_time": 5.0, "end_time": 5.0, "score": 0.0},
        {"text": "Unscored.", "start_time": 6.0, "end_time": 7.0},
    ]
    assert record["duration"] == 0.0
    assert "attributes" not in record


def test_batched_inference(whisper: FakeWhisper) -> None:
    only_record(results=run(batch_size=8))

    assert len(whisper.batched) == 1
    (call,) = whisper.calls
    assert call["batch_size"] == 8


def test_model_size_option(whisper: FakeWhisper) -> None:
    (result,) = run(model_size="large-v3")
    record = only_record(results=[result])

    assert record["producer"] == "faster-whisper-large-v3"
    assert result.source.variant.endswith("/large-v3/transcribe")
    assert whisper.loads[0]["model_size_or_path"] == "large-v3"


def test_hugging_face_repo_id(whisper: FakeWhisper) -> None:
    (result,) = run(model_size="Systran/faster-whisper-large-v3")
    record = only_record(results=[result])

    assert record["producer"] == "faster-whisper-Systran--faster-whisper-large-v3"
    _, model, task = result.source.variant.split("/")
    assert (model, task) == ("Systran--faster-whisper-large-v3", "transcribe")
    assert whisper.loads[0]["model_size_or_path"] == "Systran/faster-whisper-large-v3"


def test_local_model_path_shows_only_its_folder_name(
    whisper: FakeWhisper, tmp_path: pathlib.Path
) -> None:
    model_dir = tmp_path / "private" / "my-whisper"
    model_dir.mkdir(parents=True)
    (result,) = run(model_size=str(model_dir))
    record = only_record(results=[result])

    assert record["producer"] == "faster-whisper-my-whisper"
    assert result.source.variant.endswith("/my-whisper/transcribe")
    assert "private" not in str(record) + result.source.variant
    assert whisper.loads[0]["model_size_or_path"] == str(model_dir)


def test_model_label() -> None:
    assert model_label(model_size="base") == "base"
    assert (
        model_label(model_size="Systran/faster-whisper-large-v3")
        == "Systran--faster-whisper-large-v3"
    )
    assert model_label(model_size="x" * 300) == "x" * 128
    assert model_label(model_size=pathlib.Path.cwd().anchor) == "custom"


def test_language_code() -> None:
    assert language_code(language="en") == "eng"
    assert language_code(language="jw") == "jav"
    assert language_code(language=None) is None
    assert language_code(language="xx") is None


def test_language_code_rejects_a_non_iso_639_3_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        model_module, "normalize_language_alpha3", lambda language: "EN-GB"
    )
    assert language_code(language="en") is None


def test_segment_score() -> None:
    assert segment_score(avg_logprob=0.0) == 1.0
    assert segment_score(avg_logprob=2.0) == 1.0
    assert segment_score(avg_logprob=-1.0) == round(math.exp(-1.0), 4)
    assert segment_score(avg_logprob=float("-inf")) == 0.0
    assert segment_score(avg_logprob=float("nan")) is None


def test_cpu_uses_int8_by_default(whisper: FakeWhisper) -> None:
    only_record(results=run(device="cpu"))

    assert whisper.loads == [
        {
            "model_size_or_path": "base",
            "device": "cpu",
            "compute_type": "int8",
            "cpu_threads": 0,
        }
    ]


def test_auto_device_falls_back_to_cpu(whisper: FakeWhisper) -> None:
    whisper.failing_devices = {"auto"}
    only_record(results=run(cpu_threads=2))

    assert [
        (load["device"], load["compute_type"], load["cpu_threads"])
        for load in whisper.loads
    ] == [
        ("auto", "default", 2),
        ("cpu", "int8", 2),
    ]


def test_explicit_device_that_fails_is_an_error(whisper: FakeWhisper) -> None:
    whisper.failing_devices = {"cuda"}
    (result,) = run(device="cuda", compute_type="float16")

    assert result.error == "Failed to load model 'base': cannot load on cuda"
    assert not result.records
    assert [load["device"] for load in whisper.loads] == ["cuda"]


def test_model_cache(whisper: FakeWhisper) -> None:
    transcriber = FasterWhisperTranscriber(file_path=str(AUDIO_FILE))

    first = transcriber._load_model("tiny")
    assert FasterWhisperTranscriber._active_model == ("tiny-auto-default-0", first)
    assert transcriber._load_model("tiny") is first
    assert len(whisper.loads) == 1

    second = transcriber._load_model("base")
    assert second is not first
    assert FasterWhisperTranscriber._active_model == ("base-auto-default-0", second)
    assert len(whisper.loads) == 2


def test_runs_reuse_the_loaded_model(whisper: FakeWhisper) -> None:
    only_record(results=run())
    only_record(results=run())

    assert len(whisper.loads) == 1
    assert whisper.calls[0]["model"] is whisper.calls[1]["model"]


def test_cli_strings_are_read_as_their_declared_types(whisper: FakeWhisper) -> None:
    whisper.segments = [FakeSegment(text="a" * (MAX_SEGMENT_TEXT_LENGTH + 1))]
    only_record(
        results=run(
            force="false",
            vad_filter="false",
            beam_size="3",
            cpu_threads="2",
            batch_size="8",
            word_timestamps="False",
        )
    )

    assert whisper.loads[0]["cpu_threads"] == 2
    (call,) = whisper.calls
    assert {
        key: call[key]
        for key in ("beam_size", "vad_filter", "batch_size", "word_timestamps")
    } == {
        "beam_size": 3,
        "vad_filter": False,
        "batch_size": 8,
        "word_timestamps": False,
    }


@pytest.mark.parametrize(
    ("option", "value", "error"),
    [
        ("force", "ture", "Option 'force' must be true or false, not 'ture'."),
        (
            "word_timestamps",
            "maybe",
            "Option 'word_timestamps' must be true or false, not 'maybe'.",
        ),
        ("beam_size", "three", "Option 'beam_size' must be an integer, not 'three'."),
        ("batch_size", "8.5", "Option 'batch_size' must be an integer, not '8.5'."),
    ],
)
def test_bad_option_value_is_an_error(
    whisper: FakeWhisper, option: str, value: str, error: str
) -> None:
    (result,) = run(**{option: value})

    assert result.error == error
    assert not result.records
    assert not whisper.calls


def test_option_bool() -> None:
    assert option_bool(name="x", value=" YES ") is True
    assert option_bool(name="x", value="0") is False
    assert option_bool(name="x", value=True) is True
    assert option_bool(name="x", value=0) is False


def test_option_int() -> None:
    assert option_int(name="x", value=" 4 ") == 4
    assert option_int(name="x", value=4) == 4
    with pytest.raises(ValueError, match="must be an integer"):
        option_int(name="x", value=True)
    with pytest.raises(ValueError, match="must be an integer"):
        option_int(name="x", value=2.5)


def test_mistyped_option_is_an_error_not_a_crash(whisper: FakeWhisper) -> None:
    (result,) = run(beam_sise=3)

    assert (
        result.error
        == "Transcription failed: transcribe() got an unexpected keyword argument 'beam_sise'"
    )
    assert not result.records


def test_transcription_failure_is_an_error(whisper: FakeWhisper) -> None:
    whisper.transcribe_error = RuntimeError("decoder exploded")
    (result,) = run()

    assert result.error == "Transcription failed: decoder exploded"
    assert not result.records


def test_missing_faster_whisper_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_module, "WhisperModel", None)
    (result,) = run()

    assert result.error == "Missing dependency: 'faster-whisper'. Install via pip."
    assert not result.records


def test_srt_export(whisper: FakeWhisper) -> None:
    whisper.segments = [
        FakeSegment(text="First line.", start=1.04, end=8.04),
        FakeSegment(text="Second line.", start=9.04, end=12.04),
    ]
    record = only_record(results=run())

    srt = export_record(record, SCHEMA_ID, "srt", schema_version=SCHEMA_VERSION)

    assert srt.split("\n\n")[:2] == [
        "1\n00:00:01,040 --> 00:00:08,040\nFirst line.",
        "2\n00:00:09,040 --> 00:00:12,040\nSecond line.",
    ]
