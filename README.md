# Dorsal Whisper

<p>
  <a href="https://github.com/dorsalhub/whisper/actions/workflows/ci.yml">
    <img src="https://github.com/dorsalhub/whisper/actions/workflows/ci.yml/badge.svg" alt="Tests">
  </a>
  <a href="https://codecov.io/gh/dorsalhub/whisper">
    <img src="https://codecov.io/gh/dorsalhub/whisper/graph/badge.svg" alt="codecov">
  </a>
  <img src="https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-0ea5e9" alt="Python versions">
  <a href="https://opensource.org/licenses/Apache-2.0">
    <img src="https://img.shields.io/badge/license-Apache_2.0-0ea5e9" alt="License">
  </a>
  <a href="https://mypy-lang.org/">
    <img src="https://img.shields.io/badge/mypy-checked-0ea5e9" alt="Mypy checked">
  </a>
</p>

A [Dorsal](https://github.com/dorsalhub/dorsal) annotation model for speech-to-text, built on **[faster-whisper](https://github.com/SYSTRAN/faster-whisper)** (CTranslate2).

## Features

* **Fast inference:** `faster-whisper` runs Whisper models up to 4x faster than the original OpenAI implementation.
* **Voice activity detection:** Silero VAD filters out silence before transcription.
* **Transcribe or translate:** transcribe in the spoken language, or translate the speech to English.
* **Schema compliant:** every record validates against the [OATS](https://oatschema.org) `audio-transcription` schema (0.8.0).
* **Standard outputs:** export records to SubRip **(.srt)**, WebVTT **(.vtt)**, Markdown **(.md)** and other formats from the Dorsal CLI.

## Compatibility

* **Python:** 3.11, 3.12, 3.13, 3.14
* **Dorsal:** 0.13.3 or later, below 0.14
* **Python 3.15** isn't supported yet: `faster-whisper` needs `onnxruntime`, which has no Python 3.15 wheels yet.

## Quick Start

Run the model on an audio or video file. Dorsal installs it from DorsalHub first if it isn't installed:

```bash
dorsal run dorsalhub/whisper ./audio.wav
```

To install it without running it:

```bash
dorsal model install dorsalhub/whisper
```

### Options

Pass options with `--opt` (or `-o`). For example, use a larger model and translate the speech to English:

```bash
dorsal run dorsalhub/whisper ./audio.wav --opt model_size=large-v3 --opt task=translate
```

| Option | Default | Description |
| --- | --- | --- |
| `model_size` | `base` | Model size, Hugging Face repo id or local model path (e.g. `base`, `large-v3`, `turbo`). |
| `device` | `auto` | `auto`, `cuda` or `cpu`. Only `auto` falls back to CPU if the model can't load. |
| `compute_type` | `default` | Quantization type (e.g. `int8`, `float16`). With `device=cpu`, `default` means `int8`. |
| `cpu_threads` | `0` | Threads for CPU execution. `0` uses the default. |
| `beam_size` | `5` | Beam size for decoding. |
| `vad_filter` | `true` | Apply Silero VAD to filter out silence. |
| `force` | `false` | Don't cut a segment's text to the schema's 4096-character limit. A longer segment then fails validation, and the run gives an error and no record. |
| `language` | detected | Spoken language (e.g. `en`, `fr`). |
| `task` | `transcribe` | `transcribe`, or `translate` to English. |
| `batch_size` | unset | Use faster-whisper's `BatchedInferencePipeline` with this batch size (e.g. `8` or `16`). |
| `word_timestamps` | unset | If `true`, word-level timestamps tighten each segment's start and end times. |
| `vad_parameters` | unset | VAD settings as a JSON object (e.g. `'{"min_silence_duration_ms": 500}'`). |

Any other option is passed to faster-whisper's `transcribe` method. See `dorsal model info dorsalhub/whisper` for the full list.

To run on a GPU, you may need to install the NVIDIA libraries (cuBLAS and cuDNN) separately. See the [faster-whisper documentation](https://github.com/SYSTRAN/faster-whisper) for GPU setup.

### JSON options

Pass a dictionary option as JSON in single quotes. Tuning the VAD this way can stop the model from hallucinating or looping on background noise:

```bash
dorsal run dorsalhub/whisper ./video.mkv \
  --opt model_size=large-v2 \
  --opt vad_parameters='{"threshold": 0.8, "min_speech_duration_ms": 250}'
```

### Exporting

By default, the CLI shows the transcription in a table and saves the validated JSON record to the current directory. To export to another format:

```bash
dorsal run dorsalhub/whisper ./video.mkv --export=srt
```

## Output

Each record follows the OATS `audio-transcription` schema, version **0.8.0**:

* `text`: the full transcript.
* `segments`: timed segments, each with `text`, `start_time`, `end_time` and a 0–1 `score`. Segments with no text are left out, and so is `segments` when there's no speech.
* `language`: the ISO 639-3 code of the transcript (e.g. `eng`). It's left out when the language isn't known, and it's `eng` when translating.
* `duration`: the length of the media in seconds.
* `producer`: `faster-whisper-<model>`, such as `faster-whisper-base`, with the model written as in the variant below.
* `attributes`: `language_probability`, and `source_language` (the detected spoken language) when translating.

The annotation's source is the model id `github:dorsalhub/whisper`, with a variant naming the faster-whisper version, model and task, such as `faster-whisper-1.2.1/base/transcribe`. For a local model path, only the folder name appears. A Hugging Face repo id has each `/` written as `--`, such as `Systran--faster-whisper-large-v3`.

A transcript longer than one record allows is split by Dorsal into several records. Pass `--merge` to join them on export.

## Hardware Acceleration

**Windows (GPU):** if you see an error about `cublas64_12.dll`, install the NVIDIA runtime libraries into your Python environment:

```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

If you don't have an NVIDIA GPU, run on the CPU:

```bash
dorsal run dorsalhub/whisper path/to/file.mkv --opt device=cpu
```

## Development

```bash
uv sync --group dev
uv run pytest
```

The default tests use a stub in place of faster-whisper, so they don't download any weights. To run the tests that download real Whisper weights from Hugging Face:

```bash
uv run pytest -m weights
```

`scripts/check.sh` formats, lints and type-checks the code, then runs the tests on every supported Python version.

## License

This project is licensed under the Apache 2.0 License.
