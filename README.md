# Dorsal Whisper

A Dorsal annotation model wrapper for **[faster-whisper](https://github.com/SYSTRAN/faster-whisper)**.

High-performance speech-to-text transcription using CTranslate2-based Whisper models.

## Features

* **Fast Inference:** Uses `faster-whisper` (CTranslate2) for up to 4x speed improvements over the original OpenAI implementation.
* **Automatic VAD:** Built-in Voice Activity Detection to filter silence and improve accuracy.
* **Schema Compliant:** Outputs standardized JSON matching the `open/audio-transcription` schema.
* **Standard Outputs:** Outputs can be exported to a variety of formats including SubRip Text **(.srt)**, Markdown **(.md)** and others via Dorsal Adapters.

## Compatibility

* **Python:** 3.11, 3.12, 3.13
* **Note:** Python 3.14 is **not yet supported** due to upstream dependencies (`onnxruntime` and `faster-whisper`) that are not yet compatible with 3.14.

## Quick Start

Run the model directly against an audio or video file (downloads and installs if not already installed):

```bash
dorsal run dorsalhub/dorsal-whisper ./audio.wav
```

### Configuration Options

You can pass options to the model using the `--opt` (or `-o`) flag.

Example: use a larger whisper model and translate the output:

```bash
dorsal model run dorsalhub/dorsal-whisper ./audio.wav --opt model_size=large-v3 --opt task=translate
```

*Note: You may need to install NVIDIA libraries (cuBLAS/cuDNN) separately if you intend to run on GPU. See the [faster-whisper documentation](https://github.com/SYSTRAN/faster-whisper) for GPU setup.*

**Supported Core Options:**

* `model_size` (default: `base`)
* `beam_size` (default: `5`)
* `vad_filter` (default: `true`)
* `batch_size` (Wraps the model in a `BatchedInferencePipeline` for much faster processing)
* `compute_type` (default: `default`, can force `int8` or `float16`)
* `**kwargs`: Any additional arguments supported by `faster-whisper`'s `transcribe` method (e.g., `task="translate"`, `language="fr"`, `word_timestamps=true`).

**Passing Complex Arguments (JSON)**

Dorsal's CLI natively supports parsing JSON strings for advanced configuration. This is incredibly useful for tuning `faster-whisper`'s Voice Activity Detection (VAD) to prevent the model from hallucinating or looping on background noise.

Pass dictionary arguments by wrapping valid JSON in single quotes:

```bash
dorsal model run dorsalhub/dorsal-whisper ./video.mkv \
  --opt model_size=large-v2 \
  --opt vad_parameters='{"threshold": 0.8, "min_speech_duration_ms": 250}'
```

### Output Formats & Exporting

By default, the CLI outputs a table with timestamps, and saves a validated JSON record to the current working directory.

You also export to other standard formats right from the CLI:

```bash
# Export directly to SubRip Subtitle format (.srt)
dorsal model run dorsalhub/dorsal-whisper ./video.mkv --export=srt
```

## Output

This model produces a file annotation conforming to the [Open Validation Schemas](http://github.com/dorsalhub/open-validation-schemas) Audio Transcription schema:

* **Schema ID:** `open/audio-transcription` (v0.5.0)
* **Key Fields:**
* `text`: The full concatenated transcript.
* `segments`: An array of timed segments with `start_time`, `end_time`, and `score`.
* `language`: The detected ISO-639-3 language code (e.g., `eng`).
* `duration`: The total duration of the source media in seconds.
* `attributes`: Includes `language_probability`.

## Hardware Acceleration

**Note for Windows Users (GPU):**

If you see an error regarding `cublas64_12.dll`, you can install the required NVIDIA runtime libraries directly into your Python environment:

```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

Alternatively, if you do not have an NVIDIA GPU, force CPU mode:

```bash
dorsal run dorsalhub/whisper path/to/file.mkv --opt device=cpu
```


## Development

### Running Tests

This repository uses `pytest` for integration testing.

```bash
pip install -e .[test]
pytest
```

## License

This project is licensed under the Apache 2.0 License.
