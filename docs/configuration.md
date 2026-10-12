# Configuration reference

## Language model
- Model: `unsloth/gemma-4-E4B-it-GGUF:Q4_K_M`
- Local API: `http://127.0.0.1:8080/v1/chat/completions`
- Context: `8192`; parallel slots: `1`; GPU layers: `99`
- Flags: `--ctx-checkpoints 1 --jinja --reasoning off`
- Server path discovery: configured `NOVA_LLAMA_DIR`, project-local `llama-cuda`, `C:\llama-cuda`, then `%USERPROFILE%\llama-cuda`.
- Cache: standard Hugging Face cache used by llama.cpp. Installer does not override `LLAMA_CACHE` or `HF_HOME`.

## Speech recognition
- Whisper model: `small`; device: `cuda`; compute type: `float16`; language: `tr`
- Input: 16 kHz mono
- DLL roots are resolved relative to the NOVA `tts_env`, not a hard-coded Windows Python installation.

## Speech synthesis
- XTTS model: `tts_models/multilingual/multi-dataset/xtts_v2`
- Default speaker: `Viktor Eka`; speed: `1.15`
- Local service: `127.0.0.1:5000`
- FFmpeg search order: `NOVA_FFMPEG_BIN`, project-local `ffmpeg/bin`, then existing `PATH`.

## Local overrides
Copy `nova_paths.bat.example` to `nova_paths.bat` only to override paths manually. Example:

```bat
set "NOVA_LLAMA_DIR=D:\AI\llama-cuda"
set "NOVA_FFMPEG_BIN=D:\Tools\ffmpeg\bin"
```

The installer generates `nova_paths.bat` when absent and preserves an existing file. This file is ignored by Git and should remain machine-local.

## Optional external search
- `BRAVE_API_KEY` is read from the process environment. It is not a path and is not bundled.
- `NOVA_DEBUG_SEARCH=1` enables verbose search debugging. Review/redact logs before sharing.
