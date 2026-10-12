# Refactor notes

## Changes applied
- Renamed `nova_ui_v4.py` to `nova_app.py`.
- Renamed the large voice/conversation module to `conversation_engine.py` without rewriting its logic.
- Renamed `tts_server_optimized_v3_stream.py` to `speech_service.py`.
- Renamed `resource_monitor.py` to `system_monitor.py`.
- Renamed `speech_text.py` to `speech_utils.py`.
- Renamed launchers to `Start-NOVA.bat` and `Start-Model-Server.bat`.
- Updated direct imports, engine module diagnostics, UI launcher references, XTTS process matching, and project-root resolution.
- Removed `actions/__pycache__` from the release tree.
- Added architecture, installation, configuration, memory, privacy, and refactor notes.

## Intentionally not changed
- The internal logic of the conversation engine, XTTS service, resource monitor, and action handlers, except for startup-time CUDA DLL setup in `nova_app.py`.
- Model and service ports.
- llama.cpp flags and model quantization.
- The existing virtual-environment names and SQLite schema/auto-initialization logic.

## Packaging pass updates
- Added configurable `NOVA_LLAMA_DIR` and `NOVA_FFMPEG_BIN` path overrides via local `nova_paths.bat`; defaults are relative to the project or use FFmpeg on `PATH`.
- Removed fixed Python 3.14 CUDA DLL path assumptions.
- Registered CUDA DLL directories from the project's `tts_env` and explicitly preloaded `cublas64_12.dll` and `cublasLt64_12.dll` before importing the conversation engine. This fixes the observed Whisper runtime error on the maintainer's Windows setup.
- Router logs redact query text by default; full search snippets require explicit `NOVA_DEBUG_SEARCH=1`.
- Updated setup/privacy documentation and included the reviewed installer plus pinned requirements.

The DLL fix is machine-validated for Whisper transcription on the maintainer's setup; clean-machine installer and GPU testing remain necessary before broad distribution.
