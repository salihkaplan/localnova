# Configuration

Machine-specific paths can be configured by copying `nova_paths.bat.example` to `nova_paths.bat`. The optional variables are `NOVA_LLAMA_DIR` (folder containing `llama-server.exe`) and `NOVA_FFMPEG_BIN` (FFmpeg `bin` folder). Relative paths are resolved from the project folder. Do not store API keys in this file.

The Brave API key is read from the process environment variable `BRAVE_API_KEY`. Verbose search logs are opt-in via `NOVA_DEBUG_SEARCH=1`; keep this disabled unless needed for troubleshooting.
