# Windows installation and validation

This is an automated setup package, not an offline bundle. It downloads the official llama.cpp Windows CUDA build if needed, lets llama.cpp fetch Gemma through its Hugging Face integration when absent, and installs Python dependencies.

## Prerequisites

- Windows 10/11 x64.
- Python 3.11 x64 with `py.exe` launcher.
- NVIDIA GPU and a working NVIDIA driver visible to `nvidia-smi`.
- Internet access to GitHub Releases, PyPI, PyTorch's wheel index and Hugging Face.
- Enough disk space for Python environments, CUDA wheels, Gemma GGUF, Whisper `small` and XTTS model data.

## Read-only dry run

Run `Test-NOVA-DryRun.bat` before installing. It inspects the current system and prints the actions a normal install would take, including a hypothetical clean-machine plan. It does not install or download anything, create/delete environments, start/stop the model server, or write a report. It can make read-only requests to GitHub release metadata to validate the current release asset naming pattern. DLL checks in dry-run mode are filesystem-only; they do not load the DLLs.

## Installation sequence

1. Run `Install-NOVA.bat`.
2. Installer checks existing `llama-server.exe` locations and reuses one if found.
3. Otherwise, it downloads a compatible CUDA 12.4 Windows x64 llama.cpp build, with CUDA 12.6 as fallback. It does not overwrite `C:\llama-cuda` or another existing server directory.
4. Installer starts `unsloth/gemma-4-E4B-it-GGUF:Q4_K_M`, waits for the local model API, and submits a short completion request. Existing standard Hugging Face cache files are reused.
5. Only after Gemma passes, the installer creates missing `jarvis_ui_env` and `tts_env` environments and installs their pinned dependencies.
6. It checks PyTorch CUDA, preloads Whisper's cuBLAS DLLs and loads the Whisper `small` model, then checks XTTS imports.
7. Run `Start-NOVA.bat` and test the full audio loop manually.

The model server is required to download/test the model; this is why llama.cpp is installed/resolved before the Python audio stack. Ollama is not installed or required.

## Automatic path discovery

Path order: explicit environment/configuration -> project-local `llama-cuda` -> `C:\llama-cuda` -> `%USERPROFILE%\llama-cuda`. If none is found, the official server build is installed project-locally. A `nova_paths.bat` file is generated only when one does not already exist. Existing user configuration is preserved.

The Gemma model uses llama.cpp's default Hugging Face cache. The installer does not force `LLAMA_CACHE` or `HF_HOME`, which would bypass an existing cache. Presence of a cache directory does not prove the exact model snapshot is complete; server startup and an API completion are the authoritative check.

## CUDA DLL behavior

`nova_app.py` adds DLL directories from `tts_env\Lib\site-packages\torch\lib` and the relevant NVIDIA runtime/cuBLAS/cuDNN directories, then preloads `cublas64_12.dll` and `cublasLt64_12.dll`. The installer repeats this behavior in `scripts/smoke-test-whisper.py` and logs each successful directory/DLL load.

Do not fetch individual DLLs from unofficial websites or copy them to Windows system folders. Repair the relevant virtual environment from trusted package sources if DLLs are missing.

## Driver notes

The pinned TTS environment uses PyTorch 2.8.0 + CUDA 12.6 wheels. CUDA 12.x minor-version compatibility has a documented Windows driver floor, but driver compatibility is not a guarantee of every feature working. The installer checks actual CUDA initialization and a GPU matrix operation. If that fails, update to a current NVIDIA WHQL driver supported by the GPU, then rerun the installer/tests.

## Diagnostics

- `logs\install.log`
- `logs\gemma-install-smoke.log`
- `logs\whisper-smoke.log`

See `docs/configuration.md` and `docs/troubleshooting.md` for path behavior and recovery. Clean-machine installation and full microphone-to-speech testing must be confirmed by the user; this package has not been executed on a clean Windows host during packaging.
