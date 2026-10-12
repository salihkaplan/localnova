# Package audit report — NOVA Windows setup candidate

## Scope

This report describes the package structure and static checks performed while preparing this archive. The installer has not been executed on a clean Windows host in this packaging session.

## Installer behavior

- Adds `Test-NOVA-DryRun.bat` and `scripts/Test-NOVA-DryRun.ps1` for read-only inventory and a hypothetical clean-machine plan. The dry run does not mutate installation files; GitHub release metadata is queried with read-only HTTP requests unless skipped.

- Reuses a detected `llama-server.exe` and does not overwrite existing `C:\llama-cuda`.
- Downloads official `ggml-org/llama.cpp` CUDA 12.4 or 12.6 Windows x64 release assets only when no server is detected.
- Starts Gemma 4 E4B Q4_K_M, validates `/v1/models`, and sends a short chat completion before installing Python environments.
- Reuses standard Hugging Face model cache; no forced cache relocation.
- Reuses complete Python virtual environments and does not delete partial environments.
- Performs PyTorch CUDA, Whisper DLL/model load, and XTTS import smoke tests.

## Known limits

- NVIDIA Windows is the supported target for this installer. AMD and CPU-only modes are not enabled.
- The exact release asset name can change. If GitHub does not publish a compatible CUDA 12.4/12.6 asset, setup stops and logs the issue rather than choosing CUDA 13.
- Clean-machine installation, audio device behavior, real Whisper transcription, and complete microphone -> Gemma -> XTTS flow require validation by the user.
- Model and dependency downloads require network access and can take time.

## Validation performed during packaging

- Existing Python source files retained from the supplied archive.
- Installer, read-only dry-run entry point, and documentation files created/updated.
- Python smoke-test scripts parsed with `ast` in the packaging environment.
- ZIP content/layout checks performed after packaging.
- Dry-run PowerShell script reviewed statically; it was not executed under Windows PowerShell in this packaging environment.
