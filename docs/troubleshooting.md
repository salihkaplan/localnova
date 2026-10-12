# Troubleshooting

## Install fails before Python packages

- Confirm `py -3.11 --version` works and `nvidia-smi` reports the expected NVIDIA GPU and driver.
- Review `logs/install.log` for the exact failed stage.
- If no existing server was found, the installer queries official llama.cpp GitHub releases and selects a CUDA 12.4 or 12.6 Windows x64 package. If no compatible asset is published, it stops rather than guessing a different CUDA major version.

## Gemma API does not become ready

- Review `logs/gemma-install-smoke.log`.
- Check port 8080, free disk space, network access to Hugging Face, and GPU memory.
- A cache folder can exist while its snapshot is incomplete. The model server's successful `/v1/models` and chat-completion request are the validation steps.
- The installer will not kill a pre-existing process on port 8080. If an unrelated app occupies the port, stop it manually or configure the conflict before retrying.

## Whisper: `cublas64_12.dll` or `cublasLt64_12.dll`

1. Review `logs/whisper-smoke.log`.
2. Confirm the TTS environment contains `torch/lib/cublas64_12.dll` and `torch/lib/cublasLt64_12.dll`.
3. Confirm the installer log shows PyTorch CUDA initialization and GPU matrix multiplication succeeded.
4. Confirm DLL directories are registered before `faster_whisper` / CTranslate2 initializes.
5. Do not download loose DLL files from third-party DLL websites or copy them into `C:\Windows\System32`.

## Existing paths or environments

- `nova_paths.bat` is preserved if it already exists. Check `NOVA_LLAMA_DIR` if the wrong server is found.
- Existing complete Python environments are reused. A partial folder without `Scripts/python.exe` is not deleted automatically; inspect or rename it yourself before rerunning.
- Existing `C:\llama-cuda` is never overwritten. If it is incomplete, installer uses a separate project-local folder for a downloaded server.

## Share diagnostics safely

`logs/install.log`, `logs/gemma-install-smoke.log`, and `logs/whisper-smoke.log` can contain local file paths and system details. Review them and redact private information before sharing. Never send API keys or credentials.
