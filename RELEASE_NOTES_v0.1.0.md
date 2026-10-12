# NOVA v0.1.0 — Initial public preview

This is the first planned public release tag for the NOVA Windows setup package.

## Included

- Windows installer and path-discovery helpers.
- Cache-aware Gemma 4 E4B Q4_K_M model-server startup and API smoke test.
- Separate pinned Python environments for the UI and speech/TTS components.
- PyTorch CUDA and Whisper DLL/model-load checks.
- Read-only dry-run/preflight diagnostic.
- Setup, configuration, troubleshooting, privacy, architecture, and license documentation.

## Known limitations

- The currently verified language and prompt/tool-routing configuration is Turkish.
  Other languages may require changes to system prompts, function-calling prompts,
  tool routing, and speech-language settings.
- The first tested installation target is Windows with an NVIDIA GPU. AMD and
  CPU-only installation are not claimed as supported by this installer.
- Web search and some tool-routing behaviors may be experimental and depend on
  the configured search/action implementation.
- XTTS-v2 model weights are subject to the Coqui Public Model License, which
  permits non-commercial use only. See `MODEL_LICENSES.md`.
- Performance depends on hardware, driver, context length, quantization, Flash
  Attention, and model offload. The benchmark values in README are specific test
  results, not guarantees.

## Tagging

Create the Git tag from the reviewed release commit:

```bash
git tag -a v0.1.0 -m "NOVA v0.1.0 initial public preview"
git push origin v0.1.0
```

Then create a GitHub Release for `v0.1.0` and attach the tested installer/package
archive if you intend to distribute one. Do not tag or publish until the README,
licenses, screenshots, and release artifacts have been reviewed.


### Documentation updates
- Added genuine screenshots for the main interface and live system-resource dashboard.
- Marked Wi-Fi and Bluetooth control paths as untested/experimental.
- Clarified that displayed resource readings are capture-time values, not performance guarantees.


### Final pre-release review
- Project name expanded to **NOVA — Neural Orchestrator for Voice Assistance**.
- Laptop benchmark corrected to **16 tok/s**.
- Test status clarified: primary system verified; additional systems remain under testing.
- Added a memory/privacy note and flagged backup/restore/deletion workflows for validation before relying on them.
