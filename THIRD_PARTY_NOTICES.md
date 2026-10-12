# Third-Party Notices

NOVA's own source code is licensed under the MIT License in [`LICENSE`](LICENSE).
That license does **not** replace the licenses of third-party libraries, tools,
model weights, or assets downloaded by the installer.

This file is a practical notice for the principal components used by the current
NOVA Windows setup. It is **not a complete generated Software Bill of Materials
(SBOM)** and should not be treated as a substitute for each component's license.
The pinned requirement files include transitive dependencies; their licenses must
be checked individually before redistributing a bundled copy of those packages.

## Principal software components

| Component | Role in NOVA | Upstream license / terms | Upstream reference |
|---|---|---|---|
| `llama.cpp` / `llama-server` | Local Gemma inference server | MIT | https://github.com/ggml-org/llama.cpp/blob/master/LICENSE |
| `faster-whisper` | Speech recognition integration | MIT | https://github.com/SYSTRAN/faster-whisper/blob/master/LICENSE |
| `CTranslate2` | Inference engine used by faster-whisper | MIT | https://github.com/OpenNMT/CTranslate2 |
| `coqui-tts` | TTS Python implementation | MPL-2.0 (code) | https://github.com/idiap/coqui-ai-TTS |
| PyTorch | Tensor operations / CUDA runtime wheels | BSD-style license; see distribution notices | https://github.com/pytorch/pytorch/blob/main/LICENSE |
| Hugging Face Transformers | Model tooling / ecosystem dependency | Apache-2.0 | https://github.com/huggingface/transformers/blob/main/LICENSE |
| Hugging Face Hub | Model download/cache tooling | Apache-2.0 | https://github.com/huggingface/huggingface_hub/blob/main/LICENSE |
| NumPy | Numerical operations | BSD-3-Clause | https://github.com/numpy/numpy/blob/main/LICENSE.txt |
| Requests | HTTP requests | Apache-2.0 | https://github.com/psf/requests/blob/main/LICENSE |
| ONNX Runtime | Runtime dependency where used | MIT | https://github.com/microsoft/onnxruntime/blob/main/LICENSE |
| PySide6 / Qt for Python | Desktop UI framework | Qt licensing terms; commonly LGPLv3/GPLv3/commercial options depending on component and use | https://doc.qt.io/qtforpython-6/licenses.html |
| FFmpeg (if installed/used) | Media processing | LGPL-2.1-or-later by default; optional build features may be GPL | https://ffmpeg.org/legal.html |

## Important distribution notes

- The exact license obligations can depend on the package version, build options,
  bundled binary, and how a component is linked or distributed. Check the license
  files shipped with the exact artifacts you redistribute.
- The installer may download software/model files at install or first run rather
  than bundle them in this repository. Their upstream licenses still apply.
- Do not redistribute model weights merely because their loader or application
  code has a permissive license. See [`MODEL_LICENSES.md`](MODEL_LICENSES.md).
- Before attaching a compiled FFmpeg binary, confirm the exact build configuration
  and preserve the notices/source-offer materials required by that build's license.
- The `requirements-ui-lock.txt` and `requirements-tts-lock.txt` files pin package
  versions, but do not themselves contain a complete license inventory. For a
  release intended for broad binary redistribution, generate and review an SBOM
  and license report from the exact built environments.

This document is informational and is not legal advice.
