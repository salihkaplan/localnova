# Model Licenses and Sources

NOVA's MIT license covers NOVA's own code only. Model weights and quantized
model files keep their own license terms.

## Gemma 4 E4B Instruct — GGUF Q4_K_M

- **Used for:** local text generation and tool/function-call decisions.
- **Model source requested by the installer:** [`unsloth/gemma-4-E4B-it-GGUF`](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF)
- **Upstream model family license:** Apache License 2.0, according to Google's Gemma 4 model card. Review the official model card and any terms shown by the model host before downloading or redistributing weights.
- **Official model card:** https://ai.google.dev/gemma/docs/core/model_card_4
- **Apache 2.0 text:** https://www.apache.org/licenses/LICENSE-2.0

The quantized GGUF file is a conversion/quantization of model weights; do not assume
that this repository owns those weights or may redistribute them without checking
the upstream terms and the specific host's terms.

## Whisper / faster-whisper

- **Used for:** speech-to-text transcription.
- **Code:** faster-whisper is MIT-licensed; see its upstream repository.
- **Code reference:** https://github.com/SYSTRAN/faster-whisper
- **Whisper reference:** https://github.com/openai/whisper

Whisper model data is downloaded separately by the speech stack. Check the model
repository/source and terms for the exact model files selected by the installed
version. NOVA does not include Whisper weights in this source archive.

## XTTS-v2

- **Used for:** text-to-speech generation.
- **Model source:** https://huggingface.co/coqui/XTTS-v2
- **Model license:** Coqui Public Model License (CPML) 1.0.0.
- **License file:** https://huggingface.co/coqui/XTTS-v2/blob/main/LICENSE.txt
- **Model documentation:** https://docs.coqui.ai/en/latest/models/xtts.html

**Important: the XTTS-v2 model and its outputs are licensed for non-commercial use
only under the CPML.** This restriction is separate from the license of the Coqui
TTS Python code and separate from NOVA's MIT license. Anyone adapting NOVA for a
commercial purpose must replace XTTS-v2 with a suitably licensed alternative or
obtain applicable rights from the relevant rights holder. Do not assume a paid
commercial license is available.

## Other models and assets

NOVA may download additional speech models, tokenizer files, or other assets at
runtime. Their terms are determined by the exact model and source selected by the
installed package. Review the source/model card and license for each asset before
redistribution or commercial use.

This document is a project-level guide, not legal advice. Model licenses and host
terms may change; verify the current upstream terms at the time of use.
