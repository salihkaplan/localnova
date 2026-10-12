# Architecture

This refactor uses conservative file renames and reference updates. The high-coupling conversation engine remains a single module for now to reduce the risk of breaking the working voice flow.

```text
nova_app.py
  ├── conversation_engine.py
  │    ├── actions/dispatcher.py
  │    ├── speech_utils.py
  │    ├── SQLite long-term history
  │    ├── local LLM HTTP API (127.0.0.1:8080)
  │    └── XTTS HTTP API (127.0.0.1:5000)
  ├── system_monitor.py
  ├── speech_service.py
  └── Start-Model-Server.bat
```

## Service endpoints

- llama.cpp OpenAI-compatible API: `http://127.0.0.1:8080`
- XTTS service: `http://127.0.0.1:5000/tts_stream`
- XTTS diagnostics: `/memory` and `/memory_detail`

The services bind to loopback in the current source. Ports and endpoints are still duplicated as constants in more than one file; a later change should centralize configuration only after regression tests exist.
