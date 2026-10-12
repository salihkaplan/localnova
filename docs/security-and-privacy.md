# Security and privacy notes

## External requests
- The conversation engine sends generated research queries to Brave Search LLM Context API when research routing selects an external search. The API key is read from `BRAVE_API_KEY` and sent in the `X-Subscription-Token` header. The key is not embedded in this package. The key is removed from the environment passed to the local Gemma and XTTS child processes because they do not need it.
- Ordinary chat requests are sent to the local llama.cpp API at `127.0.0.1:8080`; speech synthesis requests are sent to the local XTTS service at `127.0.0.1:5000`.
- This is not an assertion that every dependency is offline; Brave search is an external network service.

## Persistent local data
- `nova_memory.sqlite3` stores the conversation archive and selected long-term memories in the project folder.
- `nova_router.log` stores routing diagnostics; query text is redacted by default in this package.
- `nova_search_debug.log` may contain full search queries and returned snippets, but is only written when `NOVA_DEBUG_SEARCH=1` is explicitly enabled.
- `gemma_ui.log` and `xtts_debug.log` contain process diagnostics. Review all logs before sharing them.
- User files/screenshots/audio may be created by explicitly requested features; these are not automatically bundled.

## Actions and local APIs
- The model and speech service bind to loopback. Do not expose them to a LAN or public network without authentication and a separate security design.
- File operations are restricted to common folders under the current user's profile. Move, rename, and delete require confirmation; directory deletion is not supported.
- App actions use a fixed allowlist. Still test actions carefully before broader distribution.

## API key handling
Never place API keys in source files, BAT files, archives, logs, or public repositories. Configure `BRAVE_API_KEY` as a Windows user environment variable and restart NOVA after changing it. If a key leaks, revoke and rotate it.

## Distribution warning
No secrets, user database, runtime logs, virtual environments, cached models, or model weights should be included in a public archive. This is a source/setup candidate and has not undergone a formal penetration test.
