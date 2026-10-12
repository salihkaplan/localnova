# Configuration

Machine-specific paths can be configured by copying `nova_paths.bat.example` to `nova_paths.bat`. The optional variables are `NOVA_LLAMA_DIR` (folder containing `llama-server.exe`) and `NOVA_FFMPEG_BIN` (FFmpeg `bin` folder). Relative paths are resolved from the project folder. Do not store API keys in this file.

The Brave API key is read from the process environment variable `BRAVE_API_KEY`. Verbose search logs are opt-in via `NOVA_DEBUG_SEARCH=1`; keep this disabled unless needed for troubleshooting. 
### Brave Search API Setup

NOVA's optional web search feature requires a Brave Search API key. The key is not retrieved automatically and must be configured manually.

1. Create or access your account at the [Brave Search API dashboard](https://api.search.brave.com/).
2. Generate an API key.
3. In Windows, open **Edit environment variables for your account**.
4. Under **User variables**, select **New** and enter:
   - Variable name: `BRAVE_API_KEY`
   - Variable value: your Brave Search API key
5. Save the changes and completely restart NOVA so the application can read the updated environment.

If NOVA was launched before the variable was created, it may not see the new key until restarted. If search still fails, check the diagnostic output and verify that the key is valid.

**Security:** Never commit your API key to GitHub, include it in screenshots, or distribute it inside the release package.

Web search is optional. NOVA's local voice-assistant features can continue to work without it.
