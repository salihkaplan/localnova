# Regression test plan

Run these checks in the actual supported Windows environment before treating this as a release:

1. Compile all Python source files.
2. Launch `Start-NOVA.bat`; verify the UI opens.
3. Verify the engine imports and microphone capture/transcription work.
4. Verify Gemma requests reach `127.0.0.1:8080`.
5. Verify XTTS launches from `speech_service.py` and `/tts_stream` returns audio.
6. Verify `system_monitor.py` identifies the XTTS process by its renamed script.
7. Verify long-term history schema initialization and persistence.
8. Verify a second UI launch does not create duplicate model servers.
9. Verify UI shutdown stops only processes it owns.
10. Run 10–15 voice interactions and compare RAM behavior.
