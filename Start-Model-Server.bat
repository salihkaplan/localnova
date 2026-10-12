@echo off
setlocal EnableExtensions
chcp 65001 >nul
set "ROOT=%~dp0"
cd /d "%ROOT%"
if exist "%ROOT%nova_paths.bat" call "%ROOT%nova_paths.bat"
if not defined NOVA_LLAMA_DIR if exist "%ROOT%llama-cuda\llama-server.exe" set "NOVA_LLAMA_DIR=%ROOT%llama-cuda"
if not defined NOVA_LLAMA_DIR if exist "C:\llama-cuda\llama-server.exe" set "NOVA_LLAMA_DIR=C:\llama-cuda"
if not defined NOVA_LLAMA_DIR if exist "%USERPROFILE%\llama-cuda\llama-server.exe" set "NOVA_LLAMA_DIR=%USERPROFILE%\llama-cuda"
if not defined NOVA_LLAMA_DIR (
  echo [HATA] llama-server.exe bulunamadi. Once Install-NOVA.bat calistirin.
  pause
  exit /b 1
)
for %%I in ("%NOVA_LLAMA_DIR%") do set "LLAMA_DIR=%%~fI"
set "LLAMA_EXE=%LLAMA_DIR%\llama-server.exe"
if not exist "%LLAMA_EXE%" (
  echo [HATA] llama-server.exe bulunamadi: "%LLAMA_EXE%"
  pause
  exit /b 1
)
rem Keep server DLL paths local to this process. Reuse CUDA runtime shipped in tts_env when present.
set "TTS_SITE=%ROOT%tts_env\Lib\site-packages"
set "PATH=%LLAMA_DIR%;%TTS_SITE%\nvidia\cuda_runtime\bin;%TTS_SITE%\nvidia\cublas\bin;%TTS_SITE%\nvidia\cudnn\bin;%PATH%"
set "BRAVE_API_KEY="
pushd "%LLAMA_DIR%" || (echo [HATA] llama-cuda klasorune girilemedi.& pause & exit /b 1)
echo ============================================================
echo NOVA MODEL SERVER - GEMMA 4 E4B Q4_K_M
echo ============================================================
echo Model: unsloth/gemma-4-E4B-it-GGUF:Q4_K_M
echo Klasor: %LLAMA_DIR%
echo Hugging Face cache: %%USERPROFILE%%\.cache\huggingface\hub
echo.
"%LLAMA_EXE%" -hf unsloth/gemma-4-E4B-it-GGUF:Q4_K_M -ngl 99 -c 8192 -np 1 --ctx-checkpoints 1 --jinja --reasoning off --host 127.0.0.1 --port 8080
set "SERVER_EXIT=%ERRORLEVEL%"
popd
if not "%SERVER_EXIT%"=="0" (
  echo.
  echo Gemma sunucusu hata ile kapandi. Cikis kodu: %SERVER_EXIT%
  pause
)
endlocal & exit /b %SERVER_EXIT%
