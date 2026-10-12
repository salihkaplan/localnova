@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"
set "ROOT=%~dp0"
if exist "%ROOT%nova_paths.bat" call "%ROOT%nova_paths.bat"

rem Path discovery order: explicit environment/config, project-local folder, then known legacy path.
if not defined NOVA_LLAMA_DIR if exist "%ROOT%llama-cuda\llama-server.exe" set "NOVA_LLAMA_DIR=%ROOT%llama-cuda"
if not defined NOVA_LLAMA_DIR if exist "C:\llama-cuda\llama-server.exe" set "NOVA_LLAMA_DIR=C:\llama-cuda"
if not defined NOVA_LLAMA_DIR if exist "%USERPROFILE%\llama-cuda\llama-server.exe" set "NOVA_LLAMA_DIR=%USERPROFILE%\llama-cuda"
if not defined NOVA_LLAMA_DIR (
  echo [HATA] llama-server.exe otomatik bulunamadi. Install-NOVA.bat dosyasini calistirin.
  pause
  exit /b 1
)
if not exist "%NOVA_LLAMA_DIR%\llama-server.exe" (
  echo [HATA] NOVA_LLAMA_DIR gecersiz: "%NOVA_LLAMA_DIR%"
  pause
  exit /b 1
)
set "UI_PYW=%ROOT%jarvis_ui_env\Scripts\pythonw.exe"
if not exist "%UI_PYW%" (
  echo [HATA] NOVA Python ortami bulunamadi: "%UI_PYW%"
  echo Once Install-NOVA.bat dosyasini calistirin.
  pause
  exit /b 1
)
if not exist "%ROOT%nova_app.py" (
  echo [HATA] nova_app.py bu klasorde bulunamadi.
  pause
  exit /b 1
)
echo [PATH] llama-server: "%NOVA_LLAMA_DIR%"
start "NOVA" "%UI_PYW%" "%ROOT%nova_app.py"
endlocal
exit /b 0
