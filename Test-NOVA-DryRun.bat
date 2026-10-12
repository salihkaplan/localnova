@echo off
setlocal EnableExtensions
chcp 65001 >nul
title NOVA - Read-only Dry Run
cd /d "%~dp0"
if not exist "%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" (
  echo [ERROR] Windows PowerShell was not found.
  pause
  exit /b 1
)
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\Test-NOVA-DryRun.ps1"
set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" echo [OK] Dry-run finished. No installation or download was performed.
if not "%RC%"=="0" echo [WARN] Dry-run reported one or more hard failures. Review the output above.
echo.
pause
exit /b %RC%
