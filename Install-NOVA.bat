@echo off
setlocal EnableExtensions
chcp 65001 >nul
title NOVA - Akilli Kurulum ve Dogrulama
cd /d "%~dp0"
if not exist "%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" (
  echo [HATA] Windows PowerShell bulunamadi.
  pause
  exit /b 1
)
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\Install-NOVA.ps1"
set "RC=%ERRORLEVEL%"
echo.
if not "%RC%"=="0" echo [HATA] Kurulum basarisiz. Ayrintilar logs\install.log dosyasinda.
if "%RC%"=="0" echo [OK] Kurulum ve otomatik kontroller tamamlandi.
pause
exit /b %RC%
