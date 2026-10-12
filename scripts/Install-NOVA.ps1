# NOVA setup: sequential, resumable, non-destructive Windows installer.
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$LogDir = Join-Path $Root 'logs'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Log = Join-Path $LogDir 'install.log'
Start-Transcript -Path $Log -Append | Out-Null
$ServerProcess = $null
function Say($m) { Write-Host "[$(Get-Date -Format 'HH:mm:ss')] $m" }
function Fail($m) { throw $m }
function Test-Http($uri, $timeout=3) {
  try { return Invoke-RestMethod -Uri $uri -TimeoutSec $timeout -Method Get } catch { return $null }
}
function Get-NovaConfigValue([string]$Name) {
  $cfg = Join-Path $Root 'nova_paths.bat'
  if (Test-Path $cfg) {
    $pattern = '^\s*set\s+"' + [regex]::Escape($Name) + '='
    $line = Get-Content $cfg | Where-Object { $_ -match $pattern } | Select-Object -First 1
    if ($line -match ('set\s+"' + [regex]::Escape($Name) + '=([^"]*)"')) { return $Matches[1] }
  }
  return ''
}
function Get-ServerInfo($dir) {
  if (-not $dir) { return $null }
  $exe = Join-Path $dir 'llama-server.exe'
  if (Test-Path $exe) { return [pscustomobject]@{Dir=(Resolve-Path $dir).Path; Exe=(Resolve-Path $exe).Path} }
  return $null
}
function Resolve-LlamaServer {
  $candidates = @()
  $configured = Get-NovaConfigValue 'NOVA_LLAMA_DIR'
  if ($env:NOVA_LLAMA_DIR) { $candidates += $env:NOVA_LLAMA_DIR }
  if ($configured) { $candidates += $configured }
  $candidates += (Join-Path $Root 'llama-cuda')
  $candidates += 'C:\llama-cuda'
  $candidates += (Join-Path $env:USERPROFILE 'llama-cuda')
  foreach ($c in ($candidates | Select-Object -Unique)) {
    if ($c) { $found = Get-ServerInfo $c; if ($found) { Say "llama-server bulundu: $($found.Exe)"; return $found } }
  }
  Say 'Mevcut llama-server bulunamadi. Resmi llama.cpp Windows CUDA derlemesi indirilecek.'
  $target = Join-Path $Root 'llama-cuda'
  $tmp = Join-Path $env:TEMP ('nova-llama-' + [guid]::NewGuid().ToString('N'))
  New-Item -ItemType Directory -Force -Path $tmp | Out-Null
  try {
    $headers = @{ 'User-Agent'='NOVA-Installer'; 'Accept'='application/vnd.github+json' }
    $latest = Invoke-RestMethod -Headers $headers -Uri 'https://api.github.com/repos/ggml-org/llama.cpp/releases/latest'
    $nightlyTag = $null
    if ($latest.body -match 'releases/tag/(b\d+)') { $nightlyTag = $Matches[1] }
    $release = $latest
    if ($nightlyTag) {
      try { $release = Invoke-RestMethod -Headers $headers -Uri ("https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/" + $nightlyTag) } catch { $release = $latest }
    }
    $assets = @($release.assets)
    # Prefer CUDA 12.4 for broader driver compatibility; then 12.6. Do not silently jump to CUDA 13.
    $asset = $assets | Where-Object { $_.name -match '^llama-.*-bin-win-cuda-12\.4-x64\.zip$' -and $_.name -notmatch '^cudart-' } | Select-Object -First 1
    if (-not $asset) { $asset = $assets | Where-Object { $_.name -match '^llama-.*-bin-win-cuda-12\.6-x64\.zip$' -and $_.name -notmatch '^cudart-' } | Select-Object -First 1 }
    if (-not $asset) { Fail 'GitHub releases icinde desteklenen CUDA 12.4/12.6 x64 Windows llama-server paketi bulunamadi. Rastgele bir CUDA surumune gecilmedi; logu paylas.' }
    Say "İndiriliyor: $($asset.name) (resmi ggml-org/llama.cpp release)"
    $archive = Join-Path $tmp $asset.name
    Invoke-WebRequest -Headers @{ 'User-Agent'='NOVA-Installer' } -Uri $asset.browser_download_url -OutFile $archive
    Expand-Archive -LiteralPath $archive -DestinationPath (Join-Path $tmp 'extract') -Force
    $server = Get-ChildItem -Path (Join-Path $tmp 'extract') -Filter 'llama-server.exe' -File -Recurse | Select-Object -First 1
    if (-not $server) { Fail 'Indirilen resmi arşivde llama-server.exe bulunamadi.' }
    # Copy the complete directory that contains the server, preserving adjacent DLLs.
    $sourceDir = $server.Directory.FullName
    if (Test-Path $target) {
      if (Test-Path (Join-Path $target 'llama-server.exe')) { Fail 'Hedef llama-cuda klasorunde server var; ustune yazma engellendi.' }
      $target = Join-Path $Root ('llama-cuda-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
    }
    New-Item -ItemType Directory -Force -Path $target | Out-Null
    Copy-Item -Path (Join-Path $sourceDir '*') -Destination $target -Recurse -Force
    $found = Get-ServerInfo $target
    if (-not $found) { Fail 'llama-server kopyalandi ancak dogrulanamadi.' }
    Say "llama-server hazir: $($found.Exe)"
    return $found
  } finally { Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue }
}
function Save-Paths([string]$llamaDir) {
  $cfg = Join-Path $Root 'nova_paths.bat'
  $ffmpeg = (Get-Command ffmpeg.exe -ErrorAction SilentlyContinue | Select-Object -First 1).Source
  $ffmpegDir = if ($ffmpeg) { Split-Path $ffmpeg -Parent } else { '' }
  if (Test-Path $cfg) {
    $content = Get-Content $cfg -Raw
    if ($content -notmatch '(?im)^\s*set\s+"NOVA_LLAMA_DIR=') {
      Add-Content -Path $cfg -Value ("`r`nrem Added automatically by installer`r`nset `"NOVA_LLAMA_DIR=$llamaDir`"") -Encoding Ascii
      Say "Otomatik llama-server yolu mevcut nova_paths.bat dosyasina eklendi: $llamaDir"
    } else {
      Say 'Mevcut NOVA_LLAMA_DIR ayari korunuyor; otomatik olarak ezilmeyecek.'
    }
    if ($ffmpegDir -and $content -notmatch '(?im)^\s*set\s+"NOVA_FFMPEG_BIN=') {
      Add-Content -Path $cfg -Value ("`r`nset `"NOVA_FFMPEG_BIN=$ffmpegDir`"") -Encoding Ascii
    }
    return
  }
  $lines = @('@echo off', 'rem Generated by Install-NOVA.ps1. Edit paths if you move components.', ('set "NOVA_LLAMA_DIR=' + $llamaDir + '"'))
  if ($ffmpegDir) { $lines += ('set "NOVA_FFMPEG_BIN=' + $ffmpegDir + '"') }
  [IO.File]::WriteAllLines($cfg, $lines, [Text.Encoding]::ASCII)
  Say "Yol ayarlari kaydedildi: $cfg"
}
function Test-Gemma([string]$exe, [string]$dir) {
  $healthy = Test-Http 'http://127.0.0.1:8080/v1/models' 3
  if ($healthy) {
    $json = $healthy | ConvertTo-Json -Depth 8 -Compress
    if ($json -match 'gemma|unsloth') { Say '8080 portunda Gemma API zaten hazir; ikinci server acilmadi.'; return }
    Fail '127.0.0.1:8080 adresinde API var ancak model Gemma olarak dogrulanamadi. Calisan servise dokunulmayacak; portu kontrol et.'
  }
  $gemmaLog = Join-Path $LogDir 'gemma-install-smoke.log'
  $gemmaErr = Join-Path $LogDir 'gemma-install-smoke-error.log'
  $args = @('-hf','unsloth/gemma-4-E4B-it-GGUF:Q4_K_M','-ngl','99','-c','8192','-np','1','--ctx-checkpoints','1','--jinja','--reasoning','off','--host','127.0.0.1','--port','8080')
  $env:PATH = $dir + ';' + $env:PATH
  $env:NOVA_LLAMA_DIR = $dir
  $proc = $null
  try {
    Say 'Gemma baslatiliyor. Model onbelleginde yoksa llama.cpp tarafindan indirilecek; bu adim birkac dakika veya daha uzun surebilir.'
    $proc = Start-Process -FilePath $exe -ArgumentList $args -WorkingDirectory $dir -PassThru -RedirectStandardOutput $gemmaLog -RedirectStandardError $gemmaErr
    $deadline = (Get-Date).AddMinutes(25)
    $ready = $false
    while ((Get-Date) -lt $deadline) {
      Start-Sleep -Seconds 3
      if ($proc.HasExited) {
        if (Test-Path $gemmaLog) { Get-Content $gemmaLog -Tail 60 | Write-Host }
        if (Test-Path $gemmaErr) { Get-Content $gemmaErr -Tail 60 | Write-Host }
        Fail "llama-server model yuklenirken kapandi (exit $($proc.ExitCode)). Loglar: $gemmaLog / $gemmaErr"
      }
      $models = Test-Http 'http://127.0.0.1:8080/v1/models' 3
      if ($models) { $json = $models | ConvertTo-Json -Depth 8 -Compress; if ($json -match 'gemma|unsloth') { $ready = $true; break } }
    }
    if (-not $ready) {
      if (Test-Path $gemmaLog) { Get-Content $gemmaLog -Tail 60 | Write-Host }
      if (Test-Path $gemmaErr) { Get-Content $gemmaErr -Tail 60 | Write-Host }
      Fail "Gemma API 25 dakika icinde hazir olmadi. Loglar: $gemmaLog / $gemmaErr"
    }
    $models = Test-Http 'http://127.0.0.1:8080/v1/models' 3
    $modelId = @($models.data)[0].id
    if (-not $modelId) { $modelId = 'unsloth/gemma-4-E4B-it-GGUF:Q4_K_M' }
    $body = @{ model=$modelId; messages=@(@{role='user'; content='Sadece OK yaz.'}); max_tokens=8; temperature=0 } | ConvertTo-Json -Depth 6
    try {
      $answer = Invoke-RestMethod -Uri 'http://127.0.0.1:8080/v1/chat/completions' -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 120
      if (-not $answer.choices -or -not $answer.choices[0].message) { Fail 'Gemma API yanit verdi ancak chat completion sonucu dogrulanamadi.' }
      Say 'Gemma API + test istegi basarili.'
    } catch { Fail "Gemma test istegi basarisiz: $($_.Exception.Message). Log: $gemmaLog" }
  } finally {
    if ($proc -and -not $proc.HasExited) { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue; Start-Sleep -Seconds 2 }
  }
}
try {
  Say 'NOVA akilli kurulum basladi.'
  if (-not $env:OS -or $env:OS -ne 'Windows_NT') { Fail 'Bu paket ilk surumde yalnizca Windows icindir.' }
  $py = Get-Command py.exe -ErrorAction SilentlyContinue
  if (-not $py) { Fail 'Python Launcher (py.exe) bulunamadi. Python 3.11 x64 ve Python Launcher kurup yeniden calistir.' }
  & py -3.11 --version
  if ($LASTEXITCODE -ne 0) { Fail 'Python 3.11 bulunamadi. Once Python 3.11 x64 kur.' }
  $gpu = $null
  try { $gpu = (& nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>$null | Select-Object -First 1) } catch {}
  if (-not $gpu) { Fail 'NVIDIA GPU/surucu nvidia-smi ile dogrulanamadi. Bu ilk paket NVIDIA CUDA hedeflidir; AMD/CPU kurulumu bu surumde desteklenmiyor.' }
  Say "NVIDIA GPU: $gpu"
  $server = Resolve-LlamaServer
  Save-Paths $server.Dir
  # Make sure model cache exists if already downloaded; do not set LLAMA_CACHE/HF_HOME, so standard cache is reused.
  $hfCache = Join-Path $env:USERPROFILE '.cache\huggingface\hub\models--unsloth--gemma-4-E4B-it-GGUF'
  $cachedGguf = $null
  if (Test-Path $hfCache) {
    $cachedGguf = Get-ChildItem -Path (Join-Path $hfCache 'snapshots') -Filter '*.gguf' -File -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
  }
  if ($cachedGguf) {
    Say "Gemma GGUF onbellegi bulundu: $($cachedGguf.FullName). Server testi dosyanin kullanilabildigini dogrulayacak."
  } else {
    Say 'Tamamlanmis Gemma GGUF snapshot dosyasi bulunamadi; llama.cpp model dosyalarini cache icinde tamamlayacak/indirecek.'
  }
  Test-Gemma $server.Exe $server.Dir
  Say 'Gemma testi gecti. Simdi Python ortamlarini ve diger bagimliliklari kuruyoruz.'
  # Existing environments are reused. Partial broken environments are never deleted automatically.
  $uiEnv = Join-Path $Root 'jarvis_ui_env'
  $ttsEnv = Join-Path $Root 'tts_env'
  if (-not (Test-Path (Join-Path $uiEnv 'Scripts\python.exe'))) {
    if (Test-Path $uiEnv) { Fail 'jarvis_ui_env klasoru var ama python.exe eksik. Otomatik silinmedi; klasoru inceleyip elle duzelt.' }
    Say 'UI Python 3.11 ortami olusturuluyor.'
    & py -3.11 -m venv $uiEnv; if ($LASTEXITCODE -ne 0) { Fail 'UI venv olusturulamadi.' }
  } else { Say 'Mevcut UI Python ortami kullaniliyor; silinmedi.' }
  if (-not (Test-Path (Join-Path $ttsEnv 'Scripts\python.exe'))) {
    if (Test-Path $ttsEnv) { Fail 'tts_env klasoru var ama python.exe eksik. Otomatik silinmedi; klasoru inceleyip elle duzelt.' }
    Say 'TTS Python 3.11 ortami olusturuluyor.'
    & py -3.11 -m venv $ttsEnv; if ($LASTEXITCODE -ne 0) { Fail 'TTS venv olusturulamadi.' }
  } else { Say 'Mevcut TTS Python ortami kullaniliyor; silinmedi.' }
  $uiPy = Join-Path $uiEnv 'Scripts\python.exe'
  $ttsPy = Join-Path $ttsEnv 'Scripts\python.exe'
  Say 'Pip guncelleniyor (UI ortami).'
  & $uiPy -m pip install --upgrade pip; if ($LASTEXITCODE -ne 0) { Fail 'UI pip guncellemesi basarisiz.' }
  Say 'UI bagimliliklari yukleniyor.'
  & $uiPy -m pip install -r (Join-Path $Root 'requirements-ui-lock.txt'); if ($LASTEXITCODE -ne 0) { Fail 'UI bagimliliklari kurulamadi.' }
  Say 'Pip guncelleniyor (TTS ortami).'
  & $ttsPy -m pip install --upgrade pip; if ($LASTEXITCODE -ne 0) { Fail 'TTS pip guncellemesi basarisiz.' }
  Say 'PyTorch CUDA 12.6 ve XTTS bagimliliklari yukleniyor.'
  & $ttsPy -m pip install --extra-index-url 'https://download.pytorch.org/whl/cu126' -r (Join-Path $Root 'requirements-tts-lock.txt'); if ($LASTEXITCODE -ne 0) { Fail 'TTS/PyTorch bagimliliklari kurulamadi.' }
  Say 'PyTorch CUDA dogrulamasi.'
  & $ttsPy (Join-Path $Root 'scripts\smoke-test-torch.py'); if ($LASTEXITCODE -ne 0) { Fail 'PyTorch CUDA testi basarisiz. GPU/driver/PyTorch uyumunu kontrol et.' }
  Say 'Whisper CUDA DLL + model yukleme testi (Whisper small cache yoksa indirilebilir).'
  & $uiPy (Join-Path $Root 'scripts\smoke-test-whisper.py') $Root; if ($LASTEXITCODE -ne 0) { Fail 'Whisper CUDA DLL/model yukleme testi basarisiz. logs\whisper-smoke.log ve install.log dosyalarini kontrol et.' }
  Say 'XTTS import testi.'
  & $ttsPy -c "from TTS.api import TTS; print('XTTS_IMPORT_OK')"; if ($LASTEXITCODE -ne 0) { Fail 'XTTS import testi basarisiz.' }
  Say 'NOVA kurulumu ve otomatik smoke testleri tamamlandi.'
  Say 'Sonraki adim: Start-NOVA.bat ile tam mikrofon -> Whisper -> Gemma -> XTTS akis testini yap.'
} catch {
  Write-Host "`n[HATA] $($_.Exception.Message)" -ForegroundColor Red
  Write-Host "Kurulum logu: $Log"
  if ($ServerProcess -and -not $ServerProcess.HasExited) { Stop-Process -Id $ServerProcess.Id -Force -ErrorAction SilentlyContinue }
  Stop-Transcript | Out-Null
  exit 1
}
Stop-Transcript | Out-Null
exit 0
