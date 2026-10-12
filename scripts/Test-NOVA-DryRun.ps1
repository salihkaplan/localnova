# NOVA read-only preflight / dry-run. Does not install, download, delete, or modify runtime files.
[CmdletBinding()]
param(
  [switch]$SkipNetworkCheck
)
$ErrorActionPreference = 'Continue'
$ProgressPreference = 'SilentlyContinue'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Report = New-Object System.Collections.Generic.List[string]
$Counts = @{ PASS = 0; WARN = 0; FAIL = 0; INFO = 0 }

function Add-Report([string]$Level, [string]$Message) {
  $line = ('[{0}] {1}' -f $Level, $Message)
  $Report.Add($line)
  switch ($Level) { 'PASS' { $Counts.PASS++ } 'WARN' { $Counts.WARN++ } 'FAIL' { $Counts.FAIL++ } default { $Counts.INFO++ } }
  $color = switch ($Level) { 'PASS' { 'Green' } 'WARN' { 'Yellow' } 'FAIL' { 'Red' } default { 'Gray' } }
  Write-Host $line -ForegroundColor $color
}
function Read-NovaConfig([string]$Name) {
  $cfg = Join-Path $Root 'nova_paths.bat'
  if (-not (Test-Path -LiteralPath $cfg)) { return '' }
  $pattern = '^\s*set\s+"' + [regex]::Escape($Name) + '=([^"]*)"'
  foreach ($line in (Get-Content -LiteralPath $cfg -ErrorAction SilentlyContinue)) {
    if ($line -match $pattern) { return $Matches[1] }
  }
  return ''
}
function Get-ServerAt([string]$Dir) {
  if ([string]::IsNullOrWhiteSpace($Dir)) { return $null }
  try {
    $expanded = [Environment]::ExpandEnvironmentVariables($Dir)
    if (Test-Path -LiteralPath $expanded -PathType Container) {
      $exe = Join-Path $expanded 'llama-server.exe'
      if (Test-Path -LiteralPath $exe -PathType Leaf) {
        return [pscustomobject]@{ Dir=(Resolve-Path -LiteralPath $expanded).Path; Exe=(Resolve-Path -LiteralPath $exe).Path }
      }
    }
  } catch {}
  return $null
}
function Test-DistInfo([string]$Site, [string[]]$Patterns) {
  if (-not (Test-Path -LiteralPath $Site -PathType Container)) { return $false }
  foreach ($pattern in $Patterns) {
    $found = Get-ChildItem -LiteralPath $Site -Directory -Filter $pattern -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($found) { return $true }
  }
  return $false
}

Write-Host ''
Write-Host '============================================================' -ForegroundColor Cyan
Write-Host ' NOVA - READ-ONLY DRY RUN / INSTALL PLAN' -ForegroundColor Cyan
Write-Host '============================================================' -ForegroundColor Cyan
Write-Host ('Project root: {0}' -f $Root)
Write-Host 'No package installation, model download, deletion, or runtime-file changes will be performed.'
Write-Host 'The optional GitHub check uses read-only HTTP requests only.'
Write-Host ''

# 1) Package structure
Add-Report 'INFO' '1/7 - Package structure'
$required = @(
  'nova_app.py', 'Install-NOVA.bat', 'Start-NOVA.bat', 'Start-Model-Server.bat',
  'requirements-ui-lock.txt', 'requirements-tts-lock.txt',
  'scripts\Install-NOVA.ps1', 'scripts\smoke-test-torch.py', 'scripts\smoke-test-whisper.py'
)
$missing = @($required | Where-Object { -not (Test-Path -LiteralPath (Join-Path $Root $_) -PathType Leaf) })
if ($missing.Count -eq 0) { Add-Report 'PASS' 'All required installer, app, requirements, and smoke-test files are present.' }
else { Add-Report 'FAIL' ('Missing package files: ' + ($missing -join ', ')) }

# 2) OS and Python
Add-Report 'INFO' '2/7 - Operating system and Python'
if ($env:OS -eq 'Windows_NT') { Add-Report 'PASS' 'Windows environment detected.' } else { Add-Report 'FAIL' 'This installer target is Windows only.' }
$pyCmd = Get-Command 'py.exe' -ErrorAction SilentlyContinue
if (-not $pyCmd) { Add-Report 'FAIL' 'Python Launcher py.exe was not found; installer prerequisite is missing.' }
else {
  $pyVersion = (& $pyCmd.Source -3.11 --version 2>&1 | Out-String).Trim()
  if ($LASTEXITCODE -eq 0 -and $pyVersion -match 'Python 3\.11') { Add-Report 'PASS' ('Python 3.11 launcher found: ' + $pyVersion) }
  else { Add-Report 'FAIL' ('Python 3.11 was not confirmed. py.exe output: ' + $pyVersion) }
}

# 3) NVIDIA / driver
Add-Report 'INFO' '3/7 - NVIDIA GPU and driver'
$nvidia = Get-Command 'nvidia-smi.exe' -ErrorAction SilentlyContinue
if (-not $nvidia) { $nvidia = Get-Command 'nvidia-smi' -ErrorAction SilentlyContinue }
if (-not $nvidia) { Add-Report 'FAIL' 'nvidia-smi was not found on PATH. Current installer target requires an NVIDIA GPU/driver.' }
else {
  $gpuInfo = (& $nvidia.Source --query-gpu=name,driver_version --format=csv,noheader 2>&1 | Out-String).Trim()
  if ($LASTEXITCODE -eq 0 -and $gpuInfo) { Add-Report 'PASS' ('NVIDIA GPU/driver detected: ' + ($gpuInfo -replace '[\r\n]+', '; ')) }
  else { Add-Report 'FAIL' ('nvidia-smi exists but could not query a GPU: ' + $gpuInfo) }
}

# 4) Path resolution (same order as installer / launchers)
Add-Report 'INFO' '4/7 - llama-server path discovery'
$configured = Read-NovaConfig 'NOVA_LLAMA_DIR'
$candidates = @()
if ($env:NOVA_LLAMA_DIR) { $candidates += $env:NOVA_LLAMA_DIR }
if ($configured) { $candidates += $configured }
$candidates += (Join-Path $Root 'llama-cuda')
$candidates += 'C:\llama-cuda'
if ($env:USERPROFILE) { $candidates += (Join-Path $env:USERPROFILE 'llama-cuda') }
$server = $null
foreach ($candidate in ($candidates | Select-Object -Unique)) {
  $resolved = Get-ServerAt $candidate
  if ($resolved) { $server = $resolved; break }
}
if ($server) { Add-Report 'PASS' ('llama-server auto-discovered: ' + $server.Exe) }
else { Add-Report 'WARN' 'No existing llama-server.exe found in configured/project/known paths. A normal install would attempt to download a supported official CUDA 12.x build into the project directory.' }
Add-Report 'INFO' ('Path candidates checked: ' + (($candidates | Select-Object -Unique) -join ' | '))
$ffmpegCfg = Read-NovaConfig 'NOVA_FFMPEG_BIN'
$ffmpegCmd = Get-Command 'ffmpeg.exe' -ErrorAction SilentlyContinue
if ($ffmpegCfg -and (Test-Path -LiteralPath (Join-Path $ffmpegCfg 'ffmpeg.exe') -PathType Leaf)) { Add-Report 'PASS' ('FFmpeg path config resolves: ' + (Join-Path $ffmpegCfg 'ffmpeg.exe')) }
elseif ($ffmpegCmd) { Add-Report 'PASS' ('FFmpeg is available on PATH: ' + $ffmpegCmd.Source) }
else { Add-Report 'WARN' 'FFmpeg was not found by config or PATH. Confirm whether this package/runtime requires an external FFmpeg binary.' }

# 5) Gemma cache and API (read-only)
Add-Report 'INFO' '5/7 - Gemma cache and local API'
$hubCache = ''
if ($env:HUGGINGFACE_HUB_CACHE) { $hubCache = $env:HUGGINGFACE_HUB_CACHE }
elseif ($env:HF_HOME) { $hubCache = Join-Path $env:HF_HOME 'hub' }
elseif ($env:USERPROFILE) { $hubCache = Join-Path $env:USERPROFILE '.cache\huggingface\hub' }
$repoCache = if ($hubCache) { Join-Path $hubCache 'models--unsloth--gemma-4-E4B-it-GGUF' } else { '' }
if ($repoCache -and (Test-Path -LiteralPath (Join-Path $repoCache 'snapshots') -PathType Container)) {
  $gguf = Get-ChildItem -LiteralPath (Join-Path $repoCache 'snapshots') -Filter '*.gguf' -File -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($gguf) { Add-Report 'PASS' ('Gemma GGUF snapshot file found in cache: ' + $gguf.FullName) }
  else { Add-Report 'WARN' ('Gemma cache directory exists but no GGUF snapshot file was found: ' + $repoCache) }
} else { Add-Report 'WARN' ('Gemma cache was not found at the resolved standard Hugging Face cache: ' + $repoCache) }
try {
  $api = Invoke-RestMethod -Uri 'http://127.0.0.1:8080/v1/models' -TimeoutSec 2 -Method Get
  $apiJson = $api | ConvertTo-Json -Depth 8 -Compress
  if ($apiJson -match 'gemma|unsloth') { Add-Report 'PASS' 'Local API 127.0.0.1:8080 responds and appears to serve Gemma.' }
  elseif ($api) { Add-Report 'WARN' 'Local API 127.0.0.1:8080 responds, but its model ID was not recognized as Gemma/Unsloth.' }
} catch { Add-Report 'INFO' 'No responding local model API was detected on 127.0.0.1:8080. This is normal if the model server is stopped.' }

# 6) Existing virtual environments and CUDA DLLs (filesystem-only; do not import packages)
Add-Report 'INFO' '6/7 - Python environments and CUDA DLL files'
$uiEnv = Join-Path $Root 'jarvis_ui_env'
$ttsEnv = Join-Path $Root 'tts_env'
$uiPy = Join-Path $uiEnv 'Scripts\python.exe'
$ttsPy = Join-Path $ttsEnv 'Scripts\python.exe'
if (Test-Path -LiteralPath $uiPy -PathType Leaf) { Add-Report 'PASS' ('UI Python environment exists: ' + $uiPy) }
elseif (Test-Path -LiteralPath $uiEnv) { Add-Report 'WARN' 'jarvis_ui_env folder exists but Scripts\python.exe is missing; installer intentionally will not delete it automatically.' }
else { Add-Report 'WARN' 'jarvis_ui_env does not exist; a normal install would create it and install UI requirements.' }
if (Test-Path -LiteralPath $ttsPy -PathType Leaf) { Add-Report 'PASS' ('TTS Python environment exists: ' + $ttsPy) }
elseif (Test-Path -LiteralPath $ttsEnv) { Add-Report 'WARN' 'tts_env folder exists but Scripts\python.exe is missing; installer intentionally will not delete it automatically.' }
else { Add-Report 'WARN' 'tts_env does not exist; a normal install would create it and install PyTorch CUDA 12.6 / XTTS requirements.' }
$site = Join-Path $ttsEnv 'Lib\site-packages'
$dllPaths = @(
  (Join-Path $site 'torch\lib\cublas64_12.dll'),
  (Join-Path $site 'torch\lib\cublasLt64_12.dll'),
  (Join-Path $site 'nvidia\cublas\bin\cublas64_12.dll'),
  (Join-Path $site 'nvidia\cublas\bin\cublasLt64_12.dll')
)
foreach ($dllName in @('cublas64_12.dll','cublasLt64_12.dll')) {
  $foundDll = @($dllPaths | Where-Object { (Split-Path $_ -Leaf) -eq $dllName -and (Test-Path -LiteralPath $_ -PathType Leaf) })
  if ($foundDll.Count -gt 0) { Add-Report 'PASS' ($dllName + ' file found: ' + ($foundDll -join '; ')) }
  else { Add-Report 'WARN' ($dllName + ' not found in expected tts_env paths. Dry-run checks file presence only; it does not attempt to load DLLs.') }
}
if (Test-Path -LiteralPath $uiPy -PathType Leaf) {
  $uiSite = Join-Path $uiEnv 'Lib\site-packages'
  if (Test-DistInfo $uiSite @('faster_whisper-*.dist-info','faster_whisper-*.egg-info')) { Add-Report 'PASS' 'faster-whisper package metadata appears installed in UI environment.' }
  else { Add-Report 'WARN' 'faster-whisper package metadata was not found in UI environment.' }
}
if (Test-Path -LiteralPath $ttsPy -PathType Leaf) {
  $ttsSite = Join-Path $ttsEnv 'Lib\site-packages'
  if (Test-DistInfo $ttsSite @('torch-*.dist-info')) { Add-Report 'PASS' 'PyTorch package metadata appears installed in TTS environment.' }
  else { Add-Report 'WARN' 'PyTorch package metadata was not found in TTS environment.' }
  if (Test-DistInfo $ttsSite @('TTS-*.dist-info','coqui_tts-*.dist-info')) { Add-Report 'PASS' 'Coqui TTS package metadata appears installed in TTS environment.' }
  else { Add-Report 'WARN' 'Coqui TTS package metadata was not found in TTS environment.' }
}

# 7) Lock files and clean-machine plan
Add-Report 'INFO' '7/7 - Dependency lock files and hypothetical clean-machine plan'
foreach ($req in @('requirements-ui-lock.txt','requirements-tts-lock.txt')) {
  $reqPath = Join-Path $Root $req
  if (Test-Path -LiteralPath $reqPath -PathType Leaf) {
    $entries = @(Get-Content -LiteralPath $reqPath | Where-Object { $_.Trim() -and -not $_.Trim().StartsWith('#') })
    Add-Report 'PASS' ('Lock file readable: ' + $req + ' (' + $entries.Count + ' package entries)')
  } else { Add-Report 'FAIL' ('Required lock file missing: ' + $req) }
}
Add-Report 'INFO' 'Clean-machine simulation (hypothetical; no files were changed):'
Add-Report 'INFO' ' - No llama-server.exe: installer attempts official ggml-org/llama.cpp CUDA 12.4, then 12.6 asset lookup.'
Add-Report 'INFO' ' - No Gemma cache: llama.cpp -hf should download the requested Gemma GGUF model during server startup.'
Add-Report 'INFO' ' - No Python venvs: installer creates jarvis_ui_env and tts_env, then installs pinned requirements.'
Add-Report 'INFO' ' - Missing cache/venv on this machine is not itself an error; normal install would create/download those components.'

if (-not $SkipNetworkCheck) {
  Add-Report 'INFO' 'Read-only GitHub release metadata check (no asset download)'
  try {
    $headers = @{ 'User-Agent'='NOVA-DryRun'; 'Accept'='application/vnd.github+json' }
    $release = Invoke-RestMethod -Headers $headers -Uri 'https://api.github.com/repos/ggml-org/llama.cpp/releases/latest' -TimeoutSec 12
    $nightlyTag = $null
    if ($release.body -match 'releases/tag/(b\d+)') { $nightlyTag = $Matches[1] }
    if ($nightlyTag) {
      try { $release = Invoke-RestMethod -Headers $headers -Uri ('https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/' + $nightlyTag) -TimeoutSec 12 } catch {}
    }
    $asset = @($release.assets | Where-Object { $_.name -match '^llama-.*-bin-win-cuda-12\.4-x64\.zip$' -and $_.name -notmatch '^cudart-' } | Select-Object -First 1)
    if (-not $asset) { $asset = @($release.assets | Where-Object { $_.name -match '^llama-.*-bin-win-cuda-12\.6-x64\.zip$' -and $_.name -notmatch '^cudart-' } | Select-Object -First 1) }
    if ($asset.Count -gt 0) { Add-Report 'PASS' ('GitHub metadata has a matching supported asset: ' + $asset[0].name) }
    else { Add-Report 'WARN' 'Current GitHub release metadata did not expose an asset matching the installer CUDA 12.4/12.6 filename patterns. Installer may need its asset selector updated.' }
  } catch { Add-Report 'WARN' ('Could not query GitHub release metadata; this may be a network/proxy/rate-limit issue. No download was attempted. ' + $_.Exception.Message) }
} else { Add-Report 'INFO' 'GitHub network metadata check skipped by request.' }

Write-Host ''
Write-Host '-------------------- SUMMARY --------------------' -ForegroundColor Cyan
Write-Host ('PASS: {0}   WARN: {1}   FAIL: {2}   INFO: {3}' -f $Counts.PASS, $Counts.WARN, $Counts.FAIL, $Counts.INFO)
Write-Host 'This is a read-only inventory and plan, not a clean-Windows installation test.'
Write-Host 'No report file is written; copy the console output if you want to share it.'
Write-Host '-------------------------------------------------' -ForegroundColor Cyan
if ($Counts.FAIL -gt 0) { exit 2 }
exit 0
