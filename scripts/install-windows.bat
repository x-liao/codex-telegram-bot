@echo off
setlocal EnableExtensions EnableDelayedExpansion

for %%I in ("%~dp0..") do set "ROOT_DIR=%%~fI"
set "TASK_NAME=CodexTelegramBot"
set "DEFAULT_MODEL=gpt-5.2-codex"
set "DEFAULT_SANDBOX=read-only"

echo [INFO] Root directory: %ROOT_DIR%

where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] python was not found in PATH.
  exit /b 1
)

where codex >nul 2>nul
if errorlevel 1 (
  echo [WARN] codex was not found in PATH. You can still continue if config uses an absolute codex_bin later.
)

set "TELEGRAM_BOT_TOKEN="
:ASK_TOKEN
set /p TELEGRAM_BOT_TOKEN=Telegram Bot Token: 
if "%TELEGRAM_BOT_TOKEN%"=="" (
  echo [ERROR] Token cannot be empty.
  goto ASK_TOKEN
)

set "TELEGRAM_ID="
:ASK_ID
set /p TELEGRAM_ID=Allowed Telegram numeric ID: 
if "%TELEGRAM_ID%"=="" (
  echo [ERROR] Telegram ID cannot be empty.
  goto ASK_ID
)
set "NON_NUM="
for /f "delims=0123456789" %%A in ("%TELEGRAM_ID%") do set "NON_NUM=%%A"
if defined NON_NUM (
  echo [ERROR] Telegram ID must contain only digits.
  set "NON_NUM="
  goto ASK_ID
)

set "CODEX_MODEL=%DEFAULT_MODEL%"
set /p INPUT_MODEL=Model [%DEFAULT_MODEL%]: 
if not "%INPUT_MODEL%"=="" set "CODEX_MODEL=%INPUT_MODEL%"

set "CODEX_SANDBOX=%DEFAULT_SANDBOX%"
set /p INPUT_SANDBOX=Sandbox [read-only/workspace-write/danger-full-access] (default %DEFAULT_SANDBOX%): 
if not "%INPUT_SANDBOX%"=="" set "CODEX_SANDBOX=%INPUT_SANDBOX%"
if /I not "%CODEX_SANDBOX%"=="read-only" if /I not "%CODEX_SANDBOX%"=="workspace-write" if /I not "%CODEX_SANDBOX%"=="danger-full-access" (
  echo [ERROR] Invalid sandbox value: %CODEX_SANDBOX%
  exit /b 1
)

if not exist "%ROOT_DIR%\telegram" mkdir "%ROOT_DIR%\telegram"
if not exist "%ROOT_DIR%\logs" mkdir "%ROOT_DIR%\logs"
if not exist "%ROOT_DIR%\workdir" mkdir "%ROOT_DIR%\workdir"

powershell -NoProfile -ExecutionPolicy Bypass -Command "$root=$env:ROOT_DIR; $cfg=[ordered]@{telegram_bot_token=$env:TELEGRAM_BOT_TOKEN; allow_from=@($env:TELEGRAM_ID); codex_bin='codex'; codex_workdir=(Join-Path $root 'workdir'); codex_sandbox=$env:CODEX_SANDBOX; codex_model=$env:CODEX_MODEL; codex_timeout_seconds=120; poll_timeout_seconds=30; log_file='logs/bot-runtime.log'; log_stdout=$true; text_preview_chars=100; session_enabled=$true; session_scope='chat'; session_max_messages=20; session_archive_max=10; session_title_max_chars=32; session_entry_max_chars=2000; session_system_prompt='You are a direct and practical Telegram assistant. Handle both normal conversation and coding questions. This is a continuous chat; use prior context when answering the current message.'; debug_echo_enabled=$false; debug_echo_dir='logs/codex-debug'; debug_echo_max_chars=0}; $cfg | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 (Join-Path $root 'config.json')" 
if errorlevel 1 (
  echo [ERROR] Failed to write config files.
  exit /b 1
)

set "TASK_CMD=powershell.exe -NoProfile -ExecutionPolicy Bypass -File ""%ROOT_DIR%\run-bot.ps1"""
schtasks /Create /TN "%TASK_NAME%" /SC ONLOGON /RL LIMITED /TR "%TASK_CMD%" /F >nul
if errorlevel 1 (
  echo [ERROR] Failed to create scheduled task %TASK_NAME%.
  exit /b 1
)

schtasks /Run /TN "%TASK_NAME%" >nul 2>nul

echo [OK] Deployment finished.
echo [OK] Task created: %TASK_NAME%
echo [INFO] Config: %ROOT_DIR%\config.json
echo [INFO] Allowlist key: allow_from (inside config.json)
echo [INFO] Ensure this Windows account has completed: codex login
exit /b 0
