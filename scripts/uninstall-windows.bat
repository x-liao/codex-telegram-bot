@echo off
setlocal EnableExtensions EnableDelayedExpansion

for %%I in ("%~dp0..") do set "ROOT_DIR=%%~fI"
set "TASK_NAME=CodexTelegramBot"

schtasks /Query /TN "%TASK_NAME%" >nul 2>nul
if errorlevel 1 (
  echo [INFO] Scheduled task %TASK_NAME% does not exist.
) else (
  schtasks /End /TN "%TASK_NAME%" >nul 2>nul
  schtasks /Delete /TN "%TASK_NAME%" /F >nul
  if errorlevel 1 (
    echo [WARN] Failed to delete scheduled task %TASK_NAME%.
  ) else (
    echo [OK] Removed scheduled task %TASK_NAME%.
  )
)

set "REMOVE_FILES="
set /p REMOVE_FILES=Remove generated files (config, session, logs)? [y/N]: 
if /I "%REMOVE_FILES%"=="Y" goto REMOVE_LOCAL
if /I "%REMOVE_FILES%"=="YES" goto REMOVE_LOCAL
goto DONE

:REMOVE_LOCAL
if exist "%ROOT_DIR%\config.json" del /f /q "%ROOT_DIR%\config.json"
if exist "%ROOT_DIR%\telegram\update-offset-default.json" del /f /q "%ROOT_DIR%\telegram\update-offset-default.json"
if exist "%ROOT_DIR%\telegram\session-store.json" del /f /q "%ROOT_DIR%\telegram\session-store.json"
if exist "%ROOT_DIR%\telegram\bot-instance.lock" del /f /q "%ROOT_DIR%\telegram\bot-instance.lock"
if exist "%ROOT_DIR%\logs" rmdir /s /q "%ROOT_DIR%\logs"
echo [OK] Local generated files removed.

:DONE
echo [OK] Uninstall finished.
exit /b 0
