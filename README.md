[English](README.md) | [中文](README.zh.md)

# Codex Telegram Bot (Local CLI)

This project is a locally running Telegram bot.
It receives messages through Telegram long polling (`getUpdates`), calls local `codex exec`, and sends the generated response back to Telegram.

## Features

- No public endpoint required, no extra tunnel needed
- Supports allowlist-based user access control
- Supports config file and environment variables (environment variables take precedence)
- Supports runtime logs for troubleshooting

## Directory Structure

- `bot.py`: Main program (poll Telegram + call Codex)
- `run-bot.ps1`: Watchdog startup script (auto-restart on crash)
- `config.json`: Runtime config
- `allow_from` in `config.json`: Telegram user allowlist
- `telegram/update-offset-default.json`: Polling offset state
- `logs/bot-runtime.log`: Runtime logs

## Configuration

Edit `config.json`:

```json
{
  "telegram_bot_token": "your Telegram Bot Token",
  "allow_from": ["123456789"],
  "codex_bin": "C:\\Users\\admin\\AppData\\Roaming\\npm\\codex.cmd",
  "codex_workdir": "C:\\Users\\admin\\.openclaw\\workspace",
  "codex_sandbox": "read-only",
  "codex_model": "gpt-5.2-codex",
  "codex_models": ["gpt-5.2-codex", "gpt-5-codex"],
  "codex_timeout_seconds": 120,
  "poll_timeout_seconds": 30,
  "log_file": "logs/bot-runtime.log",
  "log_stdout": true,
  "text_preview_chars": 100
}
```

Notes:

- `telegram_bot_token`: Bot token
- `allow_from`: List of allowed Telegram numeric user IDs
- `codex_bin`: Path to Codex executable (absolute path recommended)
- `codex_workdir`: Codex working directory
- `codex_sandbox`: `read-only` / `workspace-write` / `danger-full-access`
- `codex_model`: Current model name; if empty, Codex default is used
- `codex_models`: Optional candidate models shown by `/models`
- `log_file`: Log file path (relative paths are based on project root)

## Allowlist

Edit `config.json` and add or update `allow_from`:

```json
{
  "allow_from": [
    "123456789"
  ]
}
```

`allow_from` must contain Telegram numeric IDs, not usernames.

## Start on Windows

Run directly:

```powershell
cd C:\Users\admin\.openclaw\workspace\codex-telegram-bot
python .\bot.py
```

Run with watchdog (auto-restart):

```powershell
powershell -ExecutionPolicy Bypass -File .\run-bot.ps1
```

## Deploy and Auto-Start on Linux / macOS

### Prerequisites

1. Install Python 3.10+ (3.11+ recommended).
2. Install Codex CLI, and make sure the runtime account has run `codex login`.
3. Prepare project directory (examples):
   - Linux: `/opt/codex-telegram-bot`
   - macOS: `/Users/<your-username>/codex-telegram-bot`
4. Ensure Telegram bot token, allowlist, and config are ready.

### Recommended cross-platform config (`config.json`)

For Linux/macOS, set `codex_bin` to `codex` (from PATH), and set `codex_workdir` to a directory inside the project:

```json
{
  "telegram_bot_token": "your Telegram Bot Token",
  "allow_from": ["123456789"],
  "codex_bin": "codex",
  "codex_workdir": "/opt/codex-telegram-bot/workdir",
  "codex_sandbox": "read-only",
  "codex_model": "gpt-5.2-codex",
  "codex_models": ["gpt-5.2-codex", "gpt-5-codex"],
  "poll_timeout_seconds": 30,
  "log_file": "logs/bot-runtime.log",
  "log_stdout": true
}
```

On macOS, only change `codex_workdir` to your actual path, for example:
`/Users/<your-username>/codex-telegram-bot/workdir`.

### Manual start and verification (Linux/macOS)

```bash
cd /opt/codex-telegram-bot
python3 bot.py
```

Open another terminal to watch logs:

```bash
tail -f logs/bot-runtime.log
```

It is recommended to verify Codex CLI first (using the same account as the service):

```bash
codex -a never exec --sandbox read-only --skip-git-repo-check -C /opt/codex-telegram-bot/workdir "reply OK"
```

---

### Linux auto-start (systemd)

1. Create `/etc/systemd/system/codex-telegram-bot.service`:

```ini
[Unit]
Description=Codex Telegram Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=codexbot
Group=codexbot
WorkingDirectory=/opt/codex-telegram-bot
Environment=PYTHONUNBUFFERED=1
Environment=PATH=/usr/local/bin:/usr/bin:/bin
ExecStart=/usr/bin/python3 /opt/codex-telegram-bot/bot.py
Restart=always
RestartSec=3
KillSignal=SIGINT
TimeoutStopSec=20

[Install]
WantedBy=multi-user.target
```

2. Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now codex-telegram-bot.service
```

3. Check status and logs:

```bash
sudo systemctl status codex-telegram-bot.service
sudo journalctl -u codex-telegram-bot.service -f
```

4. Restart after config changes:

```bash
sudo systemctl restart codex-telegram-bot.service
```

Notes:

- `User`/`Group` must be an account that has run `codex login`.
- Ensure `PATH` contains both `codex` and `node` locations for this service account.
- If using a virtual environment, update `ExecStart` to the venv Python path, for example:
  `/opt/codex-telegram-bot/.venv/bin/python /opt/codex-telegram-bot/bot.py`.

---

### macOS auto-start (launchd)

macOS does not use `systemd`; use `launchd` (`LaunchAgents`) instead.

1. Create `~/Library/LaunchAgents/com.codex.telegram.bot.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
  <dict>
    <key>Label</key>
    <string>com.codex.telegram.bot</string>

    <key>ProgramArguments</key>
    <array>
      <string>/usr/bin/python3</string>
      <string>/Users/<your-username>/codex-telegram-bot/bot.py</string>
    </array>

    <key>WorkingDirectory</key>
    <string>/Users/<your-username>/codex-telegram-bot</string>

    <key>EnvironmentVariables</key>
    <dict>
      <key>PYTHONUNBUFFERED</key>
      <string>1</string>
      <key>PATH</key>
      <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
    </dict>

    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>

    <key>StandardOutPath</key>
    <string>/Users/<your-username>/codex-telegram-bot/logs/launchd.stdout.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/<your-username>/codex-telegram-bot/logs/launchd.stderr.log</string>
  </dict>
</plist>
```

2. Load and start:

```bash
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.codex.telegram.bot.plist 2>/dev/null || true
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.codex.telegram.bot.plist
launchctl kickstart -k gui/$(id -u)/com.codex.telegram.bot
```

3. Check status:

```bash
launchctl print gui/$(id -u)/com.codex.telegram.bot
tail -f /Users/<your-username>/codex-telegram-bot/logs/bot-runtime.log
```

Notes:

- Use the current logged-in user's `LaunchAgents`; do not run as root directly.
- If `codex` is not found, update `PATH` in the plist first, or set `codex_bin` to an absolute path in `config.json` (for example `/opt/homebrew/bin/codex`).

---

### Common Linux/macOS deployment issues

- Service exits immediately after start  
  Check `logs/bot-runtime.log` and system logs first (`journalctl` on Linux, `launchctl print` on macOS).

- `Codex binary not found`  
  Set `codex_bin` to an absolute path in `config.json`, or ensure service `PATH` includes Codex install directory.

- Works manually but fails as service  
  Usually caused by different accounts between service runtime and manual execution, so `codex login` credentials are unavailable. Use the same account.

## One-Click Install and Uninstall Scripts

The repository already includes ready-to-use templates and scripts:

- `deploy/systemd/codex-telegram-bot.service.template`
- `deploy/macos/com.codex.telegram.bot.plist.template`
- `scripts/install-windows.bat`
- `scripts/uninstall-windows.bat`
- `scripts/install-unix.sh`
- `scripts/uninstall-unix.sh`

Script prompts:

- `Telegram Bot Token`
- Allowed Telegram numeric ID (`allow_from`)
- Model name (default `gpt-5.2-codex`, press Enter to keep default)
- `codex_sandbox` (default `read-only`)

### Windows one-click install

Run in project root:

```powershell
.\scripts\install-windows.bat
```

Behavior:

- Generates `config.json` (including `allow_from`)
- Creates scheduled task `CodexTelegramBot` (auto-start at user logon)
- Tries to start the task immediately

Uninstall:

```powershell
.\scripts\uninstall-windows.bat
```

This removes the scheduled task, and optionally removes local config/log/session files.

### Linux/macOS one-click install

Run in project root:

```bash
chmod +x scripts/install-unix.sh scripts/uninstall-unix.sh
bash scripts/install-unix.sh
```

Behavior:

- Generates `config.json` (including `allow_from`)
- Linux: installs and starts `systemd` service `codex-telegram-bot.service`
- macOS: installs and starts `launchd` Agent `com.codex.telegram.bot`

Uninstall:

```bash
bash scripts/uninstall-unix.sh
```

This removes system service/agent, and optionally removes local config/log/session files.

## Logs

Windows:

```powershell
Get-Content .\logs\bot-runtime.log -Wait
```

Linux/macOS:

```bash
tail -f logs/bot-runtime.log
```

## FAQ

- `Codex binary not found`  
  Set `codex_bin` in `config.json` to an absolute path, for example:
  `C:\\Users\\admin\\AppData\\Roaming\\npm\\codex.cmd`

- Response is abnormally long and looks like JSON event stream  
  Fixed to extract only the final `agent_message`. Restart `bot.py` and use the latest version.

- Received `Unauthorized user.`  
  Check whether numeric IDs in `allow_from` in `config.json` are correct.

## Security Recommendations

- Do not commit real `telegram_bot_token` to source control.
- Replace secrets in config before sharing the project.

## Minimal Production Checklist

Before enabling auto-start, check the following:

1. Runtime account
- Run the bot with the same Windows user account that already executed `codex login`.
- Manual verification: `codex -a never exec --sandbox read-only --skip-git-repo-check -C C:\Users\admin\.openclaw\workspace "reply OK"`.

2. Start mode
- Prefer `run-bot.ps1` so the process can auto-restart on crash.
- If using Task Scheduler, use:
  - Program: `powershell.exe`
  - Arguments: `-NoProfile -ExecutionPolicy Bypass -File C:\Users\admin\.openclaw\workspace\codex-telegram-bot\run-bot.ps1`
  - Start in: `C:\Users\admin\.openclaw\workspace\codex-telegram-bot`

3. Health checks
- Review `logs/bot-runtime.log` at least once per day to confirm polling is alive.
- Watch for repeated `Codex failed` or `network error`.

4. Log rotation (weekly)
- Prevent unbounded log growth.
- Example (PowerShell): archive `logs\bot-runtime.log` to `logs\archive\bot-runtime-YYYYMMDD.log`, then create a new empty log file.

5. Basic security
- Keep `allow_from` in `config.json` least-privilege (only your IDs).
- Use `codex_sandbox: read-only` by default.
- Do not commit real `telegram_bot_token` to Git.

6. Backups
- Recommended backups:
  - `config.json`
  - `telegram/update-offset-default.json`

7. Recovery
- If the bot stops responding:
  - Restart process or script.
  - Check the tail of `logs/bot-runtime.log`.
  - Run one manual `codex exec` using the same account to verify auth and network.

## Permissions and Sandbox

`codex_sandbox` controls what Codex can access.

### 1) `read-only` (recommended default)

Use cases:

- Daily Q&A, code reading, troubleshooting, read-only analysis
- Preferred secure mode for production

Behavior:

- Can read files under `codex_workdir`
- Cannot write/create/delete files
- Cannot run commands with write side effects

Pros:

- Lowest risk and least chance of accidental file modification

Limits:

- Cannot modify code automatically
- Cannot run commands that require write permission (for example generating files, installing dependencies into project directories)

---

### 2) `workspace-write`

Use cases:

- Need Codex to directly modify project code
- Need to create/update files inside project directory

Behavior:

- Can read and write files under `codex_workdir`
- Cannot access paths outside `codex_workdir` by default
- Use `codex_add_dirs` to explicitly allow additional paths

Pros:

- Good balance between safety and usability for automatic code changes within controlled scope

Limits:

- Paths outside sandbox scope are still restricted
- System-level controls (ACL, antivirus, group policy) may still block operations

---

### 3) `danger-full-access` (high risk)

Use cases:

- Explicitly need arbitrary system path access or high-privilege operations
- Only when you fully trust the current prompt and operations

Behavior:

- Disables Codex sandbox restrictions
- Runs with current Windows user privileges
- `codex_add_dirs` is mostly irrelevant in this mode

Risks:

- Highest cost of mistakes, may affect files outside project or system state
- More likely to cause irreversible changes

Recommendations:

- Avoid unless necessary
- Enable briefly and revert to `read-only` or `workspace-write` after finishing

Example (`config.json`):

```json
{
  "codex_sandbox": "workspace-write",
  "codex_add_dirs": [
    "C:\\Users\\admin\\Desktop",
    "C:\\Users\\admin\\Documents"
  ]
}
```

For full access (high risk):

```json
{
  "codex_sandbox": "danger-full-access",
  "codex_add_dirs": []
}
```

Notes:

- `codex_add_dirs` only applies to non-full-access modes.
- Restart the bot after changes.
- You can verify active settings in `logs/bot-runtime.log`:
  - `sandbox=...`
  - `[codex] add_dirs=[...]`
- Even with full access, system-level controls (Windows policy, ACL, antivirus) may still block command execution.

## Session Memory and New Sessions

The bot supports persistent context by chat/user and supports creating new sessions.

Commands:

- `/new`: Create a new session and clear current context
- `/forget`: Clear history for current session but keep the session ID
- `/session`: Show current session info
- `/sessions`: Open clickable session menu (with pagination), click to switch
- `/history [N]`: Show last N messages in current session (default 10)
- `/models`: Show selectable model list, click to switch
- `/models add <model-name>`: Add model manually and validate; if valid, add to model list

Session storage file:

- `telegram/session-store.json`

Related settings in `config.json`:

```json
{
  "session_enabled": true,
  "session_scope": "chat",
  "session_max_messages": 20,
  "session_archive_max": 10,
  "session_title_max_chars": 32,
  "session_entry_max_chars": 2000,
  "session_system_prompt": "You are a professional, direct, and execution-focused coding assistant. This is a persistent session; answer with historical context."
}
```

Notes:

- `/new` archives the previous session for later review.
- Archived sessions are not automatically injected into new session context.
- Session title is auto-generated from user messages and shown in `/sessions` as `#session-id title`.
- `session_scope` options:
  - `chat`: one session per chat (recommended)
  - `user`: one session per user across chats
  - `chat_user`: one session per `(chat, user)`
- Restart `bot.py` after changing session settings.
