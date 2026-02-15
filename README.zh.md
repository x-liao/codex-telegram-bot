[English](README.md) | [中文](README.zh.md)

# Codex Telegram Bot（本地 CLI）

本项目是一个在本地运行的 Telegram 机器人。
它通过 Telegram 长轮询（`getUpdates`）接收消息，调用本机 `codex exec` 生成回复，并将结果发回 Telegram。

## 功能特性

- 无需公网地址，也不需要额外隧道
- 支持基于白名单的用户访问控制
- 支持配置文件和环境变量（环境变量优先）
- 支持运行日志，便于排障

## 目录结构

- `bot.py`：主程序（轮询 Telegram + 调用 Codex）
- `run-bot.ps1`：守护启动脚本（崩溃自动重启）
- `config.json`：运行配置
- `config.json` 中的 `allow_from`：Telegram 用户白名单
- `telegram/update-offset-default.json`：轮询 offset 状态
- `logs/bot-runtime.log`：运行日志

## 配置说明

编辑 `config.json`：

```json
{
  "telegram_bot_token": "你的 Telegram Bot Token",
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

说明：

- `telegram_bot_token`：机器人令牌
- `allow_from`：允许访问的 Telegram 数字用户 ID 列表
- `codex_bin`：Codex 可执行文件路径（建议绝对路径）
- `codex_workdir`：Codex 工作目录
- `codex_sandbox`：`read-only` / `workspace-write` / `danger-full-access`
- `codex_model`：当前模型名；为空时使用 Codex 默认值
- `codex_models`：`/models` 展示的候选模型列表（可选）
- `log_file`：日志文件路径（相对路径基于项目根目录）

## 白名单

编辑 `config.json`，新增或修改 `allow_from`：

```json
{
  "allow_from": [
    "123456789"
  ]
}
```

`allow_from` 必须填写 Telegram 数字 ID，而不是用户名。

## Windows 启动方式

直接运行：

```powershell
cd C:\Users\admin\.openclaw\workspace\codex-telegram-bot
python .\bot.py
```

守护运行（自动重启）：

```powershell
powershell -ExecutionPolicy Bypass -File .\run-bot.ps1
```

## Linux / macOS 部署与开机自启

### 前置条件

1. 安装 Python 3.10+（建议 3.11+）。
2. 安装 Codex CLI，并确保运行账号已经执行过 `codex login`。
3. 准备项目目录（示例）：
   - Linux：`/opt/codex-telegram-bot`
   - macOS：`/Users/<你的用户名>/codex-telegram-bot`
4. 确保 Telegram 机器人 Token、白名单与配置文件已准备好。

### 跨平台推荐配置（`config.json`）

Linux/macOS 建议将 `codex_bin` 设为 `codex`（走 PATH），并将 `codex_workdir` 设为项目内目录：

```json
{
  "telegram_bot_token": "你的 Telegram Bot Token",
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

在 macOS 上，只需把 `codex_workdir` 改成你的实际路径，例如：
`/Users/<你的用户名>/codex-telegram-bot/workdir`。

### 手动启动与验证（Linux/macOS）

```bash
cd /opt/codex-telegram-bot
python3 bot.py
```

另开一个终端查看日志：

```bash
tail -f logs/bot-runtime.log
```

建议先验证 Codex CLI 可用（使用与服务相同的账号）：

```bash
codex -a never exec --sandbox read-only --skip-git-repo-check -C /opt/codex-telegram-bot/workdir "reply OK"
```

---

### Linux 开机自启（systemd）

1. 创建 `/etc/systemd/system/codex-telegram-bot.service`：

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

2. 启用并启动：

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now codex-telegram-bot.service
```

3. 查看状态与日志：

```bash
sudo systemctl status codex-telegram-bot.service
sudo journalctl -u codex-telegram-bot.service -f
```

4. 配置变更后重启：

```bash
sudo systemctl restart codex-telegram-bot.service
```

说明：

- `User`/`Group` 必须是执行过 `codex login` 的账号。
- 请确保服务账号的 `PATH` 同时包含 `codex` 与 `node` 所在目录。
- 如果使用虚拟环境，请把 `ExecStart` 改为虚拟环境 Python 路径，例如：
  `/opt/codex-telegram-bot/.venv/bin/python /opt/codex-telegram-bot/bot.py`。

---

### macOS 开机自启（launchd）

macOS 不使用 `systemd`，请使用 `launchd`（`LaunchAgents`）。

1. 创建 `~/Library/LaunchAgents/com.codex.telegram.bot.plist`：

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

2. 加载并启动：

```bash
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.codex.telegram.bot.plist 2>/dev/null || true
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.codex.telegram.bot.plist
launchctl kickstart -k gui/$(id -u)/com.codex.telegram.bot
```

3. 查看状态：

```bash
launchctl print gui/$(id -u)/com.codex.telegram.bot
tail -f /Users/<your-username>/codex-telegram-bot/logs/bot-runtime.log
```

说明：

- 请使用当前登录用户的 `LaunchAgents`，不要直接以 root 运行。
- 如果找不到 `codex`，优先补充 plist 里的 `PATH`，或在 `config.json` 中把 `codex_bin` 设为绝对路径（例如 `/opt/homebrew/bin/codex`）。

---

### Linux/macOS 常见部署问题

- 服务启动后立刻退出  
  先检查 `logs/bot-runtime.log` 与系统日志（Linux 用 `journalctl`，macOS 用 `launchctl print`）。

- `Codex binary not found`  
  在 `config.json` 中把 `codex_bin` 设为绝对路径，或确保服务环境变量 `PATH` 包含 Codex 安装目录。

- 手工运行正常，服务模式失败  
  通常是服务运行账号与手工运行账号不同，导致 `codex login` 凭据不可用。请统一账号。

## 一键安装与卸载脚本

仓库已提供开箱即用的模板与脚本：

- `deploy/systemd/codex-telegram-bot.service.template`
- `deploy/macos/com.codex.telegram.bot.plist.template`
- `scripts/install-windows.bat`
- `scripts/uninstall-windows.bat`
- `scripts/install-unix.sh`
- `scripts/uninstall-unix.sh`

脚本会询问：

- `Telegram Bot Token`
- 允许访问的 Telegram 数字 ID（`allow_from`）
- 模型名（默认 `gpt-5.2-codex`，回车保留默认）
- `codex_sandbox`（默认 `read-only`）

### Windows 一键安装

在项目根目录运行：

```powershell
.\scripts\install-windows.bat
```

行为：

- 生成 `config.json`（包含 `allow_from`）
- 创建计划任务 `CodexTelegramBot`（用户登录时自动启动）
- 尝试立即启动任务

卸载：

```powershell
.\scripts\uninstall-windows.bat
```

将删除计划任务，并可选删除本地配置/日志/会话文件。

### Linux/macOS 一键安装

在项目根目录运行：

```bash
chmod +x scripts/install-unix.sh scripts/uninstall-unix.sh
bash scripts/install-unix.sh
```

行为：

- 生成 `config.json`（包含 `allow_from`）
- Linux：安装并启动 `systemd` 服务 `codex-telegram-bot.service`
- macOS：安装并启动 `launchd` Agent `com.codex.telegram.bot`

卸载：

```bash
bash scripts/uninstall-unix.sh
```

将删除系统服务/Agent，并可选删除本地配置/日志/会话文件。

## 日志

Windows：

```powershell
Get-Content .\logs\bot-runtime.log -Wait
```

Linux/macOS：

```bash
tail -f logs/bot-runtime.log
```

## 常见问题

- `Codex binary not found`  
  把 `config.json` 中的 `codex_bin` 设为绝对路径，例如：
  `C:\\Users\\admin\\AppData\\Roaming\\npm\\codex.cmd`

- 回复异常长，像 JSON 事件流  
  已修复为只提取最终 `agent_message`。请重启 `bot.py` 并使用最新版本。

- 收到 `Unauthorized user.`  
  检查 `config.json` 中 `allow_from` 的数字 ID 是否正确。

## 安全建议

- 不要把真实 `telegram_bot_token` 提交到版本库。
- 对外共享前，先替换配置中的密钥信息。

## 最小化生产检查清单

启用开机自启前，请检查：

1. 运行账号
- 使用和 `codex login` 相同的 Windows 账号运行机器人。
- 手动验证：`codex -a never exec --sandbox read-only --skip-git-repo-check -C C:\Users\admin\.openclaw\workspace "reply OK"`。

2. 启动方式
- 建议使用 `run-bot.ps1`，进程崩溃可自动拉起。
- 如果使用任务计划程序，建议：
  - 程序：`powershell.exe`
  - 参数：`-NoProfile -ExecutionPolicy Bypass -File C:\Users\admin\.openclaw\workspace\codex-telegram-bot\run-bot.ps1`
  - 起始目录：`C:\Users\admin\.openclaw\workspace\codex-telegram-bot`

3. 健康检查
- 至少每天查看一次 `logs/bot-runtime.log`，确认轮询持续运行。
- 关注是否反复出现 `Codex failed` 或 `network error`。

4. 日志轮转（每周）
- 控制日志体积，避免无限增长。
- 示例（PowerShell）：将 `logs\bot-runtime.log` 归档到 `logs\archive\bot-runtime-YYYYMMDD.log`，再创建新的空日志文件。

5. 基础安全
- `config.json` 中 `allow_from` 保持最小权限（仅你自己的 ID）。
- 默认使用 `codex_sandbox: read-only`。
- 不要把真实 `telegram_bot_token` 提交到 Git。

6. 备份
- 建议备份：
  - `config.json`
  - `telegram/update-offset-default.json`

7. 故障恢复
- 如果机器人不再响应：
  - 重启进程或脚本。
  - 查看 `logs/bot-runtime.log` 末尾。
  - 用同一账号手工执行一次 `codex exec`，确认认证与网络正常。

## 权限与沙箱

`codex_sandbox` 控制 Codex 的访问范围。

### 1) `read-only`（推荐默认）

适用场景：

- 日常问答、代码阅读、排障、只读分析
- 生产环境优先的安全模式

行为：

- 可读取 `codex_workdir` 下文件
- 不可写入/创建/删除文件
- 不可执行会产生写入副作用的操作

优点：

- 风险最低，不易误改项目文件

限制：

- 无法自动修改代码
- 无法执行需要写权限的命令（例如生成文件、把依赖安装到项目目录）

---

### 2) `workspace-write`

适用场景：

- 需要 Codex 直接修改项目代码
- 需要在项目目录内创建/更新文件

行为：

- 可读写 `codex_workdir` 下文件
- 默认不可访问 `codex_workdir` 外路径
- 需要额外目录时，用 `codex_add_dirs` 显式放行

优点：

- 在可控范围内支持自动改代码，安全性与可用性平衡较好

限制：

- 沙箱外路径仍受限
- 系统级限制（ACL、杀毒软件、组策略）仍可能拦截操作

---

### 3) `danger-full-access`（高风险）

适用场景：

- 明确需要访问系统任意路径或高权限操作
- 仅在你完全信任当前提示词和执行内容时使用

行为：

- 不启用 Codex 沙箱限制
- 以当前 Windows 用户权限执行
- 此模式下 `codex_add_dirs` 基本无意义

风险：

- 误操作代价最高，可能影响项目外文件或系统状态
- 更容易产生不可逆改动

建议：

- 非必要不要开启
- 仅短时启用，完成后恢复到 `read-only` 或 `workspace-write`

示例（`config.json`）：

```json
{
  "codex_sandbox": "workspace-write",
  "codex_add_dirs": [
    "C:\\Users\\admin\\Desktop",
    "C:\\Users\\admin\\Documents"
  ]
}
```

完全访问（高风险）：

```json
{
  "codex_sandbox": "danger-full-access",
  "codex_add_dirs": []
}
```

说明：

- `codex_add_dirs` 仅在非完全访问模式下有效。
- 修改后需要重启机器人。
- 可在 `logs/bot-runtime.log` 中确认配置是否生效：
  - `sandbox=...`
  - `[codex] add_dirs=[...]`
- 即使是完全访问，系统级限制（Windows 策略、ACL、杀毒软件）仍可能阻止命令执行。

## 会话记忆与新会话

机器人支持按 chat/user 持久化上下文，并支持创建新会话。

命令：

- `/new`：创建新会话并清空当前上下文
- `/forget`：清空当前会话历史，但保留会话 ID
- `/session`：显示当前会话信息
- `/sessions`：打开可点击的会话菜单（含分页），点击即可切换
- `/history [N]`：显示当前会话最近 N 条消息（默认 10）
- `/models`：显示可选模型列表，点击即可切换
- `/models add <model-name>`：手动输入并校验模型；通过后加入模型列表

会话存储文件：

- `telegram/session-store.json`

`config.json` 相关配置：

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

说明：

- `/new` 会归档上一个会话，便于后续查看。
- 归档会话不会自动注入到新会话上下文。
- 会话标题基于用户消息自动生成，在 `/sessions` 中显示为 `#session-id title`。
- `session_scope` 可选：
  - `chat`：每个 chat 一个会话（推荐）
  - `user`：同一用户跨 chat 共用一个会话
  - `chat_user`：按 `(chat, user)` 组合区分会话
- 修改会话设置后请重启 `bot.py`。
