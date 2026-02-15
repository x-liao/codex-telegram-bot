#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_NAME="codex-telegram-bot"
MAC_LABEL="com.codex.telegram.bot"
DEFAULT_MODEL="gpt-5.2-codex"
DEFAULT_SANDBOX="read-only"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[ERROR] Missing required command: $1"
    exit 1
  fi
}

escape_sed() {
  printf '%s' "$1" | sed -e 's/[\/&|]/\\&/g'
}

prepend_path_if_missing() {
  local dir="$1"
  local path_value="$2"
  if [[ -z "$dir" ]]; then
    printf '%s' "$path_value"
    return
  fi
  if [[ ":$path_value:" == *":$dir:"* ]]; then
    printf '%s' "$path_value"
    return
  fi
  if [[ -n "$path_value" ]]; then
    printf '%s:%s' "$dir" "$path_value"
    return
  fi
  printf '%s' "$dir"
}

prompt_non_empty() {
  local prompt="$1"
  local value=""
  while [[ -z "$value" ]]; do
    read -r -p "$prompt" value
    if [[ -z "$value" ]]; then
      echo "Value cannot be empty."
    fi
  done
  printf '%s' "$value"
}

validate_sandbox() {
  case "$1" in
    read-only|workspace-write|danger-full-access) return 0 ;;
    *) return 1 ;;
  esac
}

install_linux_systemd() {
  local template="$ROOT_DIR/deploy/systemd/codex-telegram-bot.service.template"
  local target="/etc/systemd/system/${SERVICE_NAME}.service"
  local run_user
  local run_group
  local python_bin
  local codex_bin
  local node_bin
  local path_env
  local tmp_file
  local sudo_cmd=""

  [[ -f "$template" ]] || { echo "[ERROR] Missing template: $template"; exit 1; }
  require_cmd systemctl

  run_user="$(id -un)"
  run_group="$(id -gn)"
  python_bin="$(command -v python3 || true)"
  codex_bin="$(command -v codex || true)"
  node_bin="$(command -v node || true)"
  [[ -n "$python_bin" ]] || { echo "[ERROR] python3 not found in PATH."; exit 1; }
  path_env="${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"
  if [[ -n "$codex_bin" ]]; then
    path_env="$(prepend_path_if_missing "$(dirname "$codex_bin")" "$path_env")"
  fi
  if [[ -n "$node_bin" ]]; then
    path_env="$(prepend_path_if_missing "$(dirname "$node_bin")" "$path_env")"
  fi

  if [[ $EUID -ne 0 ]]; then
    require_cmd sudo
    sudo_cmd="sudo"
  fi

  tmp_file="$(mktemp)"
  sed \
    -e "s|__USER__|$(escape_sed "$run_user")|g" \
    -e "s|__GROUP__|$(escape_sed "$run_group")|g" \
    -e "s|__ROOT_DIR__|$(escape_sed "$ROOT_DIR")|g" \
    -e "s|__PYTHON_BIN__|$(escape_sed "$python_bin")|g" \
    -e "s|__PATH_ENV__|$(escape_sed "$path_env")|g" \
    "$template" > "$tmp_file"

  $sudo_cmd install -m 644 "$tmp_file" "$target"
  rm -f "$tmp_file"

  $sudo_cmd systemctl daemon-reload
  $sudo_cmd systemctl enable --now "${SERVICE_NAME}.service"

  echo "[OK] Installed and started: ${SERVICE_NAME}.service"
  $sudo_cmd systemctl status "${SERVICE_NAME}.service" --no-pager || true
}

install_macos_launchd() {
  local template="$ROOT_DIR/deploy/macos/com.codex.telegram.bot.plist.template"
  local plist_dir="$HOME/Library/LaunchAgents"
  local plist_file="$plist_dir/${MAC_LABEL}.plist"
  local python_bin
  local path_env
  local tmp_file

  [[ -f "$template" ]] || { echo "[ERROR] Missing template: $template"; exit 1; }

  python_bin="$(command -v python3 || true)"
  [[ -n "$python_bin" ]] || { echo "[ERROR] python3 not found in PATH."; exit 1; }

  path_env="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
  mkdir -p "$plist_dir"

  tmp_file="$(mktemp)"
  sed \
    -e "s|__LABEL__|$(escape_sed "$MAC_LABEL")|g" \
    -e "s|__PYTHON_BIN__|$(escape_sed "$python_bin")|g" \
    -e "s|__ROOT_DIR__|$(escape_sed "$ROOT_DIR")|g" \
    -e "s|__PATH_ENV__|$(escape_sed "$path_env")|g" \
    "$template" > "$tmp_file"
  mv "$tmp_file" "$plist_file"

  launchctl bootout "gui/$(id -u)" "$plist_file" 2>/dev/null || true
  launchctl bootstrap "gui/$(id -u)" "$plist_file"
  launchctl kickstart -k "gui/$(id -u)/${MAC_LABEL}"

  echo "[OK] Installed and started: ${MAC_LABEL}"
  launchctl print "gui/$(id -u)/${MAC_LABEL}" | head -n 20 || true
}

main() {
  local os_name
  local telegram_token
  local telegram_id
  local model_input
  local sandbox_input
  local sandbox
  local workdir

  require_cmd python3
  require_cmd codex

  os_name="$(uname -s)"
  if [[ "$os_name" != "Linux" && "$os_name" != "Darwin" ]]; then
    echo "[ERROR] Unsupported OS: $os_name"
    echo "Use scripts/install-windows.bat on Windows."
    exit 1
  fi

  telegram_token="$(prompt_non_empty "Telegram Bot Token: ")"
  telegram_id="$(prompt_non_empty "Allowed Telegram numeric ID: ")"
  if [[ ! "$telegram_id" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] Telegram ID must be numeric."
    exit 1
  fi

  read -r -p "Model (default: ${DEFAULT_MODEL}): " model_input
  model_input="${model_input:-$DEFAULT_MODEL}"

  read -r -p "Sandbox [read-only/workspace-write/danger-full-access] (default: ${DEFAULT_SANDBOX}): " sandbox_input
  sandbox="${sandbox_input:-$DEFAULT_SANDBOX}"
  if ! validate_sandbox "$sandbox"; then
    echo "[ERROR] Invalid sandbox value: $sandbox"
    exit 1
  fi

  workdir="$ROOT_DIR/workdir"
  mkdir -p "$ROOT_DIR/telegram" "$ROOT_DIR/logs" "$workdir"

  cat > "$ROOT_DIR/config.json" <<EOF
{
  "telegram_bot_token": "$telegram_token",
  "allow_from": [
    "$telegram_id"
  ],
  "codex_bin": "codex",
  "codex_workdir": "$workdir",
  "codex_sandbox": "$sandbox",
  "codex_model": "$model_input",
  "codex_timeout_seconds": 120,
  "poll_timeout_seconds": 30,
  "log_file": "logs/bot-runtime.log",
  "log_stdout": true,
  "text_preview_chars": 100,
  "session_enabled": true,
  "session_scope": "chat",
  "session_max_messages": 20,
  "session_archive_max": 10,
  "session_title_max_chars": 32,
  "session_entry_max_chars": 2000,
  "session_system_prompt": "You are a direct and practical Telegram assistant. Handle both normal conversation and coding questions. This is a continuous chat; use prior context when answering the current message.",
  "debug_echo_enabled": false,
  "debug_echo_dir": "logs/codex-debug",
  "debug_echo_max_chars": 0
}
EOF

  if [[ "$os_name" == "Linux" ]]; then
    install_linux_systemd
  else
    install_macos_launchd
  fi

  echo "[OK] Deployment finished."
  echo "Config: $ROOT_DIR/config.json"
  echo "Allowlist key: allow_from (inside config.json)"
  echo "Tip: Ensure this account has completed 'codex login'."
}

main "$@"
