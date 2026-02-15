#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_NAME="codex-telegram-bot"
MAC_LABEL="com.codex.telegram.bot"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[ERROR] Missing required command: $1"
    exit 1
  fi
}

remove_linux_service() {
  local target="/etc/systemd/system/${SERVICE_NAME}.service"
  local sudo_cmd=""

  require_cmd systemctl
  if [[ $EUID -ne 0 ]]; then
    require_cmd sudo
    sudo_cmd="sudo"
  fi

  $sudo_cmd systemctl disable --now "${SERVICE_NAME}.service" >/dev/null 2>&1 || true
  $sudo_cmd rm -f "$target"
  $sudo_cmd systemctl daemon-reload
  echo "[OK] Removed systemd service: ${SERVICE_NAME}.service"
}

remove_macos_agent() {
  local plist_file="$HOME/Library/LaunchAgents/${MAC_LABEL}.plist"
  launchctl bootout "gui/$(id -u)" "$plist_file" 2>/dev/null || true
  rm -f "$plist_file"
  echo "[OK] Removed launchd agent: ${MAC_LABEL}"
}

maybe_remove_local_files() {
  local answer=""
  read -r -p "Remove local generated files (config/session/log)? [y/N]: " answer
  case "${answer,,}" in
    y|yes)
      rm -f "$ROOT_DIR/config.json"
      rm -f "$ROOT_DIR/telegram/update-offset-default.json"
      rm -f "$ROOT_DIR/telegram/session-store.json"
      rm -f "$ROOT_DIR/telegram/bot-instance.lock"
      rm -rf "$ROOT_DIR/logs"
      echo "[OK] Local generated files removed."
      ;;
    *)
      echo "[INFO] Local files kept."
      ;;
  esac
}

main() {
  local os_name

  os_name="$(uname -s)"
  if [[ "$os_name" == "Linux" ]]; then
    remove_linux_service
  elif [[ "$os_name" == "Darwin" ]]; then
    remove_macos_agent
  else
    echo "[ERROR] Unsupported OS: $os_name"
    exit 1
  fi

  maybe_remove_local_files
  echo "[OK] Uninstall finished."
}

main "$@"
