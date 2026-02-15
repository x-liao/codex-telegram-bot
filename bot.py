import json
import os
import re
import shutil
import subprocess
import threading
import time
import traceback
import atexit
import mimetypes
import uuid
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent
STATE_FILE = ROOT_DIR / "telegram" / "update-offset-default.json"
SESSION_STORE_FILE = ROOT_DIR / "telegram" / "session-store.json"
INSTANCE_LOCK_FILE = ROOT_DIR / "telegram" / "bot-instance.lock"
CONFIG_FILE = ROOT_DIR / "config.json"
DEFAULT_LOG_FILE = ROOT_DIR / "logs" / "bot-runtime.log"

MAX_TG_MESSAGE_LEN = 3900
HTTP_TIMEOUT_SECONDS = 45
LOG_FILE = DEFAULT_LOG_FILE
LOG_STDOUT = True
TEXT_PREVIEW_CHARS = 100
SESSION_ENABLED = True
SESSION_SCOPE = "chat"
SESSION_MAX_MESSAGES = 20
SESSION_ENTRY_MAX_CHARS = 2000
SESSION_ARCHIVE_MAX = 10
SESSION_TITLE_MAX_CHARS = 32
SESSIONS_PAGE_SIZE = 8
MODELS_PAGE_SIZE = 8
SESSION_SYSTEM_PROMPT = (
    "You are a direct and practical Telegram assistant. "
    "Handle both normal conversation and coding questions. "
    "This is a continuous chat; use prior context when answering the current message."
)
META_NOISE_TOKENS = [
    "AGENTS.md",
    "BOOTSTRAP.md",
    "SOUL.md",
    "USER.md",
    "MEMORY.md",
    "memory/",
    "agents instructions",
    "follow the agents instructions",
    "use skills when they're relevant",
    "use skills when they are relevant",
]
ANTI_META_PROMPT = (
    "Answer only the user's actual request directly.\n"
    "Do NOT perform bootstrap checks.\n"
    "Do NOT mention or search AGENTS.md, BOOTSTRAP.md, SOUL.md, USER.md, MEMORY.md "
    "unless the user explicitly asks about these files."
)
GENERIC_NONANSWER_TOKENS = [
    "ready when you are",
    "ready when you're ready",
    "what would you like to do",
    "what would you like me to do",
    "how do you want to proceed",
    "how would you like to proceed",
    "understood. ready",
    "got it. what would you like",
    "what do you want to work on",
    "ready. what do you want to work on",
    "what do you want to work on in",
    "please provide the task",
    "tell me what to do",
    "no actionable request provided",
    "no actionable request",
    "there is no actionable request",
    "please provide an actionable request",
    "please provide a clear request",
    "please provide a specific request",
    "please provide the specific question you want answered",
    "i don't have any prior user question to answer",
    "i do not have any prior user question to answer",
    "no request provided",
    "i need a clear request",
    "i need a specific request",
    "follow the agents instructions",
    "use skills when they're relevant",
    "use skills when they are relevant",
]
SESSION_MEDIA_KINDS = {"image", "document", "audio", "voice", "video"}

INSTANCE_LOCK_HELD = False
DEBUG_ECHO_SEQ = 0
MODEL_INPUT_PENDING: dict[str, int] = {}
LOGIN_SHELL_PATH_CACHE: str | None = None
_TRANSCRIBE_MODEL_CACHE: Any = None
_TRANSCRIBE_MODEL_CACHE_KEY: str = ""
_TRANSCRIBE_MODEL_LOCK = threading.Lock()


class TelegramApiError(RuntimeError):
    def __init__(self, message: str, payload: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.payload = payload or {}


class TelegramConflictError(TelegramApiError):
    pass


def raise_telegram_api_error(data: Any, cause: Exception | None = None) -> None:
    payload = data if isinstance(data, dict) else {}
    error_code = payload.get("error_code")
    error_cls: type[TelegramApiError] = TelegramApiError
    if str(error_code) == "409":
        error_cls = TelegramConflictError
    err = error_cls(f"Telegram API error: {data}", payload=payload)
    if cause is not None:
        raise err from cause
    raise err


def log(msg: str, level: str = "INFO") -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"{now} [{level}] {msg}"
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:  # noqa: BLE001
        pass
    if LOG_STDOUT:
        print(line, flush=True)


def load_config_file() -> dict[str, Any]:
    if not CONFIG_FILE.exists():
        return {}
    try:
        raw = json.loads(CONFIG_FILE.read_text(encoding="utf-8-sig"))
    except Exception as exc:  # noqa: BLE001
        log(f"[warn] config parse failed: {exc}")
        return {}
    if not isinstance(raw, dict):
        log("[warn] config.json root must be an object.")
        return {}
    return raw


def cfg_text(config: dict[str, Any], env_key: str, cfg_key: str, default: str) -> str:
    env_val = os.environ.get(env_key)
    if env_val is not None and env_val.strip():
        return env_val.strip()
    cfg_val = config.get(cfg_key)
    if cfg_val is None:
        return default
    text = str(cfg_val).strip()
    return text or default


def cfg_int(config: dict[str, Any], env_key: str, cfg_key: str, default: int) -> int:
    env_val = os.environ.get(env_key)
    if env_val is not None and env_val.strip():
        try:
            return int(env_val.strip())
        except ValueError:
            log(f"[warn] invalid int in env {env_key}: {env_val!r}; fallback={default}")
            return default

    cfg_val = config.get(cfg_key)
    if cfg_val is None:
        return default
    try:
        return int(cfg_val)
    except (TypeError, ValueError):
        log(f"[warn] invalid int in config {cfg_key}: {cfg_val!r}; fallback={default}")
        return default


def cfg_bool(config: dict[str, Any], env_key: str, cfg_key: str, default: bool) -> bool:
    env_val = os.environ.get(env_key)
    if env_val is not None and env_val.strip():
        v = env_val.strip().lower()
        if v in {"1", "true", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "no", "n", "off"}:
            return False
        log(f"[warn] invalid bool in env {env_key}: {env_val!r}; fallback={default}")
        return default

    cfg_val = config.get(cfg_key)
    if cfg_val is None:
        return default
    if isinstance(cfg_val, bool):
        return cfg_val
    if isinstance(cfg_val, str):
        v = cfg_val.strip().lower()
        if v in {"1", "true", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "no", "n", "off"}:
            return False
    log(f"[warn] invalid bool in config {cfg_key}: {cfg_val!r}; fallback={default}")
    return default


def cfg_list_str(
    config: dict[str, Any], env_key: str, cfg_key: str, default: list[str]
) -> list[str]:
    env_val = os.environ.get(env_key)
    if env_val is not None and env_val.strip():
        parts = [p.strip() for p in env_val.replace(";", ",").split(",")]
        return [p for p in parts if p]

    cfg_val = config.get(cfg_key)
    if cfg_val is None:
        return list(default)
    if isinstance(cfg_val, list):
        out: list[str] = []
        for item in cfg_val:
            s = str(item).strip()
            if s:
                out.append(s)
        return out
    if isinstance(cfg_val, str):
        parts = [p.strip() for p in cfg_val.replace(";", ",").split(",")]
        return [p for p in parts if p]

    log(f"[warn] invalid list in config {cfg_key}: {cfg_val!r}; fallback={default!r}")
    return list(default)


def cfg_choice(
    config: dict[str, Any],
    env_key: str,
    cfg_key: str,
    default: str,
    allowed: set[str],
) -> str:
    val = cfg_text(config, env_key, cfg_key, default).lower()
    if val in allowed:
        return val
    log(
        f"[warn] invalid choice for {cfg_key}: {val!r}; "
        f"allowed={sorted(allowed)} fallback={default!r}"
    )
    return default


def resolve_log_file(path_text: str) -> Path:
    if not path_text.strip():
        return DEFAULT_LOG_FILE
    p = Path(path_text.strip())
    if not p.is_absolute():
        p = ROOT_DIR / p
    return p


def resolve_config_path(path_text: str, default_rel: str) -> Path:
    text = (path_text or "").strip() or default_rel
    p = Path(text)
    if not p.is_absolute():
        p = ROOT_DIR / p
    return p


def preview_text(text: str, limit: int) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit] + "..."


def telegram_command_name(text: str) -> str:
    token = str(text or "").strip().split(maxsplit=1)[0].lower() if str(text or "").strip() else ""
    if not token.startswith("/"):
        return ""
    cmd = token[1:]
    if "@" in cmd:
        cmd = cmd.split("@", 1)[0]
    return cmd


def normalize_match_text(text: str) -> str:
    # Normalize punctuation variants so simple token match is stable.
    low = text.lower()
    low = (
        low.replace("’", "'")
        .replace("‘", "'")
        .replace("`", "'")
        .replace("“", '"')
        .replace("”", '"')
    )
    return " ".join(low.split())


def contains_meta_noise(text: str) -> bool:
    low = normalize_match_text(text)
    for token in META_NOISE_TOKENS:
        if token.lower() in low:
            return True
    if "agents" in low and "skills" in low and "work on" in low:
        return True
    return False


def contains_generic_nonanswer(text: str) -> bool:
    low = normalize_match_text(text)
    for token in GENERIC_NONANSWER_TOKENS:
        if token in low:
            return True
    if re.search(r"\bno\s+actionable\s+request\b", low):
        return True
    if re.search(r"\bno\s+request\s+provided\b", low):
        return True
    if re.search(r"\bprovide\s+(a|an)\s+(clear|specific|actionable)\s+request\b", low):
        return True
    if (
        len(low.strip()) <= 220
        and (
            "what would you like to do in" in low
            or "what do you want to work on in" in low
            or "what would you like me to do in" in low
        )
        and ("'" in low or ":" in low or "\\" in low or "/" in low)
    ):
        return True
    if (
        len(low.strip()) <= 220
        and re.search(r"\b(what|which)\s+do\s+you\s+want\s+to\s+work\s+on\b", low)
    ):
        return True
    if (
        len(low.strip()) <= 220
        and re.search(r"\bwhat\s+would\s+you\s+like\s+me?\s+to\s+do\b", low)
    ):
        return True
    if (
        len(low.strip()) <= 220
        and "work on" in low
        and re.search(r"\b(ready|understood|got it|what)\b", low)
    ):
        return True
    if (
        len(low.strip()) <= 220
        and "understood" in low
        and "how can i help" in low
    ):
        return True
    if "i'll be direct and practical" in low and "continuous chat" in low:
        return True
    if len(low.strip()) <= 64 and re.search(r"\b(understood|got it|ok|okay)\b", low):
        return True
    return False


def contains_cjk(text: str) -> bool:
    return re.search(r"[\u4e00-\u9fff]", text) is not None


CONFIG = load_config_file()
BOT_TOKEN = cfg_text(CONFIG, "TELEGRAM_BOT_TOKEN", "telegram_bot_token", "")
CODEX_BIN = cfg_text(CONFIG, "CODEX_BIN", "codex_bin", "codex")
CODEX_WORKDIR = cfg_text(
    CONFIG, "CODEX_WORKDIR", "codex_workdir", r"C:\Users\admin\.openclaw\workspace"
)
CODEX_SANDBOX = cfg_text(CONFIG, "CODEX_SANDBOX", "codex_sandbox", "read-only")
CODEX_MODEL = cfg_text(CONFIG, "CODEX_MODEL", "codex_model", "")
CODEX_ADD_DIRS = cfg_list_str(CONFIG, "CODEX_ADD_DIRS", "codex_add_dirs", [])
CODEX_TIMEOUT_SECONDS = max(
    15, cfg_int(CONFIG, "CODEX_TIMEOUT_SECONDS", "codex_timeout_seconds", 120)
)
POLL_TIMEOUT_SECONDS = cfg_int(CONFIG, "POLL_TIMEOUT_SECONDS", "poll_timeout_seconds", 30)
LOG_STDOUT = cfg_bool(CONFIG, "LOG_STDOUT", "log_stdout", True)
TEXT_PREVIEW_CHARS = cfg_int(CONFIG, "TEXT_PREVIEW_CHARS", "text_preview_chars", 100)
LOG_FILE = resolve_log_file(cfg_text(CONFIG, "LOG_FILE", "log_file", str(DEFAULT_LOG_FILE)))
SESSION_ENABLED = cfg_bool(CONFIG, "SESSION_ENABLED", "session_enabled", True)
SESSION_SCOPE = cfg_choice(
    CONFIG, "SESSION_SCOPE", "session_scope", "chat", {"chat", "user", "chat_user"}
)
SESSION_MAX_MESSAGES = cfg_int(CONFIG, "SESSION_MAX_MESSAGES", "session_max_messages", 20)
SESSION_ENTRY_MAX_CHARS = cfg_int(
    CONFIG, "SESSION_ENTRY_MAX_CHARS", "session_entry_max_chars", 2000
)
SESSION_ARCHIVE_MAX = cfg_int(CONFIG, "SESSION_ARCHIVE_MAX", "session_archive_max", 10)
SESSION_TITLE_MAX_CHARS = cfg_int(
    CONFIG, "SESSION_TITLE_MAX_CHARS", "session_title_max_chars", 32
)
SESSION_SYSTEM_PROMPT = cfg_text(
    CONFIG,
    "SESSION_SYSTEM_PROMPT",
    "session_system_prompt",
    SESSION_SYSTEM_PROMPT,
)
DEBUG_ECHO_ENABLED = cfg_bool(CONFIG, "DEBUG_ECHO_ENABLED", "debug_echo_enabled", False)
DEBUG_ECHO_DIR = resolve_config_path(
    cfg_text(CONFIG, "DEBUG_ECHO_DIR", "debug_echo_dir", "logs/codex-debug"),
    "logs/codex-debug",
)
DEBUG_ECHO_MAX_CHARS = cfg_int(
    CONFIG, "DEBUG_ECHO_MAX_CHARS", "debug_echo_max_chars", 0
)
TYPING_HEARTBEAT_SECONDS = max(
    1, cfg_int(CONFIG, "TYPING_HEARTBEAT_SECONDS", "typing_heartbeat_seconds", 4)
)

FILE_SEND_ENABLED = cfg_bool(CONFIG, "FILE_SEND_ENABLED", "file_send_enabled", True)
_DEFAULT_FILE_SEND_DIRS: list[str] = list(CODEX_ADD_DIRS)
if CODEX_WORKDIR:
    _DEFAULT_FILE_SEND_DIRS.append(CODEX_WORKDIR)
_DEFAULT_FILE_SEND_DIRS.append("workdir")
_DEFAULT_FILE_SEND_DIRS.append("telegram/uploads")
_DEFAULT_FILE_SEND_DIRS.append(str(Path.home() / "Desktop"))
FILE_SEND_DIRS = cfg_list_str(CONFIG, "FILE_SEND_DIRS", "file_send_dirs", _DEFAULT_FILE_SEND_DIRS)
FILE_SEND_MAX_BYTES = cfg_int(
    CONFIG, "FILE_SEND_MAX_BYTES", "file_send_max_bytes", 50 * 1024 * 1024
)
FILE_SEND_DIR_PATHS: list[Path] = [resolve_config_path(p, "workdir") for p in FILE_SEND_DIRS]

MEDIA_INPUT_ENABLED = cfg_bool(CONFIG, "MEDIA_INPUT_ENABLED", "media_input_enabled", True)
MEDIA_INPUT_DIR = resolve_config_path(
    cfg_text(CONFIG, "MEDIA_INPUT_DIR", "media_input_dir", "telegram/uploads"),
    "telegram/uploads",
)
MEDIA_INPUT_MAX_BYTES = max(
    1, cfg_int(CONFIG, "MEDIA_INPUT_MAX_BYTES", "media_input_max_bytes", 20 * 1024 * 1024)
)
MEDIA_INPUT_MAX_FILES = max(
    1, min(10, cfg_int(CONFIG, "MEDIA_INPUT_MAX_FILES", "media_input_max_files", 4))
)
MEDIA_HISTORY_MAX_ITEMS = max(
    1, min(10, cfg_int(CONFIG, "MEDIA_HISTORY_MAX_ITEMS", "media_history_max_items", 4))
)
LOCAL_TRANSCRIBE_ENABLED = cfg_bool(
    CONFIG, "LOCAL_TRANSCRIBE_ENABLED", "local_transcribe_enabled", True
)
LOCAL_TRANSCRIBE_MODEL = cfg_text(
    CONFIG, "LOCAL_TRANSCRIBE_MODEL", "local_transcribe_model", "tiny"
)
LOCAL_TRANSCRIBE_LANGUAGE = cfg_text(
    CONFIG, "LOCAL_TRANSCRIBE_LANGUAGE", "local_transcribe_language", "zh"
)
LOCAL_TRANSCRIBE_DEVICE = cfg_text(
    CONFIG, "LOCAL_TRANSCRIBE_DEVICE", "local_transcribe_device", "cpu"
)
LOCAL_TRANSCRIBE_COMPUTE_TYPE = cfg_text(
    CONFIG, "LOCAL_TRANSCRIBE_COMPUTE_TYPE", "local_transcribe_compute_type", "int8"
)
LOCAL_TRANSCRIBE_VAD = cfg_bool(
    CONFIG, "LOCAL_TRANSCRIBE_VAD", "local_transcribe_vad", True
)
LOCAL_TRANSCRIBE_MAX_CHARS = max(
    80, cfg_int(CONFIG, "LOCAL_TRANSCRIBE_MAX_CHARS", "local_transcribe_max_chars", 800)
)


def resolve_codex_bin(bin_name: str) -> str:
    normalized = (bin_name or "").strip()
    if not normalized:
        return "codex"

    expanded = Path(normalized).expanduser()

    # 1) Exact path configured
    if expanded.exists() and os.access(expanded, os.X_OK):
        return str(expanded)

    # 2) Resolve from PATH
    resolved = shutil.which(normalized)
    if resolved:
        return resolved

    # 3) Windows npm global fallback
    if os.name == "nt":
        candidates: list[Path] = []
        appdata = os.environ.get("APPDATA", "").strip()
        userprofile = os.environ.get("USERPROFILE", "").strip()
        if appdata:
            candidates.append(Path(appdata) / "npm" / "codex.cmd")
            candidates.append(Path(appdata) / "npm" / "codex.ps1")
            candidates.append(Path(appdata) / "npm" / "codex")
        if userprofile:
            base = Path(userprofile) / "AppData" / "Roaming" / "npm"
            candidates.append(base / "codex.cmd")
            candidates.append(base / "codex.ps1")
            candidates.append(base / "codex")

        for c in candidates:
            if c.exists() and os.access(c, os.X_OK):
                return str(c)
        return normalized

    return normalized


def merge_path_values(*values: str) -> str:
    seen: set[str] = set()
    items: list[str] = []
    for raw in values:
        if not raw:
            continue
        for part in raw.split(os.pathsep):
            p = part.strip()
            if not p or p in seen:
                continue
            seen.add(p)
            items.append(p)
    return os.pathsep.join(items)


def detect_login_shell_path() -> str:
    global LOGIN_SHELL_PATH_CACHE
    if LOGIN_SHELL_PATH_CACHE is not None:
        return LOGIN_SHELL_PATH_CACHE

    if os.name == "nt":
        LOGIN_SHELL_PATH_CACHE = ""
        return ""

    cmd = "printf '%s' \"$PATH\""
    candidates: list[list[str]] = []
    shell = os.environ.get("SHELL", "").strip()
    if shell:
        candidates.append([shell, "-lc", cmd])
    candidates.append(["bash", "-lc", cmd])
    candidates.append(["sh", "-lc", cmd])

    seen: set[tuple[str, str]] = set()
    for args in candidates:
        key = (args[0], args[1])
        if key in seen:
            continue
        seen.add(key)
        try:
            proc = subprocess.run(
                args,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=2,
            )
        except Exception:
            continue
        if proc.returncode != 0:
            continue
        value = (proc.stdout or "").strip()
        if value:
            LOGIN_SHELL_PATH_CACHE = value
            return value

    LOGIN_SHELL_PATH_CACHE = ""
    return ""


def resolve_codex_runtime(bin_name: str) -> tuple[str, str]:
    normalized = (bin_name or "").strip() or "codex"
    base_path = os.environ.get("PATH", "")
    shell_path = detect_login_shell_path()
    merged_path = merge_path_values(base_path, shell_path, os.defpath)

    resolved = resolve_codex_bin(normalized)
    if resolved != normalized:
        return resolved, merged_path

    if merged_path:
        fallback = shutil.which(normalized, path=merged_path)
        if fallback:
            return fallback, merged_path

    return normalized, merged_path


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def now_ts() -> int:
    return int(time.time())


def safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_model_list(raw: Any) -> list[str]:
    if raw is None:
        return []

    items: list[str]
    if isinstance(raw, list):
        items = [str(x) for x in raw]
    elif isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        items = re.split(r"[,;\n\r]+", text)
    else:
        items = [str(raw)]

    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        model = str(item).strip()
        if not model:
            continue
        key = model.lower()
        if key in {"default", "(default)", "codex-default"}:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(model)
    return out


def build_model_input_key(chat_id: int, user_id: str) -> str:
    return f"{chat_id}:{user_id}"


def is_invalid_model_error(text: str) -> bool:
    low = normalize_match_text(text)
    if "model" not in low:
        return False
    tokens = [
        "unknown model",
        "invalid model",
        "model not found",
        "unsupported model",
        "model does not exist",
        "not a valid model",
        "is not available",
        "unsupported value for '--model'",
        "unsupported value for --model",
    ]
    return any(token in low for token in tokens)


def validate_codex_model(model: str) -> tuple[bool, str]:
    target = str(model or "").strip()
    if not target:
        return False, "模型名不能为空。"

    codex_bin_runtime, runtime_path = resolve_codex_runtime(CODEX_BIN)
    cmd = [
        codex_bin_runtime,
        "-a",
        "never",
        "exec",
        "--ephemeral",
        "--json",
        "--color",
        "never",
        "--sandbox",
        CODEX_SANDBOX,
        "--skip-git-repo-check",
        "-C",
        CODEX_WORKDIR,
        "-m",
        target,
    ]
    for add_dir in CODEX_ADD_DIRS:
        cmd.extend(["--add-dir", add_dir])
    cmd.append("-")

    probe_prompt = "Reply exactly: OK"
    timeout_seconds = max(20, min(120, max(30, CODEX_TIMEOUT_SECONDS)))
    run_env = os.environ.copy()
    if runtime_path:
        run_env["PATH"] = runtime_path

    try:
        proc = subprocess.run(
            cmd,
            input=probe_prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=timeout_seconds,
            env=run_env,
        )
    except FileNotFoundError:
        return False, f"Codex 二进制不存在：{codex_bin_runtime}"
    except subprocess.TimeoutExpired:
        return False, f"验证超时（>{timeout_seconds}s），请稍后重试。"
    except Exception as exc:  # noqa: BLE001
        return False, f"验证执行失败：{exc}"

    stdout_text = proc.stdout or ""
    stderr_text = proc.stderr or ""
    reply, errors = parse_codex_output(stdout_text)
    if proc.returncode == 0 and reply:
        return True, ""

    details: list[str] = []
    details.extend([e.strip() for e in errors if isinstance(e, str) and e.strip()])
    if stderr_text.strip():
        details.append(stderr_text.strip())
    if not details:
        details.append(f"exit={proc.returncode}")
    detail = details[0][:400]

    if is_invalid_model_error("\n".join(details)):
        return False, f"模型不可用：{detail}"
    return False, f"验证失败（可能是网络或鉴权问题）：{detail}"


def add_model_to_registry(model: str) -> tuple[bool, bool, str]:
    target = str(model or "").strip()
    if not target:
        return False, False, "模型名不能为空。"

    live_cfg = load_config_file()
    models = parse_model_list(live_cfg.get("codex_models"))
    existed = any(target.lower() == item.lower() for item in models)
    if not existed:
        models.append(target)

    ok, err = write_config_updates({"codex_models": models})
    if not ok:
        return False, False, f"写入配置失败：{err}"

    CONFIG["codex_models"] = models
    return True, (not existed), ""


def verify_and_register_model(model: str) -> tuple[bool, str]:
    target = str(model or "").strip()
    ok, detail = validate_codex_model(target)
    if not ok:
        return False, detail

    save_ok, added, save_detail = add_model_to_registry(target)
    if not save_ok:
        return False, f"模型验证通过，但{save_detail}"

    if added:
        return True, f"模型验证通过，已加入列表：{target}"
    return True, f"模型验证通过，模型已在列表：{target}"


def load_models_from_config() -> list[str]:
    live_cfg = load_config_file()
    return parse_model_list(live_cfg.get("codex_models"))


def write_config_updates(updates: dict[str, Any]) -> tuple[bool, str]:
    cfg = load_config_file()
    if not isinstance(cfg, dict):
        cfg = {}
    cfg.update(updates)
    try:
        ensure_parent(CONFIG_FILE)
        tmp = CONFIG_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(CONFIG_FILE)
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)
    return True, ""


def set_active_codex_model(model: str) -> tuple[bool, str]:
    global CODEX_MODEL, CONFIG
    target = str(model or "").strip()
    CODEX_MODEL = target
    CONFIG["codex_model"] = target

    ok, err = write_config_updates({"codex_model": target})
    note = ""
    env_override = str(os.environ.get("CODEX_MODEL", "")).strip()
    if env_override:
        note = "检测到环境变量 CODEX_MODEL，重启后可能被环境变量覆盖。"

    if not ok:
        if note:
            return False, f"配置写入失败：{err}；{note}"
        return False, f"配置写入失败：{err}"
    return True, note


def clip_debug_text(text: str) -> str:
    if DEBUG_ECHO_MAX_CHARS <= 0:
        return text
    if len(text) <= DEBUG_ECHO_MAX_CHARS:
        return text
    return text[:DEBUG_ECHO_MAX_CHARS] + "\n...[truncated]..."


def write_codex_debug_echo(payload: dict[str, Any]) -> None:
    if not DEBUG_ECHO_ENABLED:
        return
    global DEBUG_ECHO_SEQ
    DEBUG_ECHO_SEQ += 1
    stamp = time.strftime("%Y%m%d-%H%M%S")
    ms = int((time.time() % 1) * 1000)
    filename = f"{stamp}-{ms:03d}-pid{os.getpid()}-{DEBUG_ECHO_SEQ:04d}.json"
    path = DEBUG_ECHO_DIR / filename
    try:
        DEBUG_ECHO_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"[debug] codex echo saved: {path}")
    except Exception as exc:  # noqa: BLE001
        log(f"[warn] failed to write codex debug echo: {exc}")


def pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def release_instance_lock() -> None:
    global INSTANCE_LOCK_HELD
    if not INSTANCE_LOCK_HELD:
        return
    try:
        if INSTANCE_LOCK_FILE.exists():
            owner_pid = -1
            try:
                raw = json.loads(INSTANCE_LOCK_FILE.read_text(encoding="utf-8-sig"))
                owner_pid = int(raw.get("pid", -1)) if isinstance(raw, dict) else -1
            except Exception:
                owner_pid = -1
            if owner_pid in {-1, os.getpid()}:
                INSTANCE_LOCK_FILE.unlink(missing_ok=True)
    except Exception:
        pass
    INSTANCE_LOCK_HELD = False


def acquire_instance_lock() -> None:
    global INSTANCE_LOCK_HELD
    ensure_parent(INSTANCE_LOCK_FILE)
    current_pid = os.getpid()

    if INSTANCE_LOCK_FILE.exists():
        stale = True
        other_pid = -1
        try:
            raw = json.loads(INSTANCE_LOCK_FILE.read_text(encoding="utf-8-sig"))
            if isinstance(raw, dict):
                other_pid = int(raw.get("pid", -1))
            if other_pid > 0 and pid_is_running(other_pid):
                stale = False
        except Exception:
            stale = False

        if not stale:
            raise SystemExit(
                f"Another bot process is already running (pid={other_pid}). "
                "Stop the old process first."
            )
        INSTANCE_LOCK_FILE.unlink(missing_ok=True)

    payload = {"pid": current_pid, "started_at": now_ts()}
    with INSTANCE_LOCK_FILE.open("x", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False))
    INSTANCE_LOCK_HELD = True
    atexit.register(release_instance_lock)


def load_session_store() -> dict[str, Any]:
    if not SESSION_STORE_FILE.exists():
        return {"version": 1, "sessions": {}}
    try:
        data = json.loads(SESSION_STORE_FILE.read_text(encoding="utf-8-sig"))
    except Exception as exc:  # noqa: BLE001
        log(f"[warn] session store parse failed: {exc}")
        return {"version": 1, "sessions": {}}
    if not isinstance(data, dict):
        return {"version": 1, "sessions": {}}
    sessions = data.get("sessions")
    if not isinstance(sessions, dict):
        sessions = {}
    return {"version": 1, "sessions": sessions}


def save_session_store() -> None:
    ensure_parent(SESSION_STORE_FILE)
    tmp = SESSION_STORE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(SESSION_STORE, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(SESSION_STORE_FILE)


def build_session_key(chat_id: int, user_id: str) -> str:
    if SESSION_SCOPE == "user":
        return f"user:{user_id}"
    if SESSION_SCOPE == "chat_user":
        return f"chat:{chat_id}:user:{user_id}"
    return f"chat:{chat_id}"


def normalize_session_title(text: str) -> str:
    cleaned = " ".join(str(text or "").strip().split())
    if not cleaned:
        return ""
    max_chars = max(12, SESSION_TITLE_MAX_CHARS)
    if len(cleaned) <= max_chars:
        return cleaned
    if max_chars <= 3:
        return cleaned[:max_chars]
    return cleaned[: max_chars - 3] + "..."


def sanitize_session_media(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind", "")).strip().lower()
        if kind not in SESSION_MEDIA_KINDS:
            continue
        entry: dict[str, Any] = {"kind": kind}
        file_name = str(item.get("file_name", "")).strip()
        if file_name:
            entry["file_name"] = clip_text(file_name, 120)
        mime_type = str(item.get("mime_type", "")).strip()
        if mime_type:
            entry["mime_type"] = clip_text(mime_type, 80)
        local_path = str(item.get("local_path", "")).strip()
        if local_path:
            entry["local_path"] = clip_text(local_path.replace("\\", "/"), 240)
        file_id = str(item.get("file_id", "")).strip()
        if file_id:
            entry["file_id"] = clip_text(file_id, 120)
        file_size = safe_int(item.get("file_size"), 0)
        if file_size > 0:
            entry["file_size"] = file_size
        duration = safe_int(item.get("duration"), 0)
        if duration > 0:
            entry["duration"] = duration
        width = safe_int(item.get("width"), 0)
        height = safe_int(item.get("height"), 0)
        if width > 0:
            entry["width"] = width
        if height > 0:
            entry["height"] = height
        transcript = str(item.get("transcript", "")).strip()
        if transcript:
            entry["transcript"] = clip_text(transcript, LOCAL_TRANSCRIBE_MAX_CHARS)
        out.append(entry)
        if len(out) >= MEDIA_HISTORY_MAX_ITEMS:
            break
    return out


def describe_media_item(item: dict[str, Any]) -> str:
    kind = str(item.get("kind", "file")).strip().lower() or "file"
    parts: list[str] = [kind]
    name = str(item.get("file_name", "")).strip()
    if name:
        parts.append(f"name={name}")
    size = safe_int(item.get("file_size"), 0)
    if size > 0:
        parts.append(f"size={size}")
    mime_type = str(item.get("mime_type", "")).strip()
    if mime_type:
        parts.append(f"mime={mime_type}")
    local_path = str(item.get("local_path", "")).strip()
    if local_path:
        normalized_local_path = local_path.replace("\\", "/")
        parts.append(f"path={normalized_local_path}")
    duration = safe_int(item.get("duration"), 0)
    if duration > 0:
        parts.append(f"duration={duration}s")
    width = safe_int(item.get("width"), 0)
    height = safe_int(item.get("height"), 0)
    if width > 0 and height > 0:
        parts.append(f"resolution={width}x{height}")
    transcript = str(item.get("transcript", "")).strip()
    if transcript:
        parts.append(f"transcript={clip_text(transcript, 80)}")
    return ", ".join(parts)


def describe_media_list(items: list[dict[str, Any]]) -> str:
    if not items:
        return ""
    return "; ".join(describe_media_item(item) for item in items if isinstance(item, dict))


def compose_session_entry_text(item: dict[str, Any]) -> str:
    text = str(item.get("text", "")).strip()
    media_items = sanitize_session_media(item.get("media", []))
    media_desc = describe_media_list(media_items)
    if text and media_desc:
        return f"{text}\n[media] {media_desc}"
    if media_desc:
        return f"[media] {media_desc}"
    return text


def build_session_title(messages: list[dict[str, Any]], session_no: int | None = None) -> str:
    fragments: list[str] = []
    frag_limit = max(8, max(12, SESSION_TITLE_MAX_CHARS) // 2)
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        if role != "user":
            continue
        raw_text = str(item.get("text", "")).strip()
        if not raw_text:
            media_desc = describe_media_list(sanitize_session_media(item.get("media", [])))
            raw_text = f"[media] {media_desc}" if media_desc else ""
        if not raw_text or raw_text.startswith("/"):
            continue
        frag = normalize_session_title(preview_text(raw_text, frag_limit))
        if not frag:
            continue
        if frag in fragments:
            continue
        fragments.append(frag)
        if len(fragments) >= 2:
            break

    if fragments:
        if len(fragments) == 1:
            return normalize_session_title(fragments[0])
        return normalize_session_title(f"{fragments[0]} / {fragments[1]}")

    no = max(1, safe_int(session_no, 1))
    return f"会话 #{no}"


def get_session_title(session: dict[str, Any], fallback_no: int | None = None) -> str:
    existing = normalize_session_title(str(session.get("title", "")))
    if existing:
        return existing
    no = max(1, safe_int(session.get("session_no"), fallback_no if fallback_no else 1))
    messages = sanitize_session_messages(session.get("messages", []))
    return build_session_title(messages, session_no=no)


def max_session_no(session: dict[str, Any]) -> int:
    value = max(1, safe_int(session.get("session_no"), 1))
    archived = session.get("archived_sessions", [])
    if not isinstance(archived, list):
        return value
    for item in archived:
        if not isinstance(item, dict):
            continue
        value = max(value, max(1, safe_int(item.get("session_no"), 1)))
    return value


def sanitize_session_messages(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        text = str(item.get("text", "")).strip()
        media_items = sanitize_session_media(item.get("media", []))
        if role not in {"user", "assistant"} or (not text and not media_items):
            continue
        entry: dict[str, Any] = {
            "role": role,
            "text": clip_text(text, max(200, SESSION_ENTRY_MAX_CHARS)),
            "ts": safe_int(item.get("ts"), now_ts()),
        }
        if media_items:
            entry["media"] = media_items
        out.append(entry)
    max_items = max(2, SESSION_MAX_MESSAGES)
    if len(out) > max_items:
        out = out[-max_items:]
    return out


def sanitize_archived_sessions(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        messages = sanitize_session_messages(item.get("messages", []))
        if not messages:
            continue
        created_at = safe_int(item.get("created_at"), now_ts())
        updated_at = safe_int(item.get("updated_at"), created_at)
        archived_at = safe_int(item.get("archived_at"), updated_at)
        out.append(
            {
                "session_no": max(1, safe_int(item.get("session_no"), 1)),
                "created_at": created_at,
                "updated_at": updated_at,
                "archived_at": archived_at,
                "reason": str(item.get("reason", "new")).strip() or "new",
                "title": normalize_session_title(str(item.get("title", "")))
                or build_session_title(messages, session_no=safe_int(item.get("session_no"), 1)),
                "messages": messages,
            }
        )
    max_archives = max(1, SESSION_ARCHIVE_MAX)
    if len(out) > max_archives:
        out = out[-max_archives:]
    return out


def normalize_session_item(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        ts = now_ts()
        return {
            "session_no": 1,
            "created_at": ts,
            "updated_at": ts,
            "title": "会话 #1",
            "messages": [],
            "archived_sessions": [],
        }

    created_at = safe_int(raw.get("created_at"), now_ts())
    updated_at = safe_int(raw.get("updated_at"), created_at)
    session_no = max(1, safe_int(raw.get("session_no"), 1))
    messages = sanitize_session_messages(raw.get("messages", []))
    title = normalize_session_title(str(raw.get("title", ""))) or build_session_title(
        messages, session_no=session_no
    )
    return {
        "session_no": session_no,
        "created_at": created_at,
        "updated_at": updated_at,
        "title": title,
        "messages": messages,
        "archived_sessions": sanitize_archived_sessions(raw.get("archived_sessions", [])),
    }


def latest_archived_session(session: dict[str, Any]) -> dict[str, Any] | None:
    archived = session.get("archived_sessions", [])
    if not isinstance(archived, list):
        return None
    for item in reversed(archived):
        if not isinstance(item, dict):
            continue
        messages = item.get("messages", [])
        if isinstance(messages, list) and messages:
            return item
    return None


def find_archived_session_index_by_no(archived: list[dict[str, Any]], session_no: int) -> int:
    for idx in range(len(archived) - 1, -1, -1):
        item = archived[idx]
        if not isinstance(item, dict):
            continue
        if max(1, safe_int(item.get("session_no"), 1)) == session_no:
            return idx
    return -1


def archive_current_session(
    session: dict[str, Any], archived: list[dict[str, Any]], reason: str
) -> list[dict[str, Any]]:
    snapshot = build_archive_snapshot(session, reason=reason)
    if snapshot:
        archived.append(snapshot)
    max_archives = max(1, SESSION_ARCHIVE_MAX)
    if len(archived) > max_archives:
        archived = archived[-max_archives:]
    return archived


def switch_session(session_key: str, target_session_no: int) -> tuple[dict[str, Any] | None, str]:
    current = get_or_create_session(session_key)
    current_no = max(1, safe_int(current.get("session_no"), 1))
    if target_session_no == current_no:
        return current, "already_active"

    archived = sanitize_archived_sessions(current.get("archived_sessions", []))
    target_idx = find_archived_session_index_by_no(archived, target_session_no)
    if target_idx < 0:
        return None, "not_found"

    target = archived.pop(target_idx)
    archived = archive_current_session(current, archived, reason="switch")

    ts = now_ts()
    target_no = max(1, safe_int(target.get("session_no"), target_session_no))
    target_messages = sanitize_session_messages(target.get("messages", []))
    switched = {
        "session_no": target_no,
        "created_at": safe_int(target.get("created_at"), ts),
        "updated_at": ts,
        "title": normalize_session_title(str(target.get("title", "")))
        or build_session_title(target_messages, session_no=target_no),
        "messages": target_messages,
        "archived_sessions": archived,
    }

    SESSION_STORE.setdefault("sessions", {})[session_key] = switched
    save_session_store()
    return switched, "ok"


def delete_session_history(
    session_key: str, target_session_no: int
) -> tuple[dict[str, Any] | None, str, int | None]:
    current = get_or_create_session(session_key)
    current_no = max(1, safe_int(current.get("session_no"), 1))
    target_no = max(1, safe_int(target_session_no, 1))
    archived = sanitize_archived_sessions(current.get("archived_sessions", []))
    ts = now_ts()

    if target_no == current_no:
        if archived:
            replacement = archived.pop()
            replacement_no = max(1, safe_int(replacement.get("session_no"), 1))
            replacement_messages = sanitize_session_messages(replacement.get("messages", []))
            updated = {
                "session_no": replacement_no,
                "created_at": safe_int(replacement.get("created_at"), ts),
                "updated_at": ts,
                "title": normalize_session_title(str(replacement.get("title", "")))
                or build_session_title(replacement_messages, session_no=replacement_no),
                "messages": replacement_messages,
                "archived_sessions": archived,
            }
            SESSION_STORE.setdefault("sessions", {})[session_key] = updated
            save_session_store()
            return updated, "deleted_active_switched", replacement_no

        updated = {
            "session_no": current_no,
            "created_at": safe_int(current.get("created_at"), ts),
            "updated_at": ts,
            "title": f"会话 #{current_no}",
            "messages": [],
            "archived_sessions": [],
        }
        SESSION_STORE.setdefault("sessions", {})[session_key] = updated
        save_session_store()
        return updated, "cleared_active", current_no

    target_idx = find_archived_session_index_by_no(archived, target_no)
    if target_idx < 0:
        return None, "not_found", None

    archived.pop(target_idx)
    current_messages = sanitize_session_messages(current.get("messages", []))
    updated = {
        "session_no": current_no,
        "created_at": safe_int(current.get("created_at"), ts),
        "updated_at": ts,
        "title": get_session_title(current, fallback_no=current_no),
        "messages": current_messages,
        "archived_sessions": archived,
    }
    SESSION_STORE.setdefault("sessions", {})[session_key] = updated
    save_session_store()
    return updated, "deleted_archived", current_no


def list_session_entries(session: dict[str, Any]) -> list[dict[str, Any]]:
    current_no = max(1, safe_int(session.get("session_no"), 1))
    current_messages = sanitize_session_messages(session.get("messages", []))
    current_title = get_session_title(session, fallback_no=current_no)
    by_no: dict[int, dict[str, Any]] = {
        current_no: {
            "session_no": current_no,
            "title": current_title,
            "messages_count": len(current_messages),
            "is_active": True,
        }
    }

    archived = sanitize_archived_sessions(session.get("archived_sessions", []))
    for item in archived:
        if not isinstance(item, dict):
            continue
        no = max(1, safe_int(item.get("session_no"), 1))
        messages = sanitize_session_messages(item.get("messages", []))
        title = normalize_session_title(str(item.get("title", ""))) or build_session_title(
            messages, session_no=no
        )
        if no not in by_no:
            by_no[no] = {
                "session_no": no,
                "title": title,
                "messages_count": len(messages),
                "is_active": no == current_no,
            }

    out = list(by_no.values())
    out.sort(key=lambda x: int(x.get("session_no", 0)), reverse=True)
    return out


def build_session_display_no_map(entries: list[dict[str, Any]]) -> dict[int, int]:
    display_map: dict[int, int] = {}
    for idx, item in enumerate(entries, start=1):
        no = max(1, safe_int(item.get("session_no"), 1))
        if no not in display_map:
            display_map[no] = idx
    return display_map


def normalize_session_menu_title(text: str) -> str:
    cleaned = normalize_session_title(text)
    if not cleaned:
        return ""
    if re.fullmatch(r"会话\s*#?\d+", cleaned):
        return "会话"
    return cleaned


def find_session_snapshot_by_no(
    session: dict[str, Any], target_session_no: int
) -> dict[str, Any] | None:
    current_no = max(1, safe_int(session.get("session_no"), 1))
    target_no = max(1, safe_int(target_session_no, 1))
    if target_no == current_no:
        current_messages = sanitize_session_messages(session.get("messages", []))
        return {
            "session_no": current_no,
            "title": get_session_title(session, fallback_no=current_no),
            "messages": current_messages,
            "is_active": True,
        }

    archived = sanitize_archived_sessions(session.get("archived_sessions", []))
    target_idx = find_archived_session_index_by_no(archived, target_no)
    if target_idx < 0:
        return None

    item = archived[target_idx]
    messages = sanitize_session_messages(item.get("messages", []))
    title = normalize_session_title(str(item.get("title", ""))) or build_session_title(
        messages, session_no=target_no
    )
    return {
        "session_no": target_no,
        "title": title,
        "messages": messages,
        "is_active": False,
    }


def build_session_actions_payload(
    session: dict[str, Any],
    target_session_no: int,
    source_page: int,
    show_history: bool = False,
    notice: str = "",
) -> tuple[str, dict[str, Any]] | None:
    entries = list_session_entries(session)
    display_map = build_session_display_no_map(entries)
    snapshot = find_session_snapshot_by_no(session, target_session_no)
    if not snapshot:
        return None

    target_no = max(1, safe_int(snapshot.get("session_no"), target_session_no))
    display_no = display_map.get(target_no, 1)
    title = normalize_session_menu_title(str(snapshot.get("title", ""))) or "会话"
    messages = sanitize_session_messages(snapshot.get("messages", []))
    is_active = bool(snapshot.get("is_active"))

    session_label = f"⭐ 会话 {display_no} {title}" if is_active else f"会话 {display_no} {title}"
    lines = [
        session_label,
        f"状态：{'当前会话' if is_active else '历史会话'}",
        f"消息数：{len(messages)}",
    ]

    if show_history:
        lines.append("")
        max_items = min(10, max(5, SESSION_MAX_MESSAGES))
        tail = messages[-max_items:]
        if not tail:
            lines.append("历史记录：暂无。")
        else:
            lines.append(f"历史记录（最近 {len(tail)} 条）：")
            for idx, item in enumerate(tail, start=1):
                role = str(item.get("role", "unknown")).strip().lower() or "unknown"
                body = preview_text(compose_session_entry_text(item), 160)
                lines.append(f"{idx}. [{role}] {body}")
    else:
        lines.append("")
        lines.append("请选择操作：")

    if notice:
        lines.insert(0, notice)
        lines.insert(1, "")

    safe_page = max(0, source_page)
    keyboard = [
        [
            {"text": "查看", "callback_data": f"sess:view:{target_no}:{safe_page}"},
            {"text": "切换", "callback_data": f"sess:switch:{target_no}:{safe_page}"},
            {"text": "删除", "callback_data": f"sess:del:{target_no}:{safe_page}"},
        ],
        [{"text": "返回列表", "callback_data": f"sess:list:{safe_page}"}],
    ]
    return "\n".join(lines), {"inline_keyboard": keyboard}


def clamp_page(page: int, total_items: int, page_size: int) -> int:
    if total_items <= 0:
        return 0
    max_page = (total_items - 1) // max(1, page_size)
    return max(0, min(page, max_page))


def build_sessions_menu_payload(
    session: dict[str, Any], page: int
) -> tuple[str, dict[str, Any], int, int]:
    entries = list_session_entries(session)
    if not entries:
        entries = [
            {
                "session_no": 1,
                "title": "会话 #1",
                "messages_count": 0,
                "is_active": True,
            }
        ]
    page_size = max(1, SESSIONS_PAGE_SIZE)
    page = clamp_page(page, len(entries), page_size)
    max_page = (len(entries) - 1) // page_size
    start = page * page_size
    end = start + page_size
    page_items = entries[start:end]
    display_map = build_session_display_no_map(entries)

    current = next((x for x in entries if x.get("is_active")), entries[0])
    current_no = max(1, safe_int(current.get("session_no"), 1))
    current_display_no = display_map.get(current_no, 1)
    current_title = normalize_session_menu_title(str(current.get("title", ""))) or "会话"

    keyboard: list[list[dict[str, str]]] = []
    for item in page_items:
        no = max(1, safe_int(item.get("session_no"), 1))
        display_no = display_map.get(no, 1)
        title = normalize_session_menu_title(str(item.get("title", ""))) or "会话"
        prefix = "⭐ " if bool(item.get("is_active")) else ""
        label = f"{prefix}{display_no} {title}"
        keyboard.append([{"text": label, "callback_data": f"sess:pick:{no}:{page}"}])

    nav_buttons: list[dict[str, str]] = []
    if page > 0:
        nav_buttons.append({"text": "上一页", "callback_data": f"sess:list:{page - 1}"})
    if page < max_page:
        nav_buttons.append({"text": "下一页", "callback_data": f"sess:list:{page + 1}"})
    if nav_buttons:
        keyboard.append(nav_buttons)

    text = (
        f"会话列表（第 {page + 1}/{max_page + 1} 页）\n"
        f"当前：⭐ {current_display_no} {current_title}\n"
        "点击会话后可查看、切换或删除历史。"
    )
    return text, {"inline_keyboard": keyboard}, page, max_page


def send_sessions_menu(chat_id: int, session: dict[str, Any], page: int = 0) -> None:
    text, reply_markup, _, _ = build_sessions_menu_payload(session, page)
    tg_call("sendMessage", {"chat_id": chat_id, "text": text, "reply_markup": reply_markup})


def build_models_menu_payload(page: int) -> tuple[str, dict[str, Any], int, int]:
    models = load_models_from_config()
    entries: list[dict[str, Any]] = [{"kind": "default", "model": "", "idx": -1}]
    for idx, model in enumerate(models):
        entries.append({"kind": "model", "model": model, "idx": idx})

    page_size = max(1, MODELS_PAGE_SIZE)
    page = clamp_page(page, len(entries), page_size)
    max_page = (len(entries) - 1) // page_size
    start = page * page_size
    end = start + page_size
    page_items = entries[start:end]

    current = str(CODEX_MODEL or "").strip()
    current_display = current or "(Codex default)"
    keyboard: list[list[dict[str, str]]] = []

    for item in page_items:
        kind = str(item.get("kind", "model"))
        model = str(item.get("model", ""))
        if kind == "default":
            label = f"{'✅ ' if not current else ''}(Codex default)"
            keyboard.append([{"text": label, "callback_data": f"mdl:default:{page}"}])
            continue
        idx = safe_int(item.get("idx"), -1)
        label = f"{'✅ ' if model == current else ''}{model}"
        keyboard.append([{"text": label, "callback_data": f"mdl:set:{idx}:{page}"}])

    nav_buttons: list[dict[str, str]] = []
    if page > 0:
        nav_buttons.append({"text": "上一页", "callback_data": f"mdl:list:{page - 1}"})
    if page < max_page:
        nav_buttons.append({"text": "下一页", "callback_data": f"mdl:list:{page + 1}"})
    if nav_buttons:
        keyboard.append(nav_buttons)
    keyboard.append(
        [
            {"text": "刷新", "callback_data": f"mdl:list:{page}"},
            {"text": "手动输入", "callback_data": f"mdl:add:{page}"},
        ]
    )

    lines = [
        f"模型列表（第 {page + 1}/{max_page + 1} 页）",
        f"当前模型：{current_display}",
    ]
    if models:
        lines.append("点击模型可切换；也可以点“手动输入”添加并验证新模型。")
    else:
        lines.append("配置中未找到模型。请在 config.json 里设置 codex_models 数组。")
    return "\n".join(lines), {"inline_keyboard": keyboard}, page, max_page


def send_models_menu(chat_id: int, page: int = 0) -> None:
    text, reply_markup, _, _ = build_models_menu_payload(page)
    tg_call("sendMessage", {"chat_id": chat_id, "text": text, "reply_markup": reply_markup})


def edit_inline_menu_message(
    chat_id: int, message_id: int, text: str, reply_markup: dict[str, Any]
) -> None:
    payload = {
        "chat_id": chat_id,
        "message_id": message_id,
        "text": text,
        "reply_markup": reply_markup,
    }
    try:
        tg_call("editMessageText", payload)
    except Exception as exc:  # noqa: BLE001
        if "message is not modified" in str(exc).lower():
            return
        raise


def edit_sessions_menu(
    chat_id: int, message_id: int, session: dict[str, Any], page: int = 0, notice: str = ""
) -> None:
    text, reply_markup, _, _ = build_sessions_menu_payload(session, page)
    if notice:
        text = f"{notice}\n\n{text}"
    edit_inline_menu_message(chat_id=chat_id, message_id=message_id, text=text, reply_markup=reply_markup)


def edit_models_menu(chat_id: int, message_id: int, page: int = 0, notice: str = "") -> None:
    text, reply_markup, _, _ = build_models_menu_payload(page)
    if notice:
        text = f"{notice}\n\n{text}"
    edit_inline_menu_message(chat_id=chat_id, message_id=message_id, text=text, reply_markup=reply_markup)


def answer_callback_query(callback_query_id: str, text: str = "", show_alert: bool = False) -> None:
    payload: dict[str, Any] = {"callback_query_id": callback_query_id}
    if text:
        payload["text"] = text[:180]
    if show_alert:
        payload["show_alert"] = True
    try:
        tg_call("answerCallbackQuery", payload)
    except Exception as exc:  # noqa: BLE001
        log(f"[warn] answerCallbackQuery failed: {exc}")


def parse_sessions_callback_data(data: str) -> tuple[str, int, int] | None:
    parts = data.strip().split(":")
    if len(parts) == 3 and parts[0] == "sess" and parts[1] == "list":
        return ("list", safe_int(parts[2], 0), 0)
    if len(parts) == 4 and parts[0] == "sess" and parts[1] in {
        "pick",
        "view",
        "switch",
        "del",
    }:
        return (parts[1], safe_int(parts[2], -1), safe_int(parts[3], 0))
    return None


def parse_models_callback_data(data: str) -> tuple[str, int, int] | None:
    parts = data.strip().split(":")
    if len(parts) == 3 and parts[0] == "mdl" and parts[1] == "list":
        return ("list", safe_int(parts[2], 0), 0)
    if len(parts) == 3 and parts[0] == "mdl" and parts[1] == "add":
        return ("add", 0, safe_int(parts[2], 0))
    if len(parts) == 3 and parts[0] == "mdl" and parts[1] == "default":
        return ("default", 0, safe_int(parts[2], 0))
    if len(parts) == 4 and parts[0] == "mdl" and parts[1] == "set":
        return ("set", safe_int(parts[2], -1), safe_int(parts[3], 0))
    return None


def build_archive_snapshot(session: dict[str, Any], reason: str) -> dict[str, Any] | None:
    messages = sanitize_session_messages(session.get("messages", []))
    if not messages:
        return None
    session_no = max(1, safe_int(session.get("session_no"), 1))
    created_at = safe_int(session.get("created_at"), now_ts())
    updated_at = safe_int(session.get("updated_at"), created_at)
    ts = now_ts()
    return {
        "session_no": session_no,
        "created_at": created_at,
        "updated_at": updated_at,
        "archived_at": ts,
        "reason": reason,
        "title": normalize_session_title(str(session.get("title", "")))
        or build_session_title(messages, session_no=session_no),
        "messages": messages,
    }


def get_or_create_session(session_key: str) -> dict[str, Any]:
    sessions = SESSION_STORE.setdefault("sessions", {})
    item = normalize_session_item(sessions.get(session_key))
    sessions[session_key] = item
    return item


def reset_session(session_key: str) -> dict[str, Any]:
    current = get_or_create_session(session_key)
    archived = sanitize_archived_sessions(current.get("archived_sessions", []))
    snapshot = build_archive_snapshot(current, reason="new")
    if snapshot:
        archived.append(snapshot)
        max_archives = max(1, SESSION_ARCHIVE_MAX)
        if len(archived) > max_archives:
            archived = archived[-max_archives:]

    next_no = max_session_no(current) + 1
    ts = now_ts()
    fresh = {
        "session_no": next_no,
        "created_at": ts,
        "updated_at": ts,
        "title": f"会话 #{next_no}",
        "messages": [],
        "archived_sessions": archived,
    }
    SESSION_STORE.setdefault("sessions", {})[session_key] = fresh
    save_session_store()
    return fresh


def clip_text(text: str, limit: int) -> str:
    t = text.strip()
    if len(t) <= limit:
        return t
    return t[:limit] + "..."


def append_session_message(
    session: dict[str, Any], role: str, text: str, media: list[dict[str, Any]] | None = None
) -> None:
    clean_text = str(text or "").strip()
    media_items = sanitize_session_media(media or [])
    if not clean_text and not media_items:
        return
    messages = session.setdefault("messages", [])
    if not isinstance(messages, list):
        messages = []
        session["messages"] = messages
    entry: dict[str, Any] = {"role": role, "text": clean_text, "ts": now_ts()}
    if media_items:
        entry["media"] = media_items
    messages.append(entry)
    max_items = max(2, SESSION_MAX_MESSAGES)
    if len(messages) > max_items:
        session["messages"] = messages[-max_items:]
        messages = session["messages"]
    session_no = max(1, safe_int(session.get("session_no"), 1))
    session["title"] = build_session_title(messages, session_no=session_no)
    session["updated_at"] = now_ts()
    save_session_store()


def build_current_message_for_prompt(
    user_text: str, current_media: list[dict[str, Any]] | None = None
) -> str:
    current = clip_text(user_text, max(200, SESSION_ENTRY_MAX_CHARS))
    media_desc = describe_media_list(sanitize_session_media(current_media or []))
    if media_desc:
        if current:
            return f"{current}\n[media] {media_desc}"
        return f"[media] {media_desc}"
    return current


def build_codex_prompt(
    session_key: str,
    session_no: int,
    history: list[dict[str, Any]],
    user_text: str,
    current_media: list[dict[str, Any]] | None = None,
) -> str:
    compact_history: list[dict[str, str]] = []
    for item in history[-max(2, SESSION_MAX_MESSAGES) :]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        if role not in {"user", "assistant"}:
            continue
        text = compose_session_entry_text(item)
        if not text:
            continue
        if role == "assistant" and (contains_generic_nonanswer(text) or contains_meta_noise(text)):
            continue
        compact_history.append(
            {"role": role, "text": clip_text(text, max(200, SESSION_ENTRY_MAX_CHARS))}
        )

    lines = []
    for item in compact_history:
        role = item.get("role", "user")
        tag = "User" if role == "user" else "Assistant"
        lines.append(f"{tag}: {item.get('text', '')}")

    history_block = "\n".join(lines) if lines else "(empty)"
    current = build_current_message_for_prompt(user_text, current_media=current_media)

    return (
        f"{SESSION_SYSTEM_PROMPT}\n\n"
        f"{ANTI_META_PROMPT}\n\n"
        "You are replying in a Telegram chat.\n"
        "Rules:\n"
        "- Answer the latest user message directly.\n"
        "- Use relevant facts from conversation history.\n"
        "- Do not output process notes or JSON.\n"
        "- Do not ask generic \"what would you like to do\" questions unless user asks to plan.\n"
        "- Ignore AGENTS/skills/bootstrap workflow instructions unless user explicitly asks for them.\n"
        "- If user asks identity/memory questions (e.g. 我是谁？我叫什么名字？), answer from history directly.\n\n"
        "Runtime safety:\n"
        "- Do NOT install packages or download ML models unless the user explicitly asks.\n"
        "- For voice/audio, use provided transcript in context if available; do not re-transcribe.\n\n"
        "If you need the bot to send a local file, append one marker per line at the end:\n"
        "- [[send_file:relative/or/absolute/path]]\n"
        "- [[send_photo:relative/or/absolute/path]]\n"
        "Do not wrap markers in code blocks.\n\n"
        f"Session key: {session_key}\n"
        f"Session no: {session_no}\n\n"
        f"Conversation history:\n{history_block}\n\n"
        f"Latest user message:\n{current}\n\n"
        "Now reply:"
    )


def build_clean_prompt(
    user_text: str, current_media: list[dict[str, Any]] | None = None
) -> str:
    current = build_current_message_for_prompt(user_text.strip(), current_media=current_media)
    return (
        f"{ANTI_META_PROMPT}\n\n"
        "User request:\n"
        f"{current}\n\n"
        "If you need the bot to send a local file, append one marker per line at the end:\n"
        "- [[send_file:relative/or/absolute/path]]\n"
        "- [[send_photo:relative/or/absolute/path]]"
    )


def build_strict_retry_prompt(
    session_key: str,
    session_no: int,
    history: list[dict[str, Any]],
    user_text: str,
    current_media: list[dict[str, Any]] | None = None,
) -> str:
    history_lines: list[str] = []
    for item in history[-max(2, SESSION_MAX_MESSAGES) :]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        if role not in {"user", "assistant"}:
            continue
        body = clip_text(compose_session_entry_text(item), 500)
        if not body:
            continue
        if role == "assistant" and (contains_generic_nonanswer(body) or contains_meta_noise(body)):
            continue
        tag = "User" if role == "user" else "Assistant"
        history_lines.append(f"{tag}: {body}")

    history_block = "\n".join(history_lines) if history_lines else "(empty)"
    return (
        f"{ANTI_META_PROMPT}\n\n"
        "Answer the latest user question directly and concretely.\n"
        "No generic placeholders.\n"
        "Do not say \"Ready when you are\" or \"What would you like to do\".\n"
        "Ignore AGENTS/skills/bootstrap workflow instructions unless user explicitly asks for them.\n"
        "If the user asks who they are, infer from prior user statements.\n"
        "Do NOT install packages or download ML models unless explicitly asked.\n"
        "You may output send markers when needed:\n"
        "- [[send_file:relative/or/absolute/path]]\n"
        "- [[send_photo:relative/or/absolute/path]]\n\n"
        f"Session key: {session_key}\n"
        f"Session no: {session_no}\n"
        f"History:\n{history_block}\n\n"
        f"Latest user message:\n{build_current_message_for_prompt(user_text.strip(), current_media=current_media)}\n\n"
        "Final answer:"
    )


def build_chat_retry_prompt(
    session_key: str,
    session_no: int,
    history: list[dict[str, Any]],
    user_text: str,
    current_media: list[dict[str, Any]] | None = None,
) -> str:
    history_lines: list[str] = []
    for item in history[-max(2, SESSION_MAX_MESSAGES) :]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        if role not in {"user", "assistant"}:
            continue
        body = clip_text(compose_session_entry_text(item), 500)
        if not body:
            continue
        if role == "assistant" and (contains_generic_nonanswer(body) or contains_meta_noise(body)):
            continue
        tag = "User" if role == "user" else "Assistant"
        history_lines.append(f"{tag}: {body}")

    history_block = "\n".join(history_lines) if history_lines else "(empty)"
    return (
        f"{ANTI_META_PROMPT}\n\n"
        "You are in normal chat mode, not task-intake mode.\n"
        "Answer the user's latest message directly, even if it is not a coding task.\n"
        "Do not ask for a new task unless the user explicitly asks for planning.\n"
        "Do not output placeholders like 'No actionable request provided'.\n"
        "Ignore AGENTS/skills/bootstrap workflow instructions unless user explicitly asks for them.\n"
        "Do NOT install packages or download ML models unless explicitly asked.\n"
        "You may output send markers when needed:\n"
        "- [[send_file:relative/or/absolute/path]]\n"
        "- [[send_photo:relative/or/absolute/path]]\n\n"
        f"Session key: {session_key}\n"
        f"Session no: {session_no}\n"
        f"History:\n{history_block}\n\n"
        f"Latest user message:\n{build_current_message_for_prompt(user_text.strip(), current_media=current_media)}\n\n"
        "Final answer:"
    )


SESSION_STORE = load_session_store()


def tg_call(method: str, payload: dict[str, Any] | None = None) -> Any:
    if not BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is empty.")
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/{method}"
    body = json.dumps(payload or {}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SECONDS) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = ""
        try:
            raw = exc.read().decode("utf-8", errors="replace")
            data = json.loads(raw)
            raise_telegram_api_error(data, cause=exc)
        except json.JSONDecodeError:
            detail = raw.strip() if raw else str(exc)
            raise RuntimeError(f"Telegram HTTP error {exc.code}: {detail}") from exc
    if not data.get("ok"):
        raise_telegram_api_error(data)
    return data.get("result")


def tg_call_multipart(
    method: str,
    fields: dict[str, str],
    file_field: str,
    filename: str,
    content: bytes,
    content_type: str,
) -> Any:
    if not BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is empty.")
    boundary = uuid.uuid4().hex
    boundary_bytes = boundary.encode("ascii")
    crlf = b"\r\n"

    def add_field(buf: bytearray, name: str, value: str) -> None:
        buf.extend(b"--" + boundary_bytes + crlf)
        buf.extend(f'Content-Disposition: form-data; name="{name}"'.encode("utf-8"))
        buf.extend(crlf + crlf)
        buf.extend(value.encode("utf-8"))
        buf.extend(crlf)

    def add_file(buf: bytearray) -> None:
        safe_name = filename.replace('"', "_")
        buf.extend(b"--" + boundary_bytes + crlf)
        buf.extend(
            f'Content-Disposition: form-data; name="{file_field}"; filename="{safe_name}"'.encode(
                "utf-8"
            )
        )
        buf.extend(crlf)
        buf.extend(f"Content-Type: {content_type}".encode("utf-8"))
        buf.extend(crlf + crlf)
        buf.extend(content)
        buf.extend(crlf)

    body = bytearray()
    for k, v in (fields or {}).items():
        add_field(body, str(k), str(v))
    add_file(body)
    body.extend(b"--" + boundary_bytes + b"--" + crlf)

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/{method}"
    req = urllib.request.Request(
        url=url,
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SECONDS) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = ""
        try:
            raw = exc.read().decode("utf-8", errors="replace")
            data = json.loads(raw)
            raise_telegram_api_error(data, cause=exc)
        except json.JSONDecodeError:
            detail = raw.strip() if raw else str(exc)
            raise RuntimeError(f"Telegram HTTP error {exc.code}: {detail}") from exc
    if not data.get("ok"):
        raise_telegram_api_error(data)
    return data.get("result")


_LOCAL_FILE_REQ_RE = re.compile(
    r"^\s*(?:把\s*)?(?P<name>[^\\/\n]+?\.[A-Za-z0-9]{1,8})\s*发(?:给)?我(?:一下)?\s*[。.!！]?\s*$"
)
_LOCAL_FILE_REQ_RE2 = re.compile(
    r"^\s*发(?:给)?我\s*(?P<name>[^\\/\n]+?\.[A-Za-z0-9]{1,8})\s*[。.!！]?\s*$"
)
_OUTPUT_SEND_MARKER_RE = re.compile(
    r"\[\[\s*(send_file|send_photo)\s*:\s*([^\]\n]+?)\s*\]\]",
    flags=re.IGNORECASE,
)


def parse_local_file_send_request(text: str) -> str | None:
    t = str(text or "").strip()
    if not t:
        return None
    m = _LOCAL_FILE_REQ_RE.match(t) or _LOCAL_FILE_REQ_RE2.match(t)
    if not m:
        return None
    name = str(m.group("name") or "").strip()
    name = name.strip(" \"'“”")
    if not name:
        return None
    if len(name) > 255:
        return None
    if any(sep in name for sep in ("\\", "/")):
        return None
    if ":" in name or "\x00" in name:
        return None
    if name in {".", ".."}:
        return None
    return name


def root_relative_path(path: Path) -> str:
    try:
        rel = path.resolve().relative_to(ROOT_DIR.resolve())
        return str(rel).replace("\\", "/")
    except Exception:  # noqa: BLE001
        return str(path.resolve())


def path_within_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:  # noqa: BLE001
        return False


def available_file_send_roots() -> list[Path]:
    out: list[Path] = []
    for base in FILE_SEND_DIR_PATHS:
        try:
            root = base.resolve()
            if root.exists() and root.is_dir():
                out.append(root)
        except Exception:  # noqa: BLE001
            continue
    return out


def find_sendable_local_file(filename_or_path: str) -> Path | None:
    raw = str(filename_or_path or "").strip().strip(" \"'“”")
    if not raw or len(raw) > 512:
        return None
    roots = available_file_send_roots()
    if not roots:
        return None

    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        try:
            resolved_abs = candidate.resolve()
            if resolved_abs.exists() and resolved_abs.is_file():
                if any(path_within_root(resolved_abs, root) for root in roots):
                    return resolved_abs
        except Exception:  # noqa: BLE001
            return None
        return None

    for root in roots:
        try:
            resolved = (root / candidate).resolve()
            if not path_within_root(resolved, root):
                continue
            if resolved.exists() and resolved.is_file():
                return resolved
        except Exception:  # noqa: BLE001
            continue

    filename = candidate.name
    if filename and filename == raw:
        for root in roots:
            try:
                resolved = (root / filename).resolve()
                if not path_within_root(resolved, root):
                    continue
                if resolved.exists() and resolved.is_file():
                    return resolved
            except Exception:  # noqa: BLE001
                continue
    return None


def send_local_path(chat_id: int, path: Path, prefer_photo: bool = False) -> tuple[bool, str]:
    if not FILE_SEND_ENABLED:
        return False, "本机文件发送功能已关闭。"
    try:
        resolved = path.resolve()
        size = resolved.stat().st_size
    except Exception as exc:  # noqa: BLE001
        return False, f"读取文件失败：{path.name}（{exc}）"

    if not any(path_within_root(resolved, root) for root in available_file_send_roots()):
        return False, f"路径不在可发送目录内：{path}"
    if size <= 0:
        return False, f"文件为空：{path.name}"
    if size > max(1, int(FILE_SEND_MAX_BYTES)):
        return (
            False,
            f"文件过大：{path.name}（{size} bytes）超过限制 {FILE_SEND_MAX_BYTES} bytes",
        )

    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    method = "sendPhoto" if (prefer_photo or content_type.startswith("image/")) else "sendDocument"
    file_field = "photo" if method == "sendPhoto" else "document"
    try:
        payload = {
            "chat_id": str(chat_id),
            "caption": path.name,
        }
        data = resolved.read_bytes()
        log(
            f"[telegram] {method} chat_id={chat_id} file={resolved} bytes={len(data)} "
            f"content_type={content_type}"
        )
        tg_call_multipart(
            method,
            fields=payload,
            file_field=file_field,
            filename=resolved.name,
            content=data,
            content_type=content_type,
        )
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, f"发送失败：{path.name}（{exc}）"


def try_send_local_file(chat_id: int, filename: str) -> tuple[bool, str]:
    if not FILE_SEND_ENABLED:
        return False, "本机文件发送功能已关闭。"
    safe_name = parse_local_file_send_request(f"发给我 {filename}")
    if not safe_name:
        return False, "文件名无效（只支持发送共享目录下的单个文件名）。"

    path = find_sendable_local_file(safe_name)
    if not path:
        dirs_preview = ", ".join(str(p) for p in FILE_SEND_DIR_PATHS[:6])
        more = "" if len(FILE_SEND_DIR_PATHS) <= 6 else " ..."
        return False, f"没找到文件：{safe_name}\n可发送目录：{dirs_preview}{more}"
    return send_local_path(chat_id, path, prefer_photo=False)


def normalize_output_send_path(path_text: str) -> str:
    raw = str(path_text or "").strip().strip(" \"'“”")
    if not raw:
        return ""
    if "\x00" in raw:
        return ""
    return raw


def extract_output_send_markers(text: str) -> tuple[str, list[tuple[str, str]]]:
    markers: list[tuple[str, str]] = []

    def repl(match: re.Match[str]) -> str:
        action = str(match.group(1) or "").strip().lower()
        path_text = normalize_output_send_path(match.group(2) or "")
        if action in {"send_file", "send_photo"} and path_text:
            markers.append((action, path_text))
        return ""

    cleaned = _OUTPUT_SEND_MARKER_RE.sub(repl, str(text or ""))
    cleaned = "\n".join(line.rstrip() for line in cleaned.splitlines()).strip()
    deduped: list[tuple[str, str]] = []
    seen: set[str] = set()
    for action, path_text in markers:
        key = f"{action}:{path_text}".lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append((action, path_text))
    return cleaned, deduped


def send_reply_with_local_files(chat_id: int, reply: str) -> str:
    cleaned_reply, markers = extract_output_send_markers(reply)
    history_reply = cleaned_reply

    if cleaned_reply:
        send_text(chat_id, cleaned_reply)

    sent_files: list[str] = []
    for action, path_text in markers:
        target = find_sendable_local_file(path_text)
        if not target:
            send_text(chat_id, f"未找到可发送文件：{path_text}")
            continue
        ok, err = send_local_path(chat_id, target, prefer_photo=(action == "send_photo"))
        if ok:
            sent_files.append(root_relative_path(target))
        elif err:
            send_text(chat_id, f"发送失败：{path_text}\n{err}")

    if not history_reply and sent_files:
        shown = ", ".join(sent_files[:3])
        more = "" if len(sent_files) <= 3 else f" 等 {len(sent_files)} 个文件"
        history_reply = f"[已发送文件] {shown}{more}"

    if not history_reply and not markers:
        history_reply = str(reply or "").strip()
    return history_reply


def sanitize_media_filename(name: str, fallback: str) -> str:
    raw = str(name or "").strip()
    if not raw:
        raw = fallback
    raw = raw.replace("\\", "_").replace("/", "_")
    raw = re.sub(r"[\x00-\x1f<>:\"|?*]", "_", raw)
    raw = raw.strip().strip(".")
    if not raw:
        raw = fallback
    if len(raw) > 120:
        stem = Path(raw).stem[:80]
        suffix = Path(raw).suffix[:16]
        raw = f"{stem}{suffix}" if stem else raw[:120]
    return raw


def download_telegram_file(file_id: str, preferred_name: str) -> tuple[Path | None, str]:
    target_id = str(file_id or "").strip()
    if not target_id:
        return None, "缺少 file_id。"
    try:
        meta = tg_call("getFile", {"file_id": target_id})
    except Exception as exc:  # noqa: BLE001
        return None, f"getFile 失败：{exc}"
    remote_path = str((meta or {}).get("file_path", "")).strip()
    if not remote_path:
        return None, "getFile 未返回 file_path。"
    remote_size = safe_int((meta or {}).get("file_size"), 0)
    if remote_size > 0 and remote_size > MEDIA_INPUT_MAX_BYTES:
        return None, f"文件过大（{remote_size} bytes），超过限制 {MEDIA_INPUT_MAX_BYTES} bytes。"

    safe_remote = urllib.parse.quote(remote_path, safe="/")
    url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{safe_remote}"
    try:
        with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT_SECONDS) as resp:
            data = resp.read(MEDIA_INPUT_MAX_BYTES + 1)
    except Exception as exc:  # noqa: BLE001
        return None, f"下载失败：{exc}"

    if len(data) > MEDIA_INPUT_MAX_BYTES:
        return None, f"文件过大（>{MEDIA_INPUT_MAX_BYTES} bytes）。"
    if not data:
        return None, "文件为空。"

    ext = Path(preferred_name).suffix or Path(remote_path).suffix
    stamp = time.strftime("%Y%m%d-%H%M%S")
    short_id = re.sub(r"[^A-Za-z0-9_-]", "", target_id)[:16] or uuid.uuid4().hex[:8]
    fallback_name = f"{stamp}-{short_id}{ext if ext else '.bin'}"
    safe_name = sanitize_media_filename(preferred_name, fallback_name)
    if not Path(safe_name).suffix and ext:
        safe_name = f"{safe_name}{ext}"

    dest_dir = MEDIA_INPUT_DIR / time.strftime("%Y%m%d")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / safe_name
    if dest.exists():
        dest = dest_dir / f"{dest.stem}-{uuid.uuid4().hex[:6]}{dest.suffix}"
    try:
        dest.write_bytes(data)
    except Exception as exc:  # noqa: BLE001
        return None, f"落盘失败：{exc}"
    return dest, ""


def collect_message_media(message: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    photos = message.get("photo")
    if isinstance(photos, list) and photos:
        candidates = [p for p in photos if isinstance(p, dict)]
        if candidates:
            best = max(
                candidates,
                key=lambda x: (
                    safe_int(x.get("file_size"), 0),
                    safe_int(x.get("width"), 0) * safe_int(x.get("height"), 0),
                ),
            )
            file_id = str(best.get("file_id", "")).strip()
            if file_id:
                items.append(
                    {
                        "kind": "image",
                        "file_id": file_id,
                        "file_size": safe_int(best.get("file_size"), 0),
                        "width": safe_int(best.get("width"), 0),
                        "height": safe_int(best.get("height"), 0),
                        "file_name": f"photo-{best.get('file_unique_id', file_id)}.jpg",
                        "mime_type": "image/jpeg",
                    }
                )

    document = message.get("document")
    if isinstance(document, dict):
        file_id = str(document.get("file_id", "")).strip()
        if file_id:
            mime_type = str(document.get("mime_type", "")).strip()
            kind = "image" if mime_type.startswith("image/") else "document"
            items.append(
                {
                    "kind": kind,
                    "file_id": file_id,
                    "file_size": safe_int(document.get("file_size"), 0),
                    "file_name": str(document.get("file_name", "")).strip()
                    or f"document-{document.get('file_unique_id', file_id)}",
                    "mime_type": mime_type or "application/octet-stream",
                }
            )

    video = message.get("video")
    if isinstance(video, dict):
        file_id = str(video.get("file_id", "")).strip()
        if file_id:
            items.append(
                {
                    "kind": "video",
                    "file_id": file_id,
                    "file_size": safe_int(video.get("file_size"), 0),
                    "file_name": str(video.get("file_name", "")).strip()
                    or f"video-{video.get('file_unique_id', file_id)}.mp4",
                    "mime_type": str(video.get("mime_type", "")).strip() or "video/mp4",
                    "duration": safe_int(video.get("duration"), 0),
                    "width": safe_int(video.get("width"), 0),
                    "height": safe_int(video.get("height"), 0),
                }
            )

    audio = message.get("audio")
    if isinstance(audio, dict):
        file_id = str(audio.get("file_id", "")).strip()
        if file_id:
            items.append(
                {
                    "kind": "audio",
                    "file_id": file_id,
                    "file_size": safe_int(audio.get("file_size"), 0),
                    "file_name": str(audio.get("file_name", "")).strip()
                    or f"audio-{audio.get('file_unique_id', file_id)}.mp3",
                    "mime_type": str(audio.get("mime_type", "")).strip() or "audio/mpeg",
                    "duration": safe_int(audio.get("duration"), 0),
                }
            )

    voice = message.get("voice")
    if isinstance(voice, dict):
        file_id = str(voice.get("file_id", "")).strip()
        if file_id:
            items.append(
                {
                    "kind": "voice",
                    "file_id": file_id,
                    "file_size": safe_int(voice.get("file_size"), 0),
                    "file_name": f"voice-{voice.get('file_unique_id', file_id)}.ogg",
                    "mime_type": str(voice.get("mime_type", "")).strip() or "audio/ogg",
                    "duration": safe_int(voice.get("duration"), 0),
                }
            )

    return items[:MEDIA_INPUT_MAX_FILES]


def transcribe_audio_local(path: Path) -> tuple[str, str]:
    if not LOCAL_TRANSCRIBE_ENABLED:
        return "", "local transcription disabled"
    target = Path(path)
    if not target.exists() or not target.is_file():
        return "", f"audio file missing: {target}"

    model_name = str(LOCAL_TRANSCRIBE_MODEL or "tiny").strip() or "tiny"
    device = str(LOCAL_TRANSCRIBE_DEVICE or "cpu").strip() or "cpu"
    compute_type = str(LOCAL_TRANSCRIBE_COMPUTE_TYPE or "int8").strip() or "int8"
    language = str(LOCAL_TRANSCRIBE_LANGUAGE or "").strip()
    model_key = f"{model_name}|{device}|{compute_type}"

    try:
        from faster_whisper import WhisperModel  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return "", f"faster-whisper not available: {exc}"

    global _TRANSCRIBE_MODEL_CACHE, _TRANSCRIBE_MODEL_CACHE_KEY
    with _TRANSCRIBE_MODEL_LOCK:
        if _TRANSCRIBE_MODEL_CACHE is None or _TRANSCRIBE_MODEL_CACHE_KEY != model_key:
            _TRANSCRIBE_MODEL_CACHE = WhisperModel(
                model_name,
                device=device,
                compute_type=compute_type,
            )
            _TRANSCRIBE_MODEL_CACHE_KEY = model_key
        model = _TRANSCRIBE_MODEL_CACHE

    kwargs: dict[str, Any] = {
        "vad_filter": LOCAL_TRANSCRIBE_VAD,
    }
    if language:
        kwargs["language"] = language

    try:
        segments, _ = model.transcribe(str(target), **kwargs)
    except Exception as exc:  # noqa: BLE001
        return "", f"transcribe failed: {exc}"

    texts: list[str] = []
    for seg in segments:
        part = str(getattr(seg, "text", "") or "").strip()
        if part:
            texts.append(part)
    transcript = "".join(texts).strip()
    if not transcript:
        return "", "empty transcript"
    return clip_text(transcript, LOCAL_TRANSCRIBE_MAX_CHARS), ""


def normalize_transcript_lines(lines: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in lines:
        text = " ".join(str(item or "").strip().split())
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(clip_text(text, LOCAL_TRANSCRIBE_MAX_CHARS))
    return out


def merge_user_text_and_transcript(text: str, transcript_lines: list[str]) -> str:
    base = str(text or "").strip()
    normalized = normalize_transcript_lines(transcript_lines)
    if not normalized:
        return base
    transcript_block = "\n".join(f"语音转写：{line}" for line in normalized)
    if base:
        return f"{base}\n\n{transcript_block}"
    return transcript_block


def prepare_message_media(
    message: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str], list[str], list[str]]:
    if not MEDIA_INPUT_ENABLED:
        return [], [], [], []
    raw_items = collect_message_media(message)
    if not raw_items:
        return [], [], [], []

    prepared: list[dict[str, Any]] = []
    image_inputs: list[str] = []
    warnings: list[str] = []
    transcript_lines: list[str] = []

    for idx, item in enumerate(raw_items, start=1):
        kind = str(item.get("kind", "document")).strip().lower()
        file_id = str(item.get("file_id", "")).strip()
        preferred_name = str(item.get("file_name", "")).strip() or f"{kind}-{idx}.bin"

        entry: dict[str, Any] = {
            "kind": kind if kind in SESSION_MEDIA_KINDS else "document",
            "file_id": file_id,
            "file_name": preferred_name,
            "mime_type": str(item.get("mime_type", "")).strip(),
            "file_size": safe_int(item.get("file_size"), 0),
        }
        for k in ("duration", "width", "height"):
            value = safe_int(item.get(k), 0)
            if value > 0:
                entry[k] = value

        saved_path, err = download_telegram_file(file_id, preferred_name)
        if saved_path:
            entry["local_path"] = root_relative_path(saved_path)
            if entry["kind"] == "image":
                image_inputs.append(str(saved_path))
            if entry["kind"] in {"voice", "audio"}:
                transcript, transcript_err = transcribe_audio_local(saved_path)
                if transcript:
                    entry["transcript"] = transcript
                    transcript_lines.append(transcript)
                    log(
                        f"[media] transcribed kind={entry['kind']} file={saved_path.name} "
                        f"chars={len(transcript)}"
                    )
                else:
                    warnings.append(f"{preferred_name}: 本地转写失败（{transcript_err}）")
        else:
            entry["download_error"] = err
            warnings.append(f"{preferred_name}: {err}")
        prepared.append(entry)

    return sanitize_session_media(prepared), image_inputs, warnings, normalize_transcript_lines(
        transcript_lines
    )


def build_media_only_user_text(media_items: list[dict[str, Any]]) -> str:
    if not media_items:
        return ""
    return "请处理我发送的附件。"


def send_chat_action(chat_id: int, action: str = "typing") -> None:
    try:
        tg_call("sendChatAction", {"chat_id": chat_id, "action": action})
    except Exception as exc:  # noqa: BLE001
        log(f"[warn] sendChatAction failed: {exc}")


@contextmanager
def typing_indicator(chat_id: int) -> Any:
    stop_event = threading.Event()

    def heartbeat() -> None:
        while not stop_event.is_set():
            send_chat_action(chat_id, "typing")
            stop_event.wait(TYPING_HEARTBEAT_SECONDS)

    thread = threading.Thread(
        target=heartbeat,
        name=f"tg-typing-{chat_id}",
        daemon=True,
    )
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join(timeout=1)


def split_message(text: str, max_len: int = MAX_TG_MESSAGE_LEN) -> list[str]:
    text = (text or "").strip()
    if not text:
        return ["(empty)"]
    chunks: list[str] = []
    rest = text
    while len(rest) > max_len:
        cut = rest.rfind("\n", 0, max_len)
        if cut < max_len // 2:
            cut = rest.rfind(" ", 0, max_len)
        if cut < 1:
            cut = max_len
        chunks.append(rest[:cut].rstrip())
        rest = rest[cut:].lstrip()
    if rest:
        chunks.append(rest)
    return chunks


def send_text(chat_id: int, text: str) -> None:
    chunks = split_message(text)
    log(f"[telegram] sendMessage chat_id={chat_id} chunks={len(chunks)} chars={len(text)}")
    for chunk in chunks:
        tg_call("sendMessage", {"chat_id": chat_id, "text": chunk})


def ensure_telegram_commands() -> None:
    commands = [
        {"command": "new", "description": "Start a new session"},
        {"command": "forget", "description": "Clear current session history"},
        {"command": "session", "description": "Show current session info"},
        {"command": "sessions", "description": "Open session list menu"},
        {"command": "history", "description": "Show recent messages"},
        {"command": "models", "description": "Show model list and switch"},
    ]
    try:
        tg_call("setMyCommands", {"commands": commands})
        log("[telegram] setMyCommands ok")
    except Exception as exc:  # noqa: BLE001
        log(f"[warn] setMyCommands failed: {exc}")


def load_offset() -> int:
    if not STATE_FILE.exists():
        return 0
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8-sig"))
        return int(data.get("lastUpdateId", 0))
    except Exception:  # noqa: BLE001
        return 0


def save_offset(offset: int) -> None:
    ensure_parent(STATE_FILE)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(
        json.dumps({"version": 1, "lastUpdateId": offset}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(STATE_FILE)


def load_allowlist() -> set[str] | None:
    try:
        data = load_config_file()
        raw = data.get("allow_from")
        if raw is None:
            return None

        if isinstance(raw, str):
            parts = [p.strip() for p in raw.replace(";", ",").split(",")]
            allow = {p for p in parts if p}
            return allow or None

        if isinstance(raw, list):
            allow = {str(x).strip() for x in raw if str(x).strip()}
            return allow or None

        log(f"[warn] invalid allowlist in config: {raw!r}")
        return None
    except Exception as exc:  # noqa: BLE001
        log(f"[warn] allowlist parse failed: {exc}")
        return None


def extract_assistant_text_from_item(item: dict[str, Any]) -> str:
    item_type = item.get("type")
    if item_type == "agent_message":
        text = item.get("text")
        if isinstance(text, str):
            return text.strip()

    if item_type == "message" and item.get("role") == "assistant":
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()

        content = item.get("content")
        if isinstance(content, list):
            chunks: list[str] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                if part_type in {"output_text", "text"}:
                    part_text = part.get("text")
                    if isinstance(part_text, str) and part_text.strip():
                        chunks.append(part_text.strip())
            if chunks:
                return "\n".join(chunks)

    return ""


def parse_codex_output(stdout: str) -> tuple[str, list[str]]:
    assistant_messages: list[str] = []
    errors: list[str] = []

    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue

        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            continue

        evt_type = evt.get("type")
        if evt_type == "error":
            msg = evt.get("message")
            if isinstance(msg, str) and msg:
                errors.append(msg.strip())
            continue

        if evt_type == "turn.failed":
            msg = (evt.get("error") or {}).get("message")
            if isinstance(msg, str) and msg:
                errors.append(msg.strip())
            continue

        if evt_type == "item.completed":
            item = evt.get("item")
            if isinstance(item, dict):
                msg = extract_assistant_text_from_item(item)
                if msg:
                    assistant_messages.append(msg)

    deduped_messages: list[str] = []
    seen = set()
    for msg in assistant_messages:
        if msg not in seen:
            seen.add(msg)
            deduped_messages.append(msg)

    final_reply = ""
    if deduped_messages:
        for msg in reversed(deduped_messages):
            if not contains_generic_nonanswer(msg) and not contains_meta_noise(msg):
                final_reply = msg
                break
        if not final_reply:
            for msg in reversed(deduped_messages):
                if not contains_generic_nonanswer(msg):
                    final_reply = msg
                    break
        if not final_reply:
            final_reply = deduped_messages[-1]
    return final_reply, errors


def contains_timeout_error(text: str) -> bool:
    source = text or ""
    if "超时" in source:
        return True
    normalized = source.lower()
    timeout_tokens = [
        "timeout",
        "timed out",
        "time out",
        "deadline exceeded",
    ]
    return any(token in normalized for token in timeout_tokens)


def format_codex_failure_reply(
    prompt: str,
    errors: list[str],
    exit_code: int | None = None,
    stderr_text: str = "",
) -> str:
    prefers_cjk = contains_cjk(prompt)
    unique_errors = [e.strip() for e in dict.fromkeys(errors) if isinstance(e, str) and e.strip()]
    has_timeout = any(contains_timeout_error(err) for err in unique_errors) or contains_timeout_error(
        stderr_text
    )

    if has_timeout:
        if prefers_cjk:
            return "Codex 返回超时，暂时无法给出结果，请稍后重试。"
        return "Codex returned a timeout and could not complete the request. Please try again later."

    detail = ""
    if unique_errors:
        detail = unique_errors[0]
    elif stderr_text.strip():
        detail = stderr_text.strip()

    if prefers_cjk:
        if exit_code is not None and exit_code != 0:
            prefix = f"Codex 返回错误（exit={exit_code}）。"
        else:
            prefix = "Codex 返回错误。"
        if detail:
            return f"{prefix}\n{detail[:1000]}"
        return f"{prefix} 请稍后重试。"

    if exit_code is not None and exit_code != 0:
        prefix = f"Codex failed (exit={exit_code})."
    else:
        prefix = "Codex failed."
    if detail:
        return f"{prefix}\n{detail[:1000]}"
    return f"{prefix} Please try again later."


def run_codex(
    prompt: str,
    reason: str = "primary",
    image_inputs: list[str] | None = None,
) -> tuple[str, bool]:
    codex_bin_runtime, runtime_path = resolve_codex_runtime(CODEX_BIN)
    cmd = [
        codex_bin_runtime,
        "-a",
        "never",
        "exec",
        "--ephemeral",
        "--json",
        "--color",
        "never",
        "--sandbox",
        CODEX_SANDBOX,
        "--skip-git-repo-check",
        "-C",
        CODEX_WORKDIR,
    ]
    if CODEX_MODEL:
        cmd.extend(["-m", CODEX_MODEL])
    for add_dir in CODEX_ADD_DIRS:
        cmd.extend(["--add-dir", add_dir])
    for image_path in image_inputs or []:
        img = str(image_path or "").strip()
        if img:
            cmd.extend(["--image", img])
    # Read prompt from stdin to avoid command-line argument truncation/encoding issues.
    cmd.append("-")

    prompt_preview = preview_text(prompt, TEXT_PREVIEW_CHARS)
    image_count = len(image_inputs or [])
    log(
        f"[codex] start reason={reason} model={CODEX_MODEL or '(default)'} sandbox={CODEX_SANDBOX} "
        f"add_dirs={len(CODEX_ADD_DIRS)} image_inputs={image_count} prompt_chars={len(prompt)} "
        f"prompt_preview={prompt_preview!r}"
    )
    debug_payload: dict[str, Any] = {
        "kind": "codex_exec",
        "reason": reason,
        "ts": now_ts(),
        "model": CODEX_MODEL or "",
        "sandbox": CODEX_SANDBOX,
        "workdir": CODEX_WORKDIR,
        "add_dirs": CODEX_ADD_DIRS,
        "image_inputs": image_inputs or [],
        "command": cmd,
        "prompt_chars": len(prompt),
        "prompt": clip_debug_text(prompt),
    }
    run_env = os.environ.copy()
    if runtime_path:
        run_env["PATH"] = runtime_path
    codex_path = Path(codex_bin_runtime).expanduser()
    codex_dir = codex_path.parent
    if str(codex_dir) not in {"", "."}:
        existing_path = run_env.get("PATH", "")
        path_items = [p for p in existing_path.split(os.pathsep) if p]
        if str(codex_dir) not in path_items:
            run_env["PATH"] = (
                f"{codex_dir}{os.pathsep}{existing_path}" if existing_path else str(codex_dir)
            )

    started_at = time.time()
    try:
        proc = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            env=run_env,
            timeout=CODEX_TIMEOUT_SECONDS,
        )
    except FileNotFoundError:
        path_text = run_env.get("PATH", "")
        path_preview = path_text if len(path_text) <= 300 else path_text[:300] + "..."
        log(f"[codex] binary missing: {codex_bin_runtime} path={path_preview!r}", level="ERROR")
        debug_payload["status"] = "binary_missing"
        debug_payload["path"] = path_text
        debug_payload["error"] = f"Codex binary not found: {codex_bin_runtime}"
        write_codex_debug_echo(debug_payload)
        return f"Codex binary not found: {codex_bin_runtime}", False
    except subprocess.TimeoutExpired as exc:
        duration_ms = int((time.time() - started_at) * 1000)
        timeout_reply = (
            "Codex 执行超时，任务已中断。请缩小问题范围后重试。"
            if contains_cjk(prompt)
            else "Codex timed out and was terminated. Please retry with a smaller request."
        )
        log(
            f"[codex] timeout reason={reason} timeout_s={CODEX_TIMEOUT_SECONDS} "
            f"duration_ms={duration_ms}",
            level="ERROR",
        )
        debug_payload.update(
            {
                "status": "timeout",
                "duration_ms": duration_ms,
                "timeout_seconds": CODEX_TIMEOUT_SECONDS,
                "stdout": clip_debug_text(exc.stdout or ""),
                "stderr": clip_debug_text(exc.stderr or ""),
                "final_ok": False,
                "final_reply": clip_debug_text(timeout_reply),
            }
        )
        write_codex_debug_echo(debug_payload)
        return timeout_reply, False

    duration_ms = int((time.time() - started_at) * 1000)
    stdout_text = proc.stdout or ""
    stderr_text = proc.stderr or ""
    stdout_len = len(stdout_text)
    stderr_len = len(stderr_text)
    log(
        f"[codex] completed exit={proc.returncode} duration_ms={duration_ms} "
        f"stdout_chars={stdout_len} stderr_chars={stderr_len}"
    )

    reply, errors = parse_codex_output(stdout_text)
    final_reply = ""
    ok = False
    if proc.returncode == 0 and reply:
        log(f"[codex] parsed assistant message chars={len(reply)}")
        final_reply = reply
        ok = True
    elif proc.returncode == 0 and not reply:
        log("[codex] exit=0 but no assistant message found in event stream", level="WARN")
        if errors:
            final_reply = format_codex_failure_reply(prompt, errors, exit_code=0, stderr_text=stderr_text)
        else:
            if contains_cjk(prompt):
                final_reply = "Codex 已完成执行，但没有返回可用的最终回复。"
            else:
                final_reply = "Codex completed but no final assistant message was captured."
        ok = False
    elif errors:
        log(
            f"[codex] failed with parsed errors count={len(errors)} first={errors[0]!r}",
            level="ERROR",
        )
        final_reply = format_codex_failure_reply(
            prompt,
            errors,
            exit_code=proc.returncode,
            stderr_text=stderr_text,
        )
        ok = False
    else:
        stderr = stderr_text.strip()
        fallback = stderr or "Unknown Codex error."
        tail = fallback[-1000:]
        log(f"[codex] failed without parsed error; stderr_tail={tail!r}", level="ERROR")
        final_reply = format_codex_failure_reply(
            prompt,
            [],
            exit_code=proc.returncode,
            stderr_text=tail,
        )
        ok = False

    debug_payload.update(
        {
            "status": "completed",
            "duration_ms": duration_ms,
            "exit_code": proc.returncode,
            "stdout_chars": stdout_len,
            "stderr_chars": stderr_len,
            "stdout": clip_debug_text(stdout_text),
            "stderr": clip_debug_text(stderr_text),
            "parsed_reply_chars": len(reply),
            "parsed_reply": clip_debug_text(reply),
            "parsed_errors": errors[:20],
            "final_ok": ok,
            "final_reply": clip_debug_text(final_reply),
        }
    )
    write_codex_debug_echo(debug_payload)
    return final_reply, ok


def process_update(update: dict[str, Any], allowlist: set[str] | None) -> None:
    callback = update.get("callback_query")
    if isinstance(callback, dict):
        callback_id = str(callback.get("id", "")).strip()
        data = str(callback.get("data", "")).strip()
        from_user = callback.get("from") or {}
        user_id = str(from_user.get("id", "")).strip()
        message = callback.get("message") or {}
        chat = message.get("chat") or {}
        chat_id = chat.get("id")
        message_id = message.get("message_id")

        log(
            f"[telegram] callback update_id={update.get('update_id')} chat_id={chat_id} "
            f"user_id={user_id} data={preview_text(data, 80)!r}"
        )

        if allowlist is not None and user_id not in allowlist:
            if callback_id:
                answer_callback_query(callback_id, "Unauthorized user.", show_alert=True)
            if chat_id:
                send_text(chat_id, "Unauthorized user.")
            log(f"[security] blocked callback user_id={user_id}")
            return

        if not chat_id or not message_id or not callback_id:
            return

        parsed_model = parse_models_callback_data(data)
        if parsed_model:
            action, value, source_page = parsed_model
            if action == "list":
                page = max(0, value)
                edit_models_menu(chat_id, int(message_id), page=page)
                answer_callback_query(callback_id)
                return

            if action == "add":
                input_key = build_model_input_key(chat_id, user_id)
                MODEL_INPUT_PENDING[input_key] = now_ts()
                edit_models_menu(
                    chat_id,
                    int(message_id),
                    page=max(0, source_page),
                    notice="请直接发送模型名（纯文本），我会先验证再加入 config.json 的 codex_models。发送 /cancel 可取消。",
                )
                answer_callback_query(callback_id, "请发送模型名")
                return

            if action == "default":
                ok, note = set_active_codex_model("")
                notice = "已切换到 Codex 默认模型。"
                if not ok and note:
                    notice = f"{notice}\n{note}"
                elif note:
                    notice = f"{notice}\n{note}"
                edit_models_menu(chat_id, int(message_id), page=max(0, source_page), notice=notice)
                answer_callback_query(callback_id, "已切换默认模型")
                log("[model] switched to default via_menu=True")
                return

            if action == "set":
                models = load_models_from_config()
                idx = value
                if idx < 0 or idx >= len(models):
                    edit_models_menu(
                        chat_id,
                        int(message_id),
                        page=max(0, source_page),
                        notice="模型列表已变化，请重新选择。",
                    )
                    answer_callback_query(callback_id, "模型不存在")
                    return
                target_model = models[idx]
                if target_model == CODEX_MODEL:
                    edit_models_menu(
                        chat_id,
                        int(message_id),
                        page=max(0, source_page),
                        notice=f"当前已使用模型 {target_model}。",
                    )
                    answer_callback_query(callback_id, "已是当前模型")
                    return
                ok, note = set_active_codex_model(target_model)
                notice = f"已切换到模型 {target_model}。"
                if not ok and note:
                    notice = f"{notice}\n{note}"
                elif note:
                    notice = f"{notice}\n{note}"
                edit_models_menu(chat_id, int(message_id), page=max(0, source_page), notice=notice)
                answer_callback_query(callback_id, f"已切换到 {target_model}")
                log(f"[model] switched to {target_model} via_menu=True")
                return

            answer_callback_query(callback_id, "无效操作。", show_alert=False)
            return

        parsed_session = parse_sessions_callback_data(data)
        if not parsed_session:
            answer_callback_query(callback_id, "无效操作。", show_alert=False)
            return

        action, value, source_page = parsed_session
        session_key = build_session_key(chat_id, user_id)
        session = get_or_create_session(session_key)

        if action == "list":
            page = max(0, value)
            edit_sessions_menu(chat_id, int(message_id), session, page=page)
            answer_callback_query(callback_id)
            return

        if action == "pick":
            target_no = value
            if target_no <= 0:
                answer_callback_query(callback_id, "会话号无效。", show_alert=False)
                return
            detail_payload = build_session_actions_payload(
                session=session,
                target_session_no=target_no,
                source_page=max(0, source_page),
                show_history=False,
            )
            if not detail_payload:
                active = get_or_create_session(session_key)
                edit_sessions_menu(
                    chat_id,
                    int(message_id),
                    active,
                    page=max(0, source_page),
                    notice="未找到该会话。",
                )
                answer_callback_query(callback_id, "未找到该会话。")
                return
            detail_text, detail_markup = detail_payload
            edit_inline_menu_message(
                chat_id=chat_id,
                message_id=int(message_id),
                text=detail_text,
                reply_markup=detail_markup,
            )
            answer_callback_query(callback_id)
            return

        if action == "view":
            target_no = value
            if target_no <= 0:
                answer_callback_query(callback_id, "会话号无效。", show_alert=False)
                return
            detail_payload = build_session_actions_payload(
                session=session,
                target_session_no=target_no,
                source_page=max(0, source_page),
                show_history=True,
            )
            if not detail_payload:
                active = get_or_create_session(session_key)
                edit_sessions_menu(
                    chat_id,
                    int(message_id),
                    active,
                    page=max(0, source_page),
                    notice="未找到该会话。",
                )
                answer_callback_query(callback_id, "未找到该会话。")
                return
            detail_text, detail_markup = detail_payload
            edit_inline_menu_message(
                chat_id=chat_id,
                message_id=int(message_id),
                text=detail_text,
                reply_markup=detail_markup,
            )
            answer_callback_query(callback_id, "已显示历史")
            return

        if action == "switch":
            target_no = value
            if target_no <= 0:
                answer_callback_query(callback_id, "会话号无效。", show_alert=False)
                return

            switched, status = switch_session(session_key, target_no)
            if status == "already_active":
                active = get_or_create_session(session_key)
                detail_payload = build_session_actions_payload(
                    session=active,
                    target_session_no=target_no,
                    source_page=max(0, source_page),
                    show_history=False,
                    notice="当前已在该会话。",
                )
                if detail_payload:
                    detail_text, detail_markup = detail_payload
                    edit_inline_menu_message(
                        chat_id=chat_id,
                        message_id=int(message_id),
                        text=detail_text,
                        reply_markup=detail_markup,
                    )
                else:
                    edit_sessions_menu(
                        chat_id,
                        int(message_id),
                        active,
                        page=max(0, source_page),
                        notice="当前已在该会话。",
                    )
                answer_callback_query(callback_id, "已在当前会话")
                return

            if status == "not_found" or not switched:
                active = get_or_create_session(session_key)
                edit_sessions_menu(
                    chat_id,
                    int(message_id),
                    active,
                    page=max(0, source_page),
                    notice="未找到该会话。",
                )
                answer_callback_query(callback_id, "未找到该会话。")
                return

            entries = list_session_entries(switched)
            display_map = build_session_display_no_map(entries)
            target_display_no = display_map.get(target_no, 1)
            target_idx = max(0, target_display_no - 1)
            page = target_idx // max(1, SESSIONS_PAGE_SIZE)
            switched_title = get_session_title(switched, fallback_no=target_no)
            edit_sessions_menu(
                chat_id,
                int(message_id),
                switched,
                page=page,
                notice=f"已切换到会话 {target_display_no} {switched_title}。",
            )
            answer_callback_query(callback_id, f"已切换到会话 {target_display_no}")
            log(f"[session] switched key={session_key} target_no={target_no} via_menu=True")
            return

        if action == "del":
            target_no = value
            if target_no <= 0:
                answer_callback_query(callback_id, "会话号无效。", show_alert=False)
                return

            updated, status, current_no_after_delete = delete_session_history(session_key, target_no)
            if status == "not_found" or not updated:
                active = get_or_create_session(session_key)
                edit_sessions_menu(
                    chat_id,
                    int(message_id),
                    active,
                    page=max(0, source_page),
                    notice="未找到该会话。",
                )
                answer_callback_query(callback_id, "未找到该会话。")
                return

            entries = list_session_entries(updated)
            display_map = build_session_display_no_map(entries)
            current_title = get_session_title(updated, fallback_no=safe_int(current_no_after_delete, 1))
            current_display_no = display_map.get(safe_int(current_no_after_delete, 1), 1)
            notice = "已删除该会话历史。"
            if status == "deleted_active_switched":
                notice = f"已删除并切换到会话 {current_display_no} {current_title}。"
            elif status == "cleared_active":
                notice = f"已清空会话 {current_display_no} 的历史记录。"

            edit_sessions_menu(
                chat_id,
                int(message_id),
                updated,
                page=max(0, source_page),
                notice=notice,
            )
            answer_callback_query(callback_id, "已删除")
            log(f"[session] deleted key={session_key} target_no={target_no} via_menu=True status={status}")
            return

        answer_callback_query(callback_id, "无效操作。", show_alert=False)
        return

    message = update.get("message")
    if not isinstance(message, dict):
        return

    chat = message.get("chat") or {}
    from_user = message.get("from") or {}
    chat_id = chat.get("id")
    user_id = str(from_user.get("id", "")).strip()
    text_message = str(message.get("text", "")).strip()
    caption_text = str(message.get("caption", "")).strip()

    if allowlist is not None and user_id not in allowlist:
        if chat_id:
            send_text(chat_id, "Unauthorized user.")
        log(f"[security] blocked user_id={user_id}")
        return

    message_media, image_inputs, media_warnings, transcript_lines = prepare_message_media(message)
    text = text_message or caption_text
    text = merge_user_text_and_transcript(text, transcript_lines)
    if not text and message_media:
        text = build_media_only_user_text(message_media)

    has_speech_media = any(
        str(item.get("kind", "")).lower() in {"voice", "audio"} for item in message_media
    )

    if not chat_id or (not text and not message_media):
        log("[telegram] skip update: missing chat_id and usable content", level="WARN")
        return

    text_preview = preview_text(text or "(media-only)", TEXT_PREVIEW_CHARS)
    log(
        f"[telegram] inbound update_id={update.get('update_id')} chat_id={chat_id} "
        f"user_id={user_id} chars={len(text)} media={len(message_media)} text_preview={text_preview!r}"
    )
    if media_warnings:
        log(f"[telegram] media warnings count={len(media_warnings)} first={media_warnings[0]!r}", "WARN")

    if has_speech_media and not transcript_lines and not text_message and not caption_text:
        send_text(
            chat_id,
            "收到语音/音频，但本地转写失败。"
            "请检查依赖（faster-whisper）和模型配置后重试。",
        )
        return

    requested_file = parse_local_file_send_request(text_message)
    if requested_file:
        ok, err = try_send_local_file(chat_id, requested_file)
        if not ok and err:
            send_text(chat_id, err)
        return

    session_key = build_session_key(chat_id, user_id)
    session = get_or_create_session(session_key)
    session_no = int(session.get("session_no", 1))
    session_title = get_session_title(session, fallback_no=session_no)
    history_messages = session.get("messages", [])
    if not isinstance(history_messages, list):
        history_messages = []
    history_count = len(history_messages)
    archived_sessions = sanitize_archived_sessions(session.get("archived_sessions", []))
    session["archived_sessions"] = archived_sessions
    archived_count = len(archived_sessions)
    cmd = telegram_command_name(text_message)
    model_input_key = build_model_input_key(chat_id, user_id)
    awaiting_model_input = model_input_key in MODEL_INPUT_PENDING

    if awaiting_model_input and cmd == "cancel":
        MODEL_INPUT_PENDING.pop(model_input_key, None)
        send_text(chat_id, "已取消手动输入模型。")
        return

    if awaiting_model_input and text_message and not cmd:
        with typing_indicator(chat_id):
            ok, detail = verify_and_register_model(text_message.strip())
        if ok:
            MODEL_INPUT_PENDING.pop(model_input_key, None)
            send_text(chat_id, f"{detail}\n发送 /models 可查看并切换。")
        else:
            send_text(chat_id, f"模型验证失败：{detail}\n请重新发送模型名，或 /cancel 取消。")
        return

    if cmd == "start":
        send_text(
            chat_id,
            "Ready. Send a prompt and I will call local codex CLI.\n"
            "支持文本与图片/文件输入。\n"
            "Commands: /new /session /sessions /history /models /forget",
        )
        return

    if cmd == "new":
        fresh = reset_session(session_key)
        send_text(chat_id, f"已新建会话 #{fresh.get('session_no', 1)}，历史上下文已清空。")
        log(f"[session] reset key={session_key} new_no={fresh.get('session_no', 1)}")
        return

    if cmd == "forget":
        session["messages"] = []
        session["title"] = f"会话 #{session_no}"
        session["updated_at"] = now_ts()
        save_session_store()
        send_text(chat_id, f"已清空当前会话 #{session_no} 的历史上下文。")
        log(f"[session] forget key={session_key} no={session_no}")
        return

    if cmd == "sessions":
        send_sessions_menu(chat_id, session, page=0)
        return

    if cmd == "models":
        parts = text_message.split(maxsplit=2)
        sub = parts[1].strip().lower() if len(parts) >= 2 else ""
        if sub in {"add", "input"}:
            if len(parts) < 3 or not parts[2].strip():
                MODEL_INPUT_PENDING[model_input_key] = now_ts()
                send_text(chat_id, "请直接发送模型名（纯文本），我会先验证再加入列表。发送 /cancel 可取消。")
                return
            candidate_model = parts[2].strip()
            with typing_indicator(chat_id):
                ok, detail = verify_and_register_model(candidate_model)
            if ok:
                MODEL_INPUT_PENDING.pop(model_input_key, None)
                send_text(chat_id, f"{detail}\n发送 /models 可查看并切换。")
            else:
                MODEL_INPUT_PENDING[model_input_key] = now_ts()
                send_text(chat_id, f"模型验证失败：{detail}\n请重新发送模型名，或 /cancel 取消。")
            return
        send_models_menu(chat_id, page=0)
        return

    if cmd == "session":
        send_text(
            chat_id,
            f"session_key={session_key}\n"
            f"session_no={session_no}\n"
            f"session_title={session_title}\n"
            f"history_messages={history_count}\n"
            f"archived_sessions={archived_count}\n"
            f"codex_model={CODEX_MODEL or '(Codex default)'}\n"
            f"session_enabled={SESSION_ENABLED}\n"
            f"session_scope={SESSION_SCOPE}\n"
            f"session_max_messages={SESSION_MAX_MESSAGES}\n"
            f"session_archive_max={SESSION_ARCHIVE_MAX}\n"
            f"session_title_max_chars={SESSION_TITLE_MAX_CHARS}\n"
            f"media_input_enabled={MEDIA_INPUT_ENABLED}\n"
            f"media_input_dir={MEDIA_INPUT_DIR}\n"
            f"media_input_max_bytes={MEDIA_INPUT_MAX_BYTES}\n"
            f"media_input_max_files={MEDIA_INPUT_MAX_FILES}\n"
            f"local_transcribe_enabled={LOCAL_TRANSCRIBE_ENABLED}\n"
            f"local_transcribe_model={LOCAL_TRANSCRIBE_MODEL}\n"
            f"local_transcribe_language={LOCAL_TRANSCRIBE_LANGUAGE}\n"
            f"debug_echo_enabled={DEBUG_ECHO_ENABLED}\n"
            f"debug_echo_dir={DEBUG_ECHO_DIR}",
        )
        return

    if cmd == "switch":
        send_text(chat_id, "请使用 /sessions 打开会话菜单后点击会话进行切换。")
        return

    if cmd == "history":
        parts = text_message.split()
        requested = 10
        if len(parts) >= 2:
            arg = parts[1].strip()
            if not arg.isdigit():
                send_text(chat_id, "用法: /history [N]")
                return
            requested = safe_int(arg, 10)
        requested = max(1, min(requested, max(5, SESSION_MAX_MESSAGES)))

        source_session_no = session_no
        source_messages = history_messages
        source_session_title = session_title
        source_title = f"会话 #{session_no} [{source_session_title}]"

        if not source_messages:
            archived = latest_archived_session(session)
            if archived:
                source_session_no = max(1, safe_int(archived.get("session_no"), session_no - 1))
                source_messages = archived.get("messages", [])
                if not isinstance(source_messages, list):
                    source_messages = []
                source_session_title = normalize_session_title(str(archived.get("title", ""))) or (
                    build_session_title(source_messages, session_no=source_session_no)
                )
                source_title = (
                    f"会话 #{session_no} [{session_title}] 当前无消息，"
                    f"已展示上一会话 #{source_session_no} [{source_session_title}]"
                )

        tail = source_messages[-requested:]
        if not tail:
            send_text(chat_id, f"会话 #{session_no} 当前没有历史消息。")
            return

        lines = [f"{source_title} 最近 {len(tail)} 条："]
        for idx, item in enumerate(tail, start=1):
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "unknown"))
            body = preview_text(compose_session_entry_text(item), 200)
            lines.append(f"{idx}. [{role}] {body}")

        send_text(chat_id, "\n".join(lines))
        return

    user_mentions_meta = contains_meta_noise(text)

    if not SESSION_ENABLED:
        direct_prompt = (
            text
            if not message_media
            else build_clean_prompt(text, current_media=message_media)
        )
        with typing_indicator(chat_id):
            reply, ok = run_codex(
                direct_prompt,
                reason="non_session_direct",
                image_inputs=image_inputs,
            )
            if ok and (not user_mentions_meta) and contains_meta_noise(reply):
                log("[guard] meta-noise detected in non-session reply; retrying with clean prompt", "WARN")
                retry_reply, retry_ok = run_codex(
                    build_clean_prompt(text, current_media=message_media),
                    reason="non_session_clean_retry",
                    image_inputs=image_inputs,
                )
                reply, ok = retry_reply, retry_ok
            if ok and (not user_mentions_meta) and contains_generic_nonanswer(reply):
                log("[guard] generic/non-answer detected in non-session reply; retrying strict prompt", "WARN")
                retry_prompt = build_strict_retry_prompt(
                    session_key=session_key,
                    session_no=session_no,
                    history=[],
                    user_text=text,
                    current_media=message_media,
                )
                retry_reply, retry_ok = run_codex(
                    retry_prompt,
                    reason="non_session_strict_retry",
                    image_inputs=image_inputs,
                )
                reply, ok = retry_reply, retry_ok
            if ok and (not user_mentions_meta) and contains_generic_nonanswer(reply):
                log("[guard] strict retry still generic in non-session reply; retrying chat prompt", "WARN")
                retry_prompt = build_chat_retry_prompt(
                    session_key=session_key,
                    session_no=session_no,
                    history=[],
                    user_text=text,
                    current_media=message_media,
                )
                retry_reply, retry_ok = run_codex(
                    retry_prompt,
                    reason="non_session_chat_retry",
                    image_inputs=image_inputs,
                )
                reply, ok = retry_reply, retry_ok

        send_reply_with_local_files(chat_id, reply)
        log(f"[session] disabled key={session_key} codex_ok={ok}")
        return

    context_prompt = build_codex_prompt(
        session_key=session_key,
        session_no=session_no,
        history=history_messages,
        user_text=text,
        current_media=message_media,
    )
    log(
        f"[session] using context key={session_key} no={session_no} "
        f"history_messages={history_count}"
    )

    with typing_indicator(chat_id):
        reply, ok = run_codex(
            context_prompt,
            reason="session_context_main",
            image_inputs=image_inputs,
        )
        if ok and (not user_mentions_meta) and contains_meta_noise(reply):
            log("[guard] meta-noise detected in contextual reply; retrying without history", "WARN")
            retry_reply, retry_ok = run_codex(
                build_clean_prompt(text, current_media=message_media),
                reason="session_clean_retry",
                image_inputs=image_inputs,
            )
            reply, ok = retry_reply, retry_ok

        if ok and (not user_mentions_meta) and contains_generic_nonanswer(reply):
            log("[guard] generic/non-answer detected in contextual reply; retrying strict prompt", "WARN")
            retry_reply, retry_ok = run_codex(
                build_strict_retry_prompt(
                    session_key=session_key,
                    session_no=session_no,
                    history=history_messages,
                    user_text=text,
                    current_media=message_media,
                ),
                reason="session_strict_retry",
                image_inputs=image_inputs,
            )
            reply, ok = retry_reply, retry_ok
        if ok and (not user_mentions_meta) and contains_generic_nonanswer(reply):
            log("[guard] strict retry still generic in contextual reply; retrying chat prompt", "WARN")
            retry_reply, retry_ok = run_codex(
                build_chat_retry_prompt(
                    session_key=session_key,
                    session_no=session_no,
                    history=history_messages,
                    user_text=text,
                    current_media=message_media,
                ),
                reason="session_chat_retry",
                image_inputs=image_inputs,
            )
            reply, ok = retry_reply, retry_ok

    history_reply = send_reply_with_local_files(chat_id, reply)

    append_session_message(session, "user", text, media=message_media)
    reply_has_meta = contains_meta_noise(history_reply)
    reply_is_generic = contains_generic_nonanswer(history_reply)
    if ok and (user_mentions_meta or (not reply_has_meta and not reply_is_generic)):
        append_session_message(session, "assistant", history_reply)
    else:
        log(
            f"[session] assistant reply not persisted key={session_key} "
            f"(codex_ok={ok}, meta_noise={reply_has_meta}, generic_nonanswer={reply_is_generic})",
            "WARN",
        )

def main() -> None:
    if not BOT_TOKEN:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN first.")

    ensure_parent(STATE_FILE)
    ensure_parent(SESSION_STORE_FILE)
    acquire_instance_lock()
    log(f"[runtime] acquired instance lock file={INSTANCE_LOCK_FILE}")

    # Force polling mode; no public webhook is required.
    try:
        tg_call("deleteWebhook", {"drop_pending_updates": False})
        log("[telegram] deleteWebhook ok")
    except Exception as exc:  # noqa: BLE001
        log(f"[warn] deleteWebhook failed: {exc}")

    ensure_telegram_commands()

    offset = load_offset()
    startup_codex_bin, startup_path = resolve_codex_runtime(CODEX_BIN)
    log(f"[telegram] start polling from offset={offset}")
    log(
        f"[codex] bin={startup_codex_bin} workdir={CODEX_WORKDIR} sandbox={CODEX_SANDBOX}"
    )
    if startup_path:
        log(f"[codex] path={startup_path!r}")
    if CODEX_ADD_DIRS:
        log(f"[codex] add_dirs={CODEX_ADD_DIRS}")
    log(f"[codex] timeout_seconds={CODEX_TIMEOUT_SECONDS}")
    log(f"[telegram] typing heartbeat interval={TYPING_HEARTBEAT_SECONDS}s")
    log(f"[log] file={LOG_FILE} stdout={LOG_STDOUT} text_preview_chars={TEXT_PREVIEW_CHARS}")
    log(
        f"[debug] echo_enabled={DEBUG_ECHO_ENABLED} "
        f"echo_dir={DEBUG_ECHO_DIR} echo_max_chars={DEBUG_ECHO_MAX_CHARS}"
    )
    log(
        f"[session] enabled={SESSION_ENABLED} scope={SESSION_SCOPE} "
        f"max_messages={SESSION_MAX_MESSAGES} archive_max={SESSION_ARCHIVE_MAX} "
        f"title_max_chars={SESSION_TITLE_MAX_CHARS} "
        f"store={SESSION_STORE_FILE}"
    )
    log(
        f"[media] input_enabled={MEDIA_INPUT_ENABLED} input_dir={MEDIA_INPUT_DIR} "
        f"input_max_bytes={MEDIA_INPUT_MAX_BYTES} input_max_files={MEDIA_INPUT_MAX_FILES} "
        f"history_max_items={MEDIA_HISTORY_MAX_ITEMS}"
    )
    log(
        f"[media] local_transcribe_enabled={LOCAL_TRANSCRIBE_ENABLED} "
        f"model={LOCAL_TRANSCRIBE_MODEL} lang={LOCAL_TRANSCRIBE_LANGUAGE or '(auto)'} "
        f"device={LOCAL_TRANSCRIBE_DEVICE} compute_type={LOCAL_TRANSCRIBE_COMPUTE_TYPE} "
        f"vad={LOCAL_TRANSCRIBE_VAD}"
    )

    last_allowlist_fingerprint: tuple[str, ...] | None = None
    warned_allow_all = False

    while True:
        try:
            allowlist = load_allowlist()
            if allowlist is None:
                if not warned_allow_all:
                    log(
                        "[security] allowlist missing/invalid in config.json; allow all users",
                        level="WARN",
                    )
                    warned_allow_all = True
            else:
                warned_allow_all = False
                fingerprint = tuple(sorted(allowlist))
                if fingerprint != last_allowlist_fingerprint:
                    log(f"[security] allowlist loaded count={len(allowlist)}")
                    last_allowlist_fingerprint = fingerprint

            updates = tg_call(
                "getUpdates",
                {
                    "timeout": POLL_TIMEOUT_SECONDS,
                    "offset": offset + 1,
                    "allowed_updates": ["message", "callback_query"],
                },
            )
            if updates:
                first_id = updates[0].get("update_id")
                last_id = updates[-1].get("update_id")
                log(
                    f"[telegram] polled updates count={len(updates)} first_id={first_id} last_id={last_id}"
                )
            for update in updates:
                update_id = int(update.get("update_id", 0))
                try:
                    process_update(update, allowlist)
                except Exception as exc:  # noqa: BLE001
                    log(f"[error] process_update failed: {exc}")
                    log(traceback.format_exc())
                finally:
                    if update_id > offset:
                        offset = update_id
                        save_offset(offset)
        except urllib.error.URLError as exc:
            log(f"[warn] network error: {exc}")
            time.sleep(2)
        except TelegramConflictError as exc:
            log(f"[fatal] polling conflict: {exc}", level="ERROR")
            log(
                "[fatal] another bot instance is already polling getUpdates; exit current process.",
                level="ERROR",
            )
            raise SystemExit(2)
        except Exception as exc:  # noqa: BLE001
            log(f"[error] polling loop failed: {exc}")
            log(traceback.format_exc())
            time.sleep(2)


if __name__ == "__main__":
    main()
