"""Central allowlisted action dispatcher used by voice and multi-step commands."""
from .apps import execute_app_action, execute_app_status, execute_list_apps
from .browser import execute_browser_action
from .audio import execute_audio_action
from .system_info import execute_system_info
from .windows_tools import execute_windows_action
from .files import execute_file_action, validate_file_command, DESTRUCTIVE_ACTIONS

APP_ACTIONS = {"open_app", "close_app", "app_status", "list_apps"}
BROWSER_ACTIONS = {"open_url", "open_incognito"}
AUDIO_ACTIONS = {"set_volume", "volume_up", "volume_down", "mute_audio", "unmute_audio"}
WINDOWS_ACTIONS = {"open_settings", "wifi_status", "bluetooth_status", "screenshot"}
FILE_ACTIONS = {"list_directory", "find_file", "file_info", "open_path", "copy_file", "move_file", "rename_file", "delete_file"}
SYSTEM_ACTIONS = {"system_status"}
SUPPORTED_ACTIONS = APP_ACTIONS | BROWSER_ACTIONS | AUDIO_ACTIONS | WINDOWS_ACTIONS | FILE_ACTIONS | SYSTEM_ACTIONS


def validate_command(command):
    """Validate an action shape before execution; does not perform side effects."""
    if not isinstance(command, dict):
        return False, "Eylem nesnesi geçersiz."
    action = command.get("action")
    if action not in SUPPORTED_ACTIONS:
        return False, f"Desteklenmeyen eylem: {action!r}."
    if action in {"open_app", "close_app", "app_status"}:
        from .apps import normalize_app_name, APP_COMMANDS
        app = normalize_app_name(command.get("app"))
        if app not in APP_COMMANDS:
            return False, "Uygulama izin verilen listede değil."
    if action in FILE_ACTIONS:
        ok, message = validate_file_command(command)
        if not ok:
            return False, message
    if action == "open_url" and command.get("url") not in {"https://www.youtube.com"}:
        return False, "Bu bağlantı adresi izin verilen listede değil."
    if action == "open_incognito" and command.get("app") != "chrome":
        return False, "Gizli pencere yalnızca Chrome için destekleniyor."
    if action == "set_volume":
        value = command.get("value")
        if not isinstance(value, (int, float)) or not 0 <= value <= 100:
            return False, "Ses seviyesi sıfır ile yüz arasında olmalı."
    if action in {"open_settings"}:
        from .windows_tools import SETTINGS_URIS
        if command.get("target") not in SETTINGS_URIS:
            return False, "Windows ayarı izin verilen listede değil."
    return True, None


def _dispatch_one(command):
    action = command.get("action")
    if action in {"open_app", "close_app"}:
        return execute_app_action(command)
    if action == "app_status":
        return execute_app_status(command)
    if action == "list_apps":
        return execute_list_apps(command)
    if action in BROWSER_ACTIONS:
        return execute_browser_action(command)
    if action in AUDIO_ACTIONS:
        return execute_audio_action(command)
    if action == "system_status":
        return execute_system_info(command)
    if action in WINDOWS_ACTIONS:
        return execute_windows_action(command)
    if action in FILE_ACTIONS:
        return execute_file_action(command)
    return False, None


def dispatch(command):
    """Return (success, message). Multi-action plans are validated before any action runs."""
    if isinstance(command, list):
        if not command or len(command) > 5:
            return False, "Bir komut planı bir ile beş adım arasında olmalı."
        if any(not isinstance(item, dict) for item in command):
            return False, "Komut planında geçersiz bir adım var."
        if any(item.get("action") in DESTRUCTIVE_ACTIONS for item in command):
            return False, "Dosya taşıma, yeniden adlandırma veya silme işlemlerini çok adımlı plandan ayrı yürütmeliyiz."
        for item in command:
            ok, message = validate_command(item)
            if not ok:
                return False, f"Plan doğrulaması başarısız; hiçbir işlem başlatılmadı. {message}"
        messages = []
        completed = 0
        for index, item in enumerate(command, start=1):
            ok, message = _dispatch_one(item)
            if message:
                messages.append(f"Adım {index}: {message}")
            if not ok:
                detail = " ".join(messages)
                return False, f"Plan {completed} adım tamamlandıktan sonra durdu. {detail}".strip()
            completed += 1
        return True, " ".join(messages) if messages else f"{completed} adım tamamlandı."

    ok, message = validate_command(command)
    if not ok:
        return False, message
    return _dispatch_one(command)
