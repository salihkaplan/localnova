"""Allowlisted Windows application actions for NOVA.

Opening and closing are restricted to applications explicitly listed here.
Closing sends WM_CLOSE to visible top-level windows; it does not force-kill
processes. File Explorer is handled specially so the Windows shell/taskbar is
never targeted.
"""
import os
import subprocess


APP_COMMANDS = {
    "chrome": ["cmd", "/c", "start", "", "chrome"],
    "spotify": ["cmd", "/c", "start", "", "spotify"],
    "notepad": ["notepad.exe"],
    "explorer": ["explorer.exe"],
    "calculator": ["calc.exe"],
}

APP_NAMES = {
    "chrome": "Chrome",
    "spotify": "Spotify",
    "notepad": "Not Defteri",
    "explorer": "Dosya Gezgini",
    "calculator": "Hesap Makinesi",
}

# Process executable names are fixed allowlist values, never user-supplied.
APP_PROCESSES = {
    "chrome": {"chrome.exe"},
    "spotify": {"spotify.exe"},
    "notepad": {"notepad.exe"},
    "explorer": {"explorer.exe"},
    "calculator": {"calculatorapp.exe", "calc.exe"},
}

# Explorer hosts the desktop and taskbar as well as File Explorer windows.
# Only these window classes are considered closable File Explorer windows.
EXPLORER_WINDOW_CLASSES = {"cabinetwclass", "explorewclass"}
WM_CLOSE = 0x0010


def _close_app_windows(app):
    """Request graceful closure of visible windows for an allowlisted app."""
    if os.name != "nt":
        return False, "Uygulama kapatma yalnızca Windows üzerinde destekleniyor."

    try:
        import ctypes
        from ctypes import wintypes
        import psutil
    except Exception as exc:
        return False, f"Windows pencere API'si başlatılamadı: {type(exc).__name__}"

    process_names = APP_PROCESSES.get(app)
    if not process_names:
        return False, None

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    enum_windows = user32.EnumWindows
    enum_proc_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    enum_windows.argtypes = [enum_proc_type, wintypes.LPARAM]
    enum_windows.restype = wintypes.BOOL
    is_window_visible = user32.IsWindowVisible
    is_window_visible.argtypes = [wintypes.HWND]
    is_window_visible.restype = wintypes.BOOL
    get_window_thread_process_id = user32.GetWindowThreadProcessId
    get_window_thread_process_id.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    get_window_thread_process_id.restype = wintypes.DWORD
    get_class_name = user32.GetClassNameW
    get_class_name.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    get_class_name.restype = ctypes.c_int
    post_message = user32.PostMessageW
    post_message.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    post_message.restype = wintypes.BOOL

    targets = []

    @enum_proc_type
    def enum_callback(hwnd, _lparam):
        if not is_window_visible(hwnd):
            return True

        pid = wintypes.DWORD(0)
        get_window_thread_process_id(hwnd, ctypes.byref(pid))
        if not pid.value:
            return True

        try:
            process_name = psutil.Process(pid.value).name().lower()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return True
        except Exception:
            return True

        if process_name not in process_names:
            return True

        if app == "explorer":
            class_buffer = ctypes.create_unicode_buffer(256)
            get_class_name(hwnd, class_buffer, len(class_buffer))
            if class_buffer.value.lower() not in EXPLORER_WINDOW_CLASSES:
                return True

        targets.append(hwnd)
        return True

    try:
        enum_windows(enum_callback, 0)
    except Exception as exc:
        return False, f"Uygulama pencereleri taranamadı: {type(exc).__name__}"

    if not targets:
        return False, f"{APP_NAMES.get(app, app)} için açık bir pencere bulamadım."

    closed_requests = 0
    for hwnd in targets:
        try:
            if post_message(hwnd, WM_CLOSE, 0, 0):
                closed_requests += 1
        except Exception:
            continue

    if closed_requests:
        return True, f"{APP_NAMES.get(app, app)} için {closed_requests} pencereye normal kapatma isteği gönderildi."
    return False, f"{APP_NAMES.get(app, app)} penceresine kapatma isteği gönderilemedi."


def execute_app_action(command):
    action = command.get("action")
    app = normalize_app_name(command.get("app")) or command.get("app")

    if not isinstance(app, str) or app not in APP_COMMANDS:
        return False, None

    if action == "open_app":
        try:
            subprocess.Popen(APP_COMMANDS[app])
            return True, f"✓ {APP_NAMES.get(app, app)} açma komutu gönderildi."
        except Exception as exc:
            return False, f"✗ {APP_NAMES.get(app, app)} açılamadı: {type(exc).__name__}: {exc}"

    if action == "close_app":
        return _close_app_windows(app)

    return False, None


APP_ALIASES = {
    "chrome": "chrome", "google chrome": "chrome",
    "spotify": "spotify",
    "notepad": "notepad", "not defteri": "notepad",
    "explorer": "explorer", "dosya gezgini": "explorer",
    "calculator": "calculator", "hesap makinesi": "calculator", "hesap makinesi uygulaması": "calculator",
}

def normalize_app_name(value):
    if not isinstance(value, str):
        return None
    return APP_ALIASES.get(value.strip().lower())

def _running_app_keys():
    try:
        import psutil
    except ImportError:
        return None, "Uygulama durumunu sorgulamak için psutil modülü eksik."
    running = set()
    errors = 0
    for proc in psutil.process_iter(["name"]):
        try:
            name = (proc.info.get("name") or "").lower()
            for app, names in APP_PROCESSES.items():
                if name in names:
                    running.add(app)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            errors += 1
        except Exception:
            errors += 1
    return running, None

def execute_app_status(command):
    app = normalize_app_name(command.get("app"))
    if app not in APP_COMMANDS:
        return False, "Bu uygulama izin verilen uygulama listesinde değil."
    running, error = _running_app_keys()
    if running is None:
        return False, error
    name = APP_NAMES.get(app, app)
    if app == "explorer":
        # explorer.exe normally hosts the desktop, so describe the process rather than a window.
        return True, ("Dosya Gezgini işlemi çalışıyor." if app in running else "Dosya Gezgini işlemi çalışmıyor.")
    return True, (f"{name} şu anda çalışıyor." if app in running else f"{name} şu anda çalışmıyor.")

def execute_list_apps(command=None):
    running, error = _running_app_keys()
    if running is None:
        return False, error
    names = [APP_NAMES[key] for key in APP_COMMANDS if key in running]
    if not names:
        return True, "İzin verilen uygulama listemde çalışan bir uygulama bulamadım."
    return True, "Çalışan desteklenen uygulamalar: " + ", ".join(names) + "."
