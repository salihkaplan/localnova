"""Allowlisted Windows utilities for NOVA. No arbitrary shell commands are accepted."""
from __future__ import annotations

import os
import subprocess
from datetime import datetime
from pathlib import Path

SETTINGS_URIS = {
    "wifi": ("ms-settings:network-wifi", "Wi-Fi ayarları"),
    "network": ("ms-settings:network", "Ağ ayarları"),
    "bluetooth": ("ms-settings:bluetooth", "Bluetooth ayarları"),
    "display": ("ms-settings:display", "Ekran ayarları"),
    "sound": ("ms-settings:sound", "Ses ayarları"),
    "apps": ("ms-settings:appsfeatures", "Uygulamalar ayarları"),
    "default_apps": ("ms-settings:defaultapps", "Varsayılan uygulamalar ayarları"),
    "windows_update": ("ms-settings:windowsupdate", "Windows Update ayarları"),
}


def open_settings(command):
    target = str(command.get("target", "")).strip().lower()
    item = SETTINGS_URIS.get(target)
    if not item:
        return False, "Bu Windows ayarı izin verilen listede değil."
    if os.name != "nt":
        return False, "Windows ayarları yalnızca Windows üzerinde açılabilir."
    uri, label = item
    try:
        os.startfile(uri)  # type: ignore[attr-defined]
        return True, f"{label} açıldı."
    except Exception as exc:
        return False, f"{label} açılamadı: {type(exc).__name__}: {exc}"


def wifi_status(command=None):
    if os.name != "nt":
        return False, "Wi-Fi durumu yalnızca Windows üzerinde sorgulanabilir."
    try:
        result = subprocess.run(
            ["netsh", "wlan", "show", "interfaces"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=8,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        output = (result.stdout or result.stderr or "").strip()
        if result.returncode != 0 or not output:
            return False, "Wi-Fi bağlantı bilgisi alınamadı. Kablosuz ağ adaptörü bulunmayabilir."
        lowered = output.lower()
        if "disconnected" in lowered or "bağlı değil" in lowered or "bağlantı kesildi" in lowered:
            return True, "Wi-Fi şu anda bağlı değil."
        if "connected" in lowered or "bağlı" in lowered:
            return True, "Wi-Fi bağlantısı etkin görünüyor."
        for line in output.splitlines():
            if "state" in line.lower() or "durum" in line.lower():
                return True, "Wi-Fi durumu: " + line.split(":", 1)[-1].strip() + "."
        return True, "Wi-Fi arayüzü bulundu; bağlantı durumu net olarak çözümlenemedi."
    except subprocess.TimeoutExpired:
        return False, "Wi-Fi durumu sorgusu zaman aşımına uğradı."
    except Exception as exc:
        return False, f"Wi-Fi durumu alınamadı: {type(exc).__name__}: {exc}"


def bluetooth_status(command=None):
    """Avoid claiming the Bluetooth radio is on/off when only portable device data is available."""
    if os.name != "nt":
        return False, "Bluetooth durumu yalnızca Windows üzerinde sorgulanabilir."
    return False, (
        "Bu sürüm Windows Bluetooth radyosunun açık veya kapalı olduğunu güvenilir biçimde doğrulayamıyor. "
        "Durumu kontrol etmek için 'Bluetooth ayarlarını aç' diyebilirsin."
    )


def take_screenshot(command=None):
    """Capture the current desktop to Pictures/NOVA_Screenshots."""
    if os.name != "nt":
        return False, "Ekran görüntüsü alma bu sürümde Windows üzerinde destekleniyor."
    try:
        from PIL import ImageGrab
    except ImportError:
        return False, "Ekran görüntüsü için Pillow eksik. Aktif NOVA Python ortamına pillow kurulmalı."
    try:
        pictures = Path.home() / "Pictures"
        if not pictures.is_dir():
            pictures = Path.home() / "Resimler"
        folder = pictures / "NOVA_Screenshots"
        folder.mkdir(parents=True, exist_ok=True)
        filename = "NOVA_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".png"
        target = folder / filename
        ImageGrab.grab(all_screens=True).save(target, "PNG")
        return True, f"Ekran görüntüsü Resimler klasöründeki NOVA_Screenshots klasörüne kaydedildi: {filename}"
    except Exception as exc:
        return False, f"Ekran görüntüsü alınamadı: {type(exc).__name__}: {exc}"


def execute_windows_action(command):
    action = command.get("action") if isinstance(command, dict) else None
    if action == "open_settings":
        return open_settings(command)
    if action == "wifi_status":
        return wifi_status(command)
    if action == "bluetooth_status":
        return bluetooth_status(command)
    if action == "screenshot":
        return take_screenshot(command)
    return False, None
