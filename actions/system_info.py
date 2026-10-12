"""Read-only Windows system status queries for NOVA. Requires psutil."""
from __future__ import annotations

import os
import time


def _fmt_bytes(value: int | float) -> str:
    value = float(value)
    units = ("bayt", "kilobayt", "megabayt", "gigabayt", "terabayt")
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "bayt":
                return f"{int(value)} {unit}"
            rounded = round(value, 1)
            amount = str(int(rounded)) if rounded.is_integer() else str(rounded).replace(".", ",")
            return f"{amount} {unit}"
        value /= 1024
    return f"{value:.1f} terabayt"


def _psutil():
    try:
        import psutil
        return psutil
    except ImportError as exc:
        raise RuntimeError(
            "Sistem durumu için psutil modülü eksik. NOVA ortamında "
            "python -m pip install psutil komutunu çalıştır."
        ) from exc


def _cpu(psutil) -> str:
    percent = psutil.cpu_percent(interval=0.25)
    return f"İşlemci kullanımı şu anda yüzde {round(percent)}."


def _memory(psutil) -> str:
    mem = psutil.virtual_memory()
    return (
        f"Bellek kullanımı yüzde {round(mem.percent)}. "
        f"{_fmt_bytes(mem.used)} kullanılıyor, "
        f"{_fmt_bytes(mem.available)} kullanılabilir; toplam {_fmt_bytes(mem.total)}."
    )


def _disks(psutil) -> str:
    # Windows system drive is the most useful first answer; fall back to root.
    system_drive = os.environ.get("SystemDrive", "C:")
    path = system_drive + "\\" if os.name == "nt" else "/"
    try:
        usage = psutil.disk_usage(path)
        return (
            f"Sistem diskinde yüzde {round(usage.percent)} alan dolu. "
            f"Toplam {_fmt_bytes(usage.total)}, boş alan {_fmt_bytes(usage.free)}."
        )
    except Exception:
        usage = psutil.disk_usage("/")
        return (
            f"Sistem diskinde yüzde {round(usage.percent)} alan dolu. "
            f"Toplam {_fmt_bytes(usage.total)}, boş alan {_fmt_bytes(usage.free)}."
        )


def _battery(psutil) -> str:
    try:
        battery = psutil.sensors_battery()
    except Exception:
        battery = None
    if battery is None:
        return "Bu cihazdan batarya bilgisi alınamıyor; masaüstü bilgisayar olabilir."
    state = "şarj oluyor" if battery.power_plugged else "bataryadan çalışıyor"
    return f"Batarya yüzde {round(battery.percent)} seviyesinde ve cihaz {state}."


def _uptime(psutil) -> str:
    seconds = max(0, int(time.time() - psutil.boot_time()))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes = seconds // 60
    pieces = []
    if days:
        pieces.append(f"{days} gün")
    if hours:
        pieces.append(f"{hours} saat")
    if minutes or not pieces:
        pieces.append(f"{minutes} dakika")
    return "Bilgisayar yaklaşık " + " ".join(pieces) + " açık."


def execute_system_info(command):
    """Return (success, spoken status text); performs no system modifications."""
    target = str(command.get("target", "summary")).strip().lower()
    if target == "cpu_temp":
        return True, (
            "İşlemci sıcaklığını güvenilir şekilde okuyamıyorum. Windows bu bilgiyi "
            "her bilgisayarda standart olarak sunmuyor; sıcaklık sensörü entegrasyonunu "
            "ayrı bir aşamada ekleyebiliriz."
        )
    try:
        psutil = _psutil()
        if target == "cpu":
            return True, _cpu(psutil)
        if target == "ram":
            return True, _memory(psutil)
        if target == "disk":
            return True, _disks(psutil)
        if target == "battery":
            return True, _battery(psutil)
        if target == "uptime":
            return True, _uptime(psutil)
        if target == "summary":
            parts = [_cpu(psutil), _memory(psutil), _disks(psutil), _uptime(psutil)]
            battery = _battery(psutil)
            if "batarya bilgisi alınamıyor" not in battery:
                parts.append(battery)
            return True, "Bilgisayar durumu şöyle: " + " ".join(parts)
        return False, f"Bilinmeyen sistem durumu sorgusu: {target}."
    except Exception as exc:
        return False, f"Sistem durumu okunamadı: {type(exc).__name__}: {exc}"
