"""Safe file helpers for NOVA, restricted to common folders under the current user profile."""
from __future__ import annotations

import fnmatch
import os
import shutil
from pathlib import Path

ROOT_ALIASES = {
    "desktop": ("Desktop", "Masaüstü"),
    "documents": ("Documents", "Belgeler"),
    "downloads": ("Downloads", "İndirilenler"),
    "pictures": ("Pictures", "Resimler"),
    "videos": ("Videos", "Videolar"),
    "music": ("Music", "Müzik"),
}
SKIP_DIRS = {".git", "__pycache__", "node_modules", "$recycle.bin", "system volume information"}
DESTRUCTIVE_ACTIONS = {"move_file", "rename_file", "delete_file"}


def _safe_roots():
    home = Path.home().resolve()
    roots = []
    for alias, names in ROOT_ALIASES.items():
        # Prefer the existing English or localized folder; otherwise keep the canonical
        # profile-relative path as an allowed future destination, without broadening to home itself.
        candidates = [home / name for name in names]
        existing = next((path for path in candidates if path.exists() and path.is_dir()), None)
        path = existing or candidates[0]
        roots.append((alias, path.resolve(strict=False)))
    # Some Windows profiles redirect known folders. Include only if the path is still
    # within the user's profile, so redirected external roots are never silently trusted.
    unique = []
    seen = set()
    for alias, path in roots:
        if str(path).lower() not in seen:
            unique.append((alias, path))
            seen.add(str(path).lower())
    return unique


def _resolve_safe_path(value, default_root="documents"):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Dosya veya klasör yolu belirtilmedi.")
    raw = value.strip().strip('"').strip("'")
    roots = _safe_roots()
    if not roots:
        raise ValueError("Kullanıcı klasörleri bulunamadı.")
    aliases = {alias: path for alias, path in roots}
    # Accept Turkish/English common folder aliases as path prefixes.
    normalized = raw.replace("\\", "/")
    first, sep, remainder = normalized.partition("/")
    alias_map = {
        "desktop": "desktop", "masaüstü": "desktop", "masaustu": "desktop",
        "documents": "documents", "belgeler": "documents",
        "downloads": "downloads", "indirilenler": "downloads",
        "pictures": "pictures", "resimler": "pictures",
        "videos": "videos", "videolar": "videos",
        "music": "music", "müzik": "music", "muzik": "music",
    }
    alias = alias_map.get(first.lower())
    if alias:
        base = aliases.get(alias)
        if base is None:
            # Resolve a known folder name under the home directory even if it did not exist yet.
            english, turkish = ROOT_ALIASES[alias]
            base = home_candidate = Path.home() / english
            if not base.exists():
                base = Path.home() / turkish
        candidate = base / remainder if sep else base
    else:
        p = Path(raw).expanduser()
        if p.is_absolute():
            candidate = p
        else:
            base = aliases.get(default_root)
            if base is None:
                raise ValueError("Varsayılan klasör bulunamadı.")
            candidate = base / p
    candidate = candidate.resolve(strict=False)
    allowed = [path for _, path in roots]
    # Also allow not-yet-existing descendants of a known folder.
    for alias_name, names in ROOT_ALIASES.items():
        for name in names:
            p = (Path.home() / name).resolve(strict=False)
            if p.exists() and p.is_dir() and p not in allowed:
                allowed.append(p)
    if not any(candidate == root or root in candidate.parents for root in allowed):
        raise ValueError("Güvenlik nedeniyle yalnızca kullanıcı klasörleri içindeki yollar kullanılabilir.")
    return candidate


def _display(path: Path) -> str:
    try:
        relative = path.relative_to(Path.home())
    except ValueError:
        return path.name
    if not relative.parts:
        return "kullanıcı klasörü"
    folder_names = {
        "desktop": "Masaüstü", "documents": "Belgeler", "downloads": "İndirilenler",
        "pictures": "Resimler", "videos": "Videolar", "music": "Müzik",
        "masaüstü": "Masaüstü", "belgeler": "Belgeler", "indirilenler": "İndirilenler",
        "resimler": "Resimler", "videolar": "Videolar", "müzik": "Müzik",
    }
    parts = list(relative.parts)
    parts[0] = folder_names.get(parts[0].lower(), parts[0])
    if len(parts) == 1:
        return parts[0] + " klasörü"
    return parts[0] + " klasörü içindeki " + ", ".join(parts[1:])


def _list_directory(command):
    target = _resolve_safe_path(command.get("path", "documents"))
    if not target.exists() or not target.is_dir():
        return False, "Belirtilen klasör bulunamadı."
    try:
        entries = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))[:40]
        if not entries:
            return True, f"{_display(target)} klasörü boş."
        names = [f"📁 {p.name}" if p.is_dir() else f"📄 {p.name}" for p in entries]
        total = sum(1 for _ in target.iterdir())
        more = f"\n… ve {total - len(entries)} öğe daha." if total > len(entries) else ""
        heading = f"{_display(target)} klasöründeki öğeler ({total}):"
        return True, heading + "\n" + "\n".join(f"• {name}" for name in names) + more
    except PermissionError:
        return False, "Bu klasörü listelemek için izin yok."


def _find_file(command):
    root = _resolve_safe_path(command.get("path", "downloads"))
    pattern = str(command.get("pattern", "*"))[:160].strip() or "*"
    if any(part in pattern for part in ("/", "\\", "..")):
        return False, "Arama ifadesi dosya adı veya uzantısı olmalı; yol içeremez."
    if not root.exists() or not root.is_dir():
        return False, "Aranacak klasör bulunamadı."
    recursive = bool(command.get("recursive", False))
    matches = []
    visited = 0
    try:
        if recursive:
            for current, dirs, files in os.walk(root):
                dirs[:] = [d for d in dirs if d.lower() not in SKIP_DIRS and not d.startswith(".")]
                visited += 1
                if visited > 5000:
                    break
                for name in files:
                    if fnmatch.fnmatch(name.lower(), pattern.lower()):
                        matches.append(Path(current) / name)
                        if len(matches) >= 20:
                            break
                if len(matches) >= 20:
                    break
        else:
            matches = [p for p in root.iterdir() if p.is_file() and fnmatch.fnmatch(p.name.lower(), pattern.lower())][:20]
    except PermissionError:
        return False, "Arama sırasında erişim izni olmayan bir klasöre rastlandı."
    if not matches:
        return True, f"{_display(root)} içinde {pattern} eşleşmesi bulamadım."
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    lines = [f"{p.name} — {_display(p)}" for p in matches[:10]]
    suffix = "\n… Daha fazla sonuç olabilir." if len(matches) > len(lines) or visited > 5000 else ""
    heading = f"Bulduğum dosyalar ({len(lines)} gösteriliyor):"
    return True, heading + "\n" + "\n".join(f"• {line}" for line in lines) + suffix


def _file_info(command):
    target = _resolve_safe_path(command.get("path", ""))
    if not target.exists():
        return False, "Belirtilen dosya veya klasör bulunamadı."
    stat = target.stat()
    kind = "klasör" if target.is_dir() else "dosya"
    size = ""
    if target.is_file():
        size = f", boyutu {stat.st_size} bayt"
    modified = __import__("datetime").datetime.fromtimestamp(stat.st_mtime).strftime("%d.%m.%Y %H:%M")
    return True, f"{target.name} bir {kind}{size}. Son değiştirilme zamanı {modified}."


def _open_path(command):
    target = _resolve_safe_path(command.get("path", ""))
    if not target.exists():
        return False, "Açılacak dosya veya klasör bulunamadı."
    if os.name != "nt":
        return False, "Dosya ve klasör açma bu sürümde Windows üzerinde destekleniyor."
    try:
        os.startfile(str(target))  # type: ignore[attr-defined]
        return True, f"Açıldı: {_display(target)}"
    except Exception as exc:
        return False, f"Dosya veya klasör açılamadı: {type(exc).__name__}: {exc}"


def _copy_file(command):
    source = _resolve_safe_path(command.get("source", ""))
    destination = _resolve_safe_path(command.get("destination", ""))
    if not source.exists() or not source.is_file():
        return False, "Kopyalanacak kaynak dosya bulunamadı."
    if destination.exists() and destination.is_dir():
        destination = destination / source.name
    if destination.exists():
        return False, "Hedefte aynı adlı öğe zaten var; üzerine yazmadım. Önce hedefi netleştir."
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True, f"Dosya kopyalandı: {_display(destination)}"


def _move_file(command):
    if not command.get("confirmed"):
        return False, "Bu işlem kullanıcı onayı gerektiriyor."
    source = _resolve_safe_path(command.get("source", ""))
    destination = _resolve_safe_path(command.get("destination", ""))
    if not source.exists() or not source.is_file():
        return False, "Taşınacak kaynak dosya bulunamadı."
    if destination.exists() and destination.is_dir():
        destination = destination / source.name
    if destination.exists():
        return False, "Hedefte aynı adlı öğe var; üzerine yazmadım."
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))
    return True, f"Dosya taşındı: {_display(destination)}"


def _rename_file(command):
    if not command.get("confirmed"):
        return False, "Bu işlem kullanıcı onayı gerektiriyor."
    source = _resolve_safe_path(command.get("path", ""))
    new_name = str(command.get("new_name", "")).strip()
    if not source.exists() or not source.is_file():
        return False, "Yeniden adlandırılacak dosya bulunamadı."
    if not new_name or new_name in {".", ".."} or Path(new_name).name != new_name or any(ch in new_name for ch in '<>:"/\\|?*'):
        return False, "Yeni ad yalnızca geçerli bir dosya adı olabilir; yol içeremez."
    destination = source.with_name(new_name)
    if destination.exists():
        return False, "Bu isimde bir dosya zaten var; üzerine yazmadım."
    source.rename(destination)
    return True, f"Dosyanın yeni adı: {destination.name}"


def _delete_file(command):
    if not command.get("confirmed"):
        return False, "Bu işlem kullanıcı onayı gerektiriyor."
    target = _resolve_safe_path(command.get("path", ""))
    if not target.exists():
        return False, "Silinecek dosya bulunamadı."
    if not target.is_file():
        return False, "Güvenlik nedeniyle bu eylemle yalnızca dosyalar silinebilir; klasör silinmez."
    target.unlink()
    return True, f"Dosya silindi: {_display(target)}"


def validate_file_command(command):
    """Preflight a file action without changing the filesystem."""
    action = command.get("action") if isinstance(command, dict) else None
    allowed = {"list_directory", "find_file", "file_info", "open_path", "copy_file", "move_file", "rename_file", "delete_file"}
    if action not in allowed:
        return False, "Bilinmeyen dosya işlemi."
    try:
        if action in {"list_directory", "find_file"}:
            target = _resolve_safe_path(command.get("path", "documents" if action == "list_directory" else "downloads"))
            if not target.exists() or not target.is_dir():
                return False, "Aranacak veya listelenecek klasör bulunamadı."
            if action == "find_file":
                pattern = str(command.get("pattern", "*"))[:160].strip() or "*"
                if any(part in pattern for part in ("/", "\\", "..")):
                    return False, "Arama ifadesi yalnızca dosya adı veya uzantısı olabilir."
        elif action in {"file_info", "open_path", "delete_file", "rename_file"}:
            target = _resolve_safe_path(command.get("path", ""))
            if not target.exists():
                return False, "Belirtilen dosya veya klasör bulunamadı."
            if action in {"delete_file", "rename_file"} and not target.is_file():
                return False, "Bu işlem yalnızca dosyalarda destekleniyor; klasörler değiştirilmiyor."
            if action == "rename_file":
                new_name = str(command.get("new_name", "")).strip()
                if not new_name or new_name in {".", ".."} or Path(new_name).name != new_name or any(ch in new_name for ch in '<>:"/\\|?*'):
                    return False, "Yeni ad geçerli bir dosya adı olmalı ve yol içeremez."
                if target.with_name(new_name).exists():
                    return False, "Bu isimde bir dosya zaten var; üzerine yazmadım."
        elif action in {"copy_file", "move_file"}:
            source = _resolve_safe_path(command.get("source", ""))
            destination = _resolve_safe_path(command.get("destination", ""))
            if not source.exists() or not source.is_file():
                return False, "Kaynak dosya bulunamadı."
            if destination.exists() and destination.is_dir():
                destination = destination / source.name
            if destination.exists():
                return False, "Hedefte aynı adlı öğe zaten var; üzerine yazmadım."
        return True, None
    except (ValueError, OSError) as exc:
        return False, str(exc)

def execute_file_action(command):
    ok, error = validate_file_command(command)
    if not ok:
        return False, error
    action = command.get("action")
    try:
        if action == "list_directory": return _list_directory(command)
        if action == "find_file": return _find_file(command)
        if action == "file_info": return _file_info(command)
        if action == "open_path": return _open_path(command)
        if action == "copy_file": return _copy_file(command)
        if action == "move_file": return _move_file(command)
        if action == "rename_file": return _rename_file(command)
        if action == "delete_file": return _delete_file(command)
    except PermissionError:
        return False, "Dosya işlemine erişim izni verilmedi."
    except OSError as exc:
        return False, f"Dosya işlemi tamamlanamadı: {type(exc).__name__}: {exc}"
    except Exception as exc:
        return False, f"Dosya işlemi tamamlanamadı: {type(exc).__name__}: {exc}"
    return False, "Dosya işlemi tanınmadı."
