# -*- coding: utf-8 -*-
"""NOVA Voice Assistant UI v4 — liquid orb, bilingual interface; engine lifecycle preserved."""
import contextlib
import ctypes
import io
import json
import os
import subprocess
import sys
import time
import socket
import threading
import shutil
from pathlib import Path

# Register and explicitly preload the CUDA DLLs used by CTranslate2/faster-whisper.
# On Windows, adding a DLL directory alone may not load cuBLAS early enough; the
# failure can otherwise appear only on the first transcription/inference call.
_DLL_DIR_HANDLES = []
_CUDA_DLL_HANDLES = []
_CUDA_DLL_SETUP_ERRORS = []
if os.name == "nt":
    _ROOT_FOR_DLLS = Path(__file__).resolve().parent
    _SITE_FOR_DLLS = _ROOT_FOR_DLLS / "tts_env" / "Lib" / "site-packages"
    _DLL_DIR_CANDIDATES = [
        _SITE_FOR_DLLS / "torch" / "lib",
        _SITE_FOR_DLLS / "nvidia" / "cuda_runtime" / "bin",
        _SITE_FOR_DLLS / "nvidia" / "cublas" / "bin",
        _SITE_FOR_DLLS / "nvidia" / "cudnn" / "bin",
    ]
    for _dll_dir in _DLL_DIR_CANDIDATES:
        if _dll_dir.is_dir():
            try:
                _DLL_DIR_HANDLES.append(os.add_dll_directory(str(_dll_dir)))
            except (OSError, AttributeError) as _exc:
                _CUDA_DLL_SETUP_ERRORS.append(f"DLL dizini eklenemedi ({_dll_dir}): {_exc}")

    # Preload from the known-good tts_env torch\lib location first, matching
    # the successful standalone CUDA Whisper diagnostic test.
    _CUDA_DLL_PRELOADS = [
        _SITE_FOR_DLLS / "torch" / "lib" / "cublas64_12.dll",
        _SITE_FOR_DLLS / "torch" / "lib" / "cublasLt64_12.dll",
    ]
    for _dll_path in _CUDA_DLL_PRELOADS:
        if _dll_path.is_file():
            try:
                _CUDA_DLL_HANDLES.append(ctypes.WinDLL(str(_dll_path)))
            except OSError as _exc:
                _CUDA_DLL_SETUP_ERRORS.append(f"CUDA DLL yüklenemedi ({_dll_path.name}): {_exc}")
        else:
            _CUDA_DLL_SETUP_ERRORS.append(f"CUDA DLL bulunamadı: {_dll_path}")

import requests
from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer, QPoint
from PySide6.QtGui import QPainter, QColor, QPen, QRadialGradient, QPainterPath
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QTextBrowser, QLabel, QFrame, QPushButton, QSizePolicy,
    QGridLayout, QProgressBar, QFileDialog
)

# Engine import errors are retained and shown in the UI instead of being silently swallowed.
from system_monitor import ResourceCollector

ENGINE_IMPORT_ERROR = None
try:
    from conversation_engine import (
        WhisperModel, record_until_silence, transcribe,
        parse_emoji_instruction, ask_jarvis, show_result, try_parse_json,
        WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE, JARVIS_URL
    )
except Exception as exc:
    ENGINE_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


def tcp_ready(host="127.0.0.1", port=5000, timeout=0.35):
    """Check whether a local service accepts TCP connections without generating HTTP 404 logs."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def gemma_ready(timeout=0.35):
    """Gemma is ready only when its OpenAI-compatible models endpoint returns 200."""
    try:
        return requests.get("http://127.0.0.1:8080/v1/models", timeout=timeout).status_code == 200
    except requests.RequestException:
        return False


class ResourceMonitorThread(QThread):
    """Collect telemetry off the UI thread so live metrics never block voice controls."""
    snapshot_ready = Signal(object)

    def __init__(self, parent=None, interval_seconds=2.0):
        super().__init__(parent)
        self.interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._collector = ResourceCollector()

    def stop(self):
        self._stop_event.set()

    def run(self):
        while not self._stop_event.is_set():
            try:
                self.snapshot_ready.emit(self._collector.collect())
            except Exception as exc:
                self.snapshot_ready.emit({"timestamp": "—", "errors": [f"Ölçüm hatası: {type(exc).__name__}"],
                                          "gpu": {}, "models": {}, "disks": []})
            self._stop_event.wait(self.interval_seconds)


class ServerManager:
    """Starts only missing services, using the known-good Gemma BAT, and stops UI-owned processes."""
    def __init__(self, log_callback=None):
        self.log_callback = log_callback
        self.project_dir = Path(__file__).resolve().parent
        self.llama_process = None
        self.xtts_process = None
        self.llama_owned = False
        self.xtts_owned = False
        self._log_handles = []
        self.gemma_log_path = self.project_dir / "gemma_ui.log"
        self.xtts_log_path = self.project_dir / "xtts_debug.log"

    def log(self, message):
        if self.log_callback:
            self.log_callback(message)

    def _creation_flags(self):
        return getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0

    def _open_log(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(path, "w", encoding="utf-8", buffering=1, errors="replace")
        self._log_handles.append(handle)
        return handle

    def _configured_path(self, variable, default_path):
        """Resolve an optional user path; relative overrides are relative to NOVA's folder."""
        raw = os.environ.get(variable, "").strip().strip('"')
        path = Path(raw).expanduser() if raw else Path(default_path)
        if not path.is_absolute():
            path = self.project_dir / path
        return path.resolve(strict=False)

    def _gemma_env(self):
        env = os.environ.copy()
        # The model server does not need the Brave credential; do not inherit it.
        env.pop("BRAVE_API_KEY", None)
        env["PYTHONUNBUFFERED"] = "1"
        llama_dir = self._configured_path("NOVA_LLAMA_DIR", self.project_dir / "llama-cuda")
        env["NOVA_LLAMA_DIR"] = str(llama_dir)
        # Use the project-local Python 3.11 CUDA runtime when available.
        nvidia_site = self.project_dir / "tts_env" / "Lib" / "site-packages" / "nvidia"
        extra = [llama_dir, nvidia_site / "cuda_runtime" / "bin",
                 nvidia_site / "cublas" / "bin", nvidia_site / "cudnn" / "bin"]
        env["PATH"] = os.pathsep.join([str(p) for p in extra if p.exists()] + [env.get("PATH", "")])
        return env

    def _xtts_env(self):
        env = os.environ.copy()
        # XTTS does not need the Brave credential; do not inherit it.
        env.pop("BRAVE_API_KEY", None)
        env["PYTHONUNBUFFERED"] = "1"
        venv_site = self.project_dir / "tts_env" / "Lib" / "site-packages"
        # Keep CUDA DLLs from this project environment ahead of unrelated global installs.
        old_parts = [p for p in env.get("PATH", "").split(os.pathsep) if p]
        extra = [venv_site / "torch" / "lib",
                 venv_site / "nvidia" / "cuda_runtime" / "bin",
                 venv_site / "nvidia" / "cublas" / "bin",
                 venv_site / "nvidia" / "cudnn" / "bin"]
        ffmpeg_raw = os.environ.get("NOVA_FFMPEG_BIN", "").strip().strip('"')
        if ffmpeg_raw:
            ffmpeg_bin = Path(ffmpeg_raw).expanduser()
            if not ffmpeg_bin.is_absolute():
                ffmpeg_bin = self.project_dir / ffmpeg_bin
            if ffmpeg_bin.is_dir():
                extra.append(ffmpeg_bin)
        else:
            local_ffmpeg = self.project_dir / "ffmpeg" / "bin"
            if (local_ffmpeg / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")).is_file():
                extra.append(local_ffmpeg)
            else:
                discovered = shutil.which("ffmpeg")
                if discovered:
                    extra.append(Path(discovered).resolve().parent)
        env["PATH"] = os.pathsep.join([str(p) for p in extra if p.exists()] + old_parts)
        return env

    def _llama_server_process_exists(self):
        """Avoid duplicate llama-server.exe processes while a model is loading."""
        if os.name != "nt":
            return False
        try:
            result = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq llama-server.exe", "/FO", "CSV", "/NH"],
                capture_output=True, text=True, timeout=2,
                creationflags=self._creation_flags()
            )
            output = (result.stdout or "").lower()
            return "llama-server.exe" in output
        except Exception as exc:
            self.log(f"[WARN] llama-server süreç kontrolü yapılamadı: {exc}")
            # Conservative: if process inspection fails, don't risk another GPU server.
            return True

    def start_servers(self):
        self.project_dir = Path(__file__).resolve().parent
        if not self.project_dir.is_dir():
            self.log(f"[CRITICAL] NOVA proje klasörü bulunamadı: {self.project_dir}")
            return

        # Reuse a healthy server instead of spawning another copy on every UI launch.
        if gemma_ready():
            self.log("[SERVER] Gemma zaten 8080 portunda çalışıyor; yeni llama-server açılmadı.")
        elif self._llama_server_process_exists():
            # A previous launch may still be loading the model or may be stuck.
            # Do not start another copy: that was causing multiple GPU/RAM-heavy processes.
            self.log("[WAIT] llama-server.exe zaten çalışıyor ancak API henüz hazır değil. Yeni kopya açılmayacak; gemma_ui.log kontrol ediliyor.")
        else:
            # Use the bundled corrected-template launcher. The old v15 BAT forces
            # `--chat-template gemma`, overriding Gemma 4's embedded template.
            gemma_bat = self.project_dir / "Start-Model-Server.bat"
            if not gemma_bat.is_file():
                self.log(f"[CRITICAL] Düzeltilmiş Gemma başlatıcısı bulunamadı: {gemma_bat}")
                self.log("[INFO] ZIP içindeki Start-Model-Server.bat dosyasını NOVA proje klasörüne çıkart.")
            else:
                try:
                    handle = self._open_log(self.gemma_log_path)
                    # cmd.exe /c requires the special double-quote form for a quoted
                    # batch-file path. Passing `call "path"` as a list argument can
                    # make cmd treat the quotes as part of the command name.
                    command_line = f'cmd.exe /d /s /c ""{gemma_bat}""'
                    self.llama_process = subprocess.Popen(
                        command_line,
                        cwd=str(gemma_bat.parent), env=self._gemma_env(),
                        stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                        creationflags=self._creation_flags()
                    )
                    self.llama_owned = True
                    self.log(f"[START] Kullanıcının Gemma BAT dosyası çalıştırıldı (PID {self.llama_process.pid}).")
                    self.log(f"[LOG] Gemma çıktısı: {self.gemma_log_path.name}")
                except Exception as exc:
                    self.log(f"[HATA] Gemma BAT başlatılamadı: {type(exc).__name__}: {exc}")

        if tcp_ready("127.0.0.1", 5000):
            self.log("[SERVER] XTTS zaten 5000 portunda çalışıyor; yeni Python sunucusu açılmadı.")
        else:
            py = self.project_dir / "tts_env" / "Scripts" / "python.exe"
            script = self.project_dir / "speech_service.py"
            if not py.is_file():
                self.log(f"[CRITICAL] XTTS Python bulunamadı: {py}")
            elif not script.is_file():
                self.log(f"[CRITICAL] Bilinen çalışan XTTS V3 dosyası bulunamadı: {script}")
            else:
                try:
                    handle = self._open_log(self.xtts_log_path)
                    self.xtts_process = subprocess.Popen(
                        [str(py), "-u", "-X", "faulthandler", str(script)],
                        cwd=str(self.project_dir), env=self._xtts_env(),
                        stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                        creationflags=self._creation_flags()
                    )
                    self.xtts_owned = True
                    self.log(f"[START] XTTS V3 başlatıldı (PID {self.xtts_process.pid}). Log: {self.xtts_log_path.name}")
                except Exception as exc:
                    self.log(f"[HATA] XTTS başlatılamadı: {exc}")

    def _stop_owned_process(self, process, label):
        if process is None or process.poll() is not None:
            return
        try:
            if os.name == "nt":
                # Kill only this process tree. Never use taskkill /IM for normal UI shutdown.
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               timeout=5, check=False, creationflags=self._creation_flags())
            else:
                process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
            self.log(f"[STOP] UI tarafından başlatılan {label} kapatıldı.")
        except Exception as exc:
            self.log(f"[WARN] {label} kapatılırken sorun: {exc}")

    def stop_servers(self):
        if self.llama_owned:
            self._stop_owned_process(self.llama_process, "llama-server")
        if self.xtts_owned:
            self._stop_owned_process(self.xtts_process, "XTTS")
        for handle in self._log_handles:
            try:
                handle.close()
            except Exception:
                pass
        self._log_handles.clear()


class JarvisWorker(QThread):
    log_signal = Signal(str)
    conversation_signal = Signal(str, str)
    status_signal = Signal(str, str)

    def __init__(self):
        super().__init__()
        self.running = True
        self.whisper_model = None

    def run(self):
        if ENGINE_IMPORT_ERROR:
            self.log_signal.emit(f"[KRİTİK] NOVA motoru yüklenemedi: {ENGINE_IMPORT_ERROR}")
            self.status_signal.emit("MOTOR HATASI", "idle")
            return
        engine_path = getattr(sys.modules.get("conversation_engine"), "__file__", "bilinmiyor")
        self.log_signal.emit(f"[MOTOR] Kullanılan NOVA motoru: {engine_path}")
        for _dll_error in _CUDA_DLL_SETUP_ERRORS:
            self.log_signal.emit(f"[CUDA DLL] {_dll_error}")
        if _CUDA_DLL_HANDLES:
            self.log_signal.emit(f"[CUDA DLL] Ön yükleme tamamlandı: {len(_CUDA_DLL_HANDLES)} DLL")
        self.log_signal.emit("[SİSTEM] Whisper modeli CUDA üzerinde yükleniyor...")
        try:
            self.whisper_model = WhisperModel(
                WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE
            )
            self.log_signal.emit("[HAZIR] Whisper modeli aktif.")
            self.log_signal.emit(f"[BAĞLANTI] NOVA sunucusu: {JARVIS_URL}")
            try:
                model_response = requests.get("http://127.0.0.1:8080/v1/models", timeout=2)
                if model_response.ok:
                    model_items = model_response.json().get("data", [])
                    model_ids = [str(item.get("id", "")) for item in model_items if item.get("id")]
                    self.log_signal.emit("[MODEL] Sunucudaki model: " + (", ".join(model_ids) if model_ids else "ID dönmedi"))
                else:
                    self.log_signal.emit(f"[MODEL] Model kimliği alınamadı (HTTP {model_response.status_code}).")
            except Exception as model_exc:
                self.log_signal.emit(f"[MODEL] Model kimliği kontrol edilemedi: {model_exc}")
        except Exception as exc:
            self.log_signal.emit(f"[KRİTİK] Whisper yükleme hatası: {exc}")
            self.status_signal.emit("WHISPER HATASI", "idle")
            return

        self.status_signal.emit("SİSTEM HAZIR", "idle")
        while self.running:
            try:
                self.status_signal.emit("DİNLENİYOR...", "listening")
                audio = record_until_silence()
                if not self.running:
                    break
                if audio is None:
                    continue
                self.status_signal.emit("ANALİZ EDİLİYOR...", "thinking")
                self.log_signal.emit("[ANALİZ] Ses girdisi çözümleniyor...")
                t0 = time.perf_counter()
                text = transcribe(self.whisper_model, audio)
                if not text:
                    continue
                # User-facing conversation stays separate from timing/diagnostic logs.
                self.conversation_signal.emit("user", text.strip())
                self.log_signal.emit(f"[SEN] {text} ({time.perf_counter() - t0:.2f}s)")
                # pythonw has no visible stdout. Capture engine routing/action traces and
                # forward them to the Diagnostics panel so live routing can be verified.
                engine_output = io.StringIO()
                try:
                    with contextlib.redirect_stdout(engine_output):
                        answer = ask_jarvis(text, force_emoji_this_turn=parse_emoji_instruction(text))
                finally:
                    for line in engine_output.getvalue().splitlines():
                        if line.strip():
                            self.log_signal.emit(line.strip())
                if answer:
                    self.status_signal.emit("YANIT VERİLİYOR...", "speaking")
                    visible_answer = self._visible_answer(answer)
                    if visible_answer:
                        self.conversation_signal.emit("jarvis", visible_answer)
                    self._log_answer(answer)
                    # Capture engine diagnostics without changing the working TTS call.
                    output = io.StringIO()
                    try:
                        with contextlib.redirect_stdout(output):
                            show_result(answer, text)
                    finally:
                        captured = output.getvalue()
                        for line in captured.splitlines():
                            if line.strip():
                                self.log_signal.emit(line.strip())
                        # For computer actions, display the human-facing result emitted by
                        # show_result, not the internal action name or playback diagnostics.
                        action_message = self._action_result_for_chat(answer, captured)
                        if action_message:
                            self.conversation_signal.emit("jarvis", action_message)
                self.status_signal.emit("DİNLENİYOR...", "listening")
            except Exception as exc:
                self.log_signal.emit(f"[HATA] {type(exc).__name__}: {exc}")
                self.status_signal.emit("HATA", "idle")
                time.sleep(0.25)

    def _visible_answer(self, answer):
        # Parse the same response shape as the working engine; never show raw JSON.
        try:
            payload = try_parse_json(answer)
            if isinstance(payload, dict):
                action = payload.get("action")
                if action == "chat":
                    return str(payload.get("text", "")).strip()
                if action:
                    # Action results are added after show_result returns, using its user-facing output.
                    return ""
            if isinstance(payload, list) and payload and all(
                isinstance(item, dict) and item.get("action") not in (None, "chat") for item in payload
            ):
                # Never leak raw action-plan JSON into the conversation panel.
                return ""
            return str(answer).strip()
        except Exception:
            return str(answer).strip()

    def _action_result_for_chat(self, answer, captured):
        try:
            payload = try_parse_json(answer)
        except Exception:
            payload = None
        # Ordinary chat and Brave research return action=chat; never replace their
        # substantive response with a canned computer-action acknowledgement.
        if isinstance(payload, dict):
            if not payload.get("action") or payload.get("action") == "chat":
                return ""
        elif isinstance(payload, list):
            if not payload or any(not isinstance(item, dict) or item.get("action") in (None, "chat") for item in payload):
                return ""
        else:
            return ""

        technical_starts = (
            "🔊", "▶️", "⏱", "playback", "xtts", "istemci ilk chunk", "ilk chunk",
            "sunucu toplam", "toplam süre", "toplam sure", "chunk ", "[ACTION DETAIL]",
            "❌ XTTS", "⚠️ XTTS", "Traceback", "File ", "=" * 8
        )
        lines = captured.splitlines()
        for index, raw_line in enumerate(lines):
            line = raw_line.strip()
            if line.startswith("🤖 NOVA:"):
                collected = [line.split(":", 1)[1].strip()]
                for continuation in lines[index + 1:]:
                    candidate = continuation.strip()
                    if not candidate:
                        continue
                    if candidate.startswith(technical_starts):
                        break
                    # A new log/diagnostic entry should never become part of the chat bubble.
                    if candidate.startswith(("[LIVE TRACE]", "[ACTION ROUTE]", "[NOVA]", "❌ Action hatası:")):
                        break
                    collected.append(candidate)
                return "\n".join(part for part in collected if part)
            if line.startswith("❌ Action hatası:"):
                return "Komutu çalıştırırken bir hata oluştu."
        return ""

    def _log_answer(self, answer):
        visible = self._visible_answer(answer)
        if visible:
            self.log_signal.emit(f"[NOVA] {visible}")

    def stop(self):
        # Cooperative stop; the recorder has a 5-second no-speech timeout.
        self.running = False


class OrbWidget(QWidget):
    """Minimal liquid-light orb. State changes its organic contour, not decorative orbitals."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(280, 280)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.state = "idle"
        self.phase = 0.0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(33)

    def set_state(self, state):
        self.state = state if state in {"idle", "listening", "thinking", "speaking"} else "idle"
        self.update()

    def _tick(self):
        speed = {"idle": 0.018, "listening": 0.042, "thinking": 0.068, "speaking": 0.052}.get(self.state, 0.025)
        self.phase = (self.phase + speed) % (2 * 3.14159265)
        self.update()

    def paintEvent(self, event):
        import math
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2
        r = min(w, h) * 0.285
        if self.state == "listening":
            color = QColor(98, 218, 204)
            strength, speed = 0.050, 1.45
        elif self.state == "thinking":
            color = QColor(145, 139, 255)
            strength, speed = 0.105, 2.25
        elif self.state == "speaking":
            color = QColor(105, 181, 255)
            strength, speed = 0.080, 1.85
        else:
            color = QColor(105, 164, 220)
            strength, speed = 0.025, 0.75

        pulse = 1.0 + strength * math.sin(self.phase * speed)
        # One quiet bloom behind the object; no rings, orbiting dots, or wireframe.
        glow = QRadialGradient(cx, cy, r * 2.0)
        glow.setColorAt(0.0, QColor(color.red(), color.green(), color.blue(), 42))
        glow.setColorAt(0.55, QColor(color.red(), color.green(), color.blue(), 14))
        glow.setColorAt(1.0, QColor(5, 10, 20, 0))
        p.setPen(Qt.NoPen)
        p.setBrush(glow)
        p.drawEllipse(cx-r*2, cy-r*2, r*4, r*4)

        # Build a gently shifting, soft-edged liquid silhouette.
        points = []
        count = 96
        for i in range(count):
            a = (2 * math.pi * i) / count
            wave_a = math.sin(3*a + self.phase * speed) * 0.035
            wave_b = math.sin(2*a - self.phase * speed * 0.63 + 1.2) * 0.025
            wave_c = math.cos(5*a + self.phase * 0.55) * (0.018 if self.state != "thinking" else 0.045)
            radius = r * pulse * (1.0 + wave_a + wave_b + wave_c)
            points.append((cx + math.cos(a)*radius, cy + math.sin(a)*radius))
        path = QPainterPath()
        first_mid = ((points[-1][0] + points[0][0])/2, (points[-1][1] + points[0][1])/2)
        path.moveTo(*first_mid)
        for i in range(count):
            current = points[i]
            nxt = points[(i+1) % count]
            mid = ((current[0]+nxt[0])/2, (current[1]+nxt[1])/2)
            path.quadTo(current[0], current[1], mid[0], mid[1])
        path.closeSubpath()

        # Soft lower shadow adds depth without making it look like a literal glass ball.
        shadow = QRadialGradient(cx, cy + r*0.58, r*1.05)
        shadow.setColorAt(0.0, QColor(1, 4, 12, 100))
        shadow.setColorAt(1.0, QColor(1, 4, 12, 0))
        p.setBrush(shadow)
        p.drawEllipse(cx-r*1.05, cy-r*0.65, r*2.1, r*1.65)

        liquid = QRadialGradient(cx-r*0.35, cy-r*0.48, r*1.8)
        liquid.setColorAt(0.0, QColor(246, 255, 255, 250))
        liquid.setColorAt(0.13, QColor(min(255, color.red()+65), min(255, color.green()+60), min(255, color.blue()+50), 245))
        liquid.setColorAt(0.40, QColor(color.red(), color.green(), color.blue(), 235))
        liquid.setColorAt(0.72, QColor(max(0, color.red()-35), max(0, color.green()-43), max(0, color.blue()-25), 225))
        liquid.setColorAt(1.0, QColor(13, 22, 42, 220))
        p.setPen(QPen(QColor(color.red(), color.green(), color.blue(), 115), 0.8))
        p.setBrush(liquid)
        p.drawPath(path)

        # A broad, translucent highlight makes the surface feel luminous and alive.
        highlight = QRadialGradient(cx-r*0.48, cy-r*0.55, r*0.9)
        highlight.setColorAt(0.0, QColor(255, 255, 255, 55 if self.state == "idle" else 78))
        highlight.setColorAt(0.55, QColor(238, 252, 255, 15))
        highlight.setColorAt(1.0, QColor(238, 252, 255, 0))
        p.setPen(Qt.NoPen)
        p.setBrush(highlight)
        p.drawEllipse(cx-r*0.83, cy-r*0.82, r*1.35, r*1.25)
        p.end()


class NovaMainWindow(QMainWindow):
    def __init__(self, server_manager):
        super().__init__()
        self.server_manager = server_manager
        self.server_manager.log_callback = self.append_log
        self.worker = None
        self.worker_started = False
        self.drag_position = QPoint()
        self._close_pending = False
        self._shutdown_complete = False
        self._log_offsets = {}
        self._has_chat_message = False
        self._first_listening_seen = False
        self._menu_open = False
        self._audio_panel_open = False
        self.language = "tr"

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.resize(1360, 850)
        self.setMinimumSize(1060, 700)
        self.init_ui()
        self.apply_language()
        self.apply_stylesheet()

        self.health_timer = QTimer(self)
        self.health_timer.timeout.connect(self.check_servers_health)
        self.health_timer.start(2000)
        self.log_timer = QTimer(self)
        self.log_timer.timeout.connect(self.poll_server_logs)
        self.log_timer.start(350)
        self.resource_thread = ResourceMonitorThread(self, interval_seconds=2.0)
        self.resource_thread.snapshot_ready.connect(self.update_resource_view)
        self.resource_thread.start()
        self.startup_timer = QTimer(self)
        self.startup_timer.timeout.connect(self.try_start_worker)
        self.startup_timer.start(1000)
        QTimer.singleShot(100, self.auto_start_system)

    def init_ui(self):
        root = QWidget(); root.setObjectName("appRoot")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root); outer.setContentsMargins(12, 12, 12, 12); outer.setSpacing(0)
        shell = QFrame(); shell.setObjectName("appShell")
        outer.addWidget(shell)
        shell_row = QHBoxLayout(shell); shell_row.setContentsMargins(0,0,0,0); shell_row.setSpacing(0)

        # Collapsible navigation drawer. Every item has a real UI action.
        self.nav_drawer = QFrame(); self.nav_drawer.setObjectName("navDrawer")
        self.nav_drawer.setFixedWidth(190); self.nav_drawer.hide()
        nav = QVBoxLayout(self.nav_drawer); nav.setContentsMargins(14,20,14,18); nav.setSpacing(9)
        nav_brand = QLabel("NOVA"); nav_brand.setObjectName("navBrand")
        nav_caption = QLabel("PERSONAL AI · LOCAL"); nav_caption.setObjectName("navCaption")
        nav.addWidget(nav_brand); nav.addWidget(nav_caption)
        rule = QFrame(); rule.setObjectName("rule"); rule.setFixedHeight(1); nav.addWidget(rule)
        self.btn_core = QPushButton("◉   Core"); self.btn_core.setObjectName("navButton")
        self.btn_chat = QPushButton("▤   Conversation"); self.btn_chat.setObjectName("navButton")
        self.btn_audio = QPushButton("⌁   Audio status"); self.btn_audio.setObjectName("navButton")
        self.btn_menu_diag = QPushButton("⌘   Diagnostics"); self.btn_menu_diag.setObjectName("navButton")
        self.btn_resources = QPushButton("▥   System resources"); self.btn_resources.setObjectName("navButton")
        for b in (self.btn_core, self.btn_chat, self.btn_audio, self.btn_menu_diag, self.btn_resources): nav.addWidget(b)
        nav.addStretch(1)
        nav_note = QLabel("ON-DEVICE PROCESSING\nPrivate by design"); nav_note.setObjectName("navNote")
        nav.addWidget(nav_note)
        shell_row.addWidget(self.nav_drawer)

        main = QWidget(); main.setObjectName("mainArea")
        shell_row.addWidget(main, 1)
        main_layout = QVBoxLayout(main); main_layout.setContentsMargins(24,18,24,18); main_layout.setSpacing(10)

        # Compact title bar remains; the oversized intro strip folds away once listening begins.
        titlebar = QWidget(); titlebar.setObjectName("titleBar")
        title_row = QHBoxLayout(titlebar); title_row.setContentsMargins(0,0,0,0); title_row.setSpacing(10)
        self.btn_menu = QPushButton("☰"); self.btn_menu.setObjectName("menuButton")
        self.btn_menu.setToolTip("Menüyü aç / kapat")
        title_row.addWidget(self.btn_menu)
        title_stack = QVBoxLayout(); title_stack.setSpacing(0)
        self.brand_label = QLabel("NOVA"); self.brand_label.setObjectName("brandLabel")
        self.brand_subtitle = QLabel("VOICE-FIRST PERSONAL INTELLIGENCE"); self.brand_subtitle.setObjectName("brandSubtitle")
        title_stack.addWidget(self.brand_label); title_stack.addWidget(self.brand_subtitle)
        title_row.addLayout(title_stack)
        title_row.addStretch(1)
        self.header_private = QLabel("LOCAL  /  PRIVATE SESSION"); self.header_private.setObjectName("privateTag")
        title_row.addWidget(self.header_private)
        self.btn_language = QPushButton("EN"); self.btn_language.setObjectName("quietButton"); self.btn_language.setFixedWidth(48)
        title_row.addWidget(self.btn_language)
        self.btn_chat_top = QPushButton("CHAT  +"); self.btn_chat_top.setObjectName("quietButton")
        self.btn_diag_top = QPushButton("DIAGNOSTICS"); self.btn_diag_top.setObjectName("quietButton")
        title_row.addWidget(self.btn_chat_top); title_row.addWidget(self.btn_diag_top)
        self.btn_minimize = QPushButton("—"); self.btn_minimize.setObjectName("windowButton"); self.btn_minimize.clicked.connect(self.showMinimized)
        self.btn_maximize = QPushButton("□"); self.btn_maximize.setObjectName("windowButton"); self.btn_maximize.clicked.connect(self.toggle_maximize)
        self.btn_exit = QPushButton("×"); self.btn_exit.setObjectName("exitButton"); self.btn_exit.clicked.connect(self.close)
        title_row.addWidget(self.btn_minimize); title_row.addWidget(self.btn_maximize); title_row.addWidget(self.btn_exit)
        main_layout.addWidget(titlebar)

        # Startup intro folds away after the engine reaches listening state.
        self.intro_strip = QFrame(); self.intro_strip.setObjectName("introStrip")
        intro = QHBoxLayout(self.intro_strip); intro.setContentsMargins(18,13,18,13); intro.setSpacing(14)
        intro_mark = QLabel("✦"); intro_mark.setObjectName("introMark")
        intro.addWidget(intro_mark)
        intro_text = QVBoxLayout(); intro_text.setSpacing(3)
        intro_title = QLabel("A quieter interface for a smarter assistant."); intro_title.setObjectName("introTitle")
        intro_desc = QLabel("Speak naturally. Your conversation stays local, and the interface stays out of your way.")
        intro_desc.setObjectName("introDesc"); intro_desc.setWordWrap(True)
        intro_text.addWidget(intro_title); intro_text.addWidget(intro_desc); intro.addLayout(intro_text,1)
        self.intro_status = QLabel("INITIALIZING"); self.intro_status.setObjectName("introStatus")
        intro.addWidget(self.intro_status)
        main_layout.addWidget(self.intro_strip)

        # Stage: orb first, no chat on screen by default. Chat is an optional side panel.
        stage = QFrame(); stage.setObjectName("stage")
        stage_row = QHBoxLayout(stage); stage_row.setContentsMargins(0,0,0,0); stage_row.setSpacing(14)
        self.core_panel = QFrame(); self.core_panel.setObjectName("corePanel")
        core = QVBoxLayout(self.core_panel); core.setContentsMargins(20,18,20,18); core.setSpacing(7)
        core_top = QHBoxLayout()
        self.core_kicker = QLabel("NOVA  /  LIVE CORE"); self.core_kicker.setObjectName("coreKicker")
        core_top.addWidget(self.core_kicker); core_top.addStretch(1)
        self.core_index = QLabel("CORE 01"); self.core_index.setObjectName("coreIndex")
        core_top.addWidget(self.core_index); core.addLayout(core_top)
        self.orb = OrbWidget(); self.orb.setMinimumSize(280,280)
        core.addWidget(self.orb,1)
        self.status_label = QLabel("CONNECTING TO LOCAL CORE"); self.status_label.setObjectName("statusLabel")
        self.status_label.setAlignment(Qt.AlignCenter); self.status_label.setWordWrap(True)
        core.addWidget(self.status_label)
        self.status_desc = QLabel("Gemma ve ses servisi kontrol ediliyor."); self.status_desc.setObjectName("statusDescription")
        self.status_desc.setAlignment(Qt.AlignCenter); self.status_desc.setWordWrap(True)
        core.addWidget(self.status_desc)
        # Minimal audio-reactive style indicator; this is status only, not a fake microphone control.
        wave_row = QHBoxLayout(); wave_row.setSpacing(5); wave_row.addStretch(1)
        self.wave_bars = []
        for i in range(17):
            bar = QFrame(); bar.setObjectName("waveBar"); bar.setFixedWidth(3); bar.setFixedHeight(5 + (i*7 % 19))
            self.wave_bars.append(bar); wave_row.addWidget(bar, alignment=Qt.AlignVCenter)
        wave_row.addStretch(1); core.addLayout(wave_row)
        service_row = QHBoxLayout(); service_row.setSpacing(8)
        self.gemma_badge = QLabel("GEMMA  ·  OFFLINE"); self.gemma_badge.setObjectName("serviceOffline"); self.gemma_badge.setAlignment(Qt.AlignCenter)
        self.xtts_badge = QLabel("XTTS  ·  OFFLINE"); self.xtts_badge.setObjectName("serviceOffline"); self.xtts_badge.setAlignment(Qt.AlignCenter)
        service_row.addWidget(self.gemma_badge); service_row.addWidget(self.xtts_badge)
        core.addLayout(service_row)
        stage_row.addWidget(self.core_panel, 1)

        self.chat_panel = QFrame(); self.chat_panel.setObjectName("chatPanel"); self.chat_panel.setFixedWidth(385); self.chat_panel.hide()
        chat_layout = QVBoxLayout(self.chat_panel); chat_layout.setContentsMargins(18,18,18,18); chat_layout.setSpacing(10)
        chat_header = QHBoxLayout()
        chat_titles = QVBoxLayout(); chat_titles.setSpacing(2)
        chat_kicker = QLabel("PRIVATE SESSION"); chat_kicker.setObjectName("chatKicker")
        chat_heading = QLabel("Conversation"); chat_heading.setObjectName("chatHeading")
        chat_titles.addWidget(chat_kicker); chat_titles.addWidget(chat_heading); chat_header.addLayout(chat_titles); chat_header.addStretch(1)
        self.btn_close_chat = QPushButton("×"); self.btn_close_chat.setObjectName("windowButton"); chat_header.addWidget(self.btn_close_chat)
        chat_layout.addLayout(chat_header)
        self.chat_view = QTextBrowser(); self.chat_view.setObjectName("chatView"); self.chat_view.setOpenExternalLinks(False); self.chat_view.setReadOnly(True); self.chat_view.setFrameShape(QFrame.NoFrame)
        self.chat_view.setHtml('<div style="color:#71849d;text-align:center;margin-top:120px;font-size:10pt;">Conversation history will appear here when opened.</div>')
        chat_layout.addWidget(self.chat_view,1)
        chat_footer = QLabel("VOICE SESSION  ·  LOCAL HISTORY IN THIS RUN"); chat_footer.setObjectName("chatFooter")
        chat_layout.addWidget(chat_footer)
        stage_row.addWidget(self.chat_panel)

        # Resource view occupies the full stage. Every metric stays visible without a scroll area.
        self.resources_panel = QFrame(); self.resources_panel.setObjectName("resourcesPanel")
        self.resources_panel.hide()
        resources_outer = QVBoxLayout(self.resources_panel); resources_outer.setContentsMargins(14,10,14,10); resources_outer.setSpacing(7)
        resources_header = QHBoxLayout()
        resources_title = QLabel("SYSTEM RESOURCES / LIVE"); resources_title.setObjectName("resourceSectionTitle")
        resources_header.addWidget(resources_title); resources_header.addStretch(1)
        self.btn_export_resources = QPushButton("EXPORT JSON"); self.btn_export_resources.setObjectName("quietButton")
        self.btn_export_resources.clicked.connect(self.export_resource_snapshot)
        resources_header.addWidget(self.btn_export_resources)
        self.resource_time = QLabel("Refresh: —"); self.resource_time.setObjectName("resourceMeta")
        resources_header.addWidget(self.resource_time); resources_outer.addLayout(resources_header)
        self.resource_widgets = {}
        metric_grid = QGridLayout(); metric_grid.setContentsMargins(0,0,0,0); metric_grid.setSpacing(8)
        for index, (key, title, subtitle) in enumerate((
            ("cpu", "CPU", "Usage · Temperature"), ("ram", "RAM", "Used / total"),
            ("gpu", "GPU", "Load · Temperature"), ("vram", "VRAM", "Used / total"))):
            card, refs = self._make_resource_card(key, title, subtitle, with_bar=True)
            metric_grid.addWidget(card, 0, index)
            self.resource_widgets[key] = refs
            metric_grid.setColumnStretch(index, 1)
        resources_outer.addLayout(metric_grid, 2)

        disk_card, disk_refs = self._make_resource_card("disks", "DISKS", "Capacity and I/O", with_bar=False)
        resources_outer.addWidget(disk_card); self.resource_widgets["disks"] = disk_refs

        model_heading = QLabel("MODEL PROCESSES"); model_heading.setObjectName("resourceSectionTitle")
        resources_outer.addWidget(model_heading)
        model_row = QHBoxLayout(); model_row.setContentsMargins(0,0,0,0); model_row.setSpacing(8)
        for key, title, subtitle in (("whisper", "Whisper / NOVA process", "Includes the NOVA UI process"),
                                     ("gemma", "Gemma / llama-server", "Separate inference process"),
                                     ("xtts", "XTTS / Python server", "Separate speech server")):
            card, refs = self._make_resource_card(key, title, subtitle, with_bar=False)
            model_row.addWidget(card, 1); self.resource_widgets[key] = refs
        resources_outer.addLayout(model_row, 2)

        speed_heading = QLabel("GEMMA INFERENCE SPEED"); speed_heading.setObjectName("resourceSectionTitle")
        resources_outer.addWidget(speed_heading)
        speed_row = QHBoxLayout(); speed_row.setContentsMargins(0,0,0,0); speed_row.setSpacing(8)
        for key, title, subtitle in (("prompt_speed", "READ / PROMPT TOK/S", "Prompt evaluation speed"),
                                     ("generation_speed", "GENERATE / TOK/S", "Output token generation speed")):
            card, refs = self._make_resource_card(key, title, subtitle, with_bar=False)
            speed_row.addWidget(card, 1); self.resource_widgets[key] = refs
        resources_outer.addLayout(speed_row, 1)

        self.resource_note = QLabel("Measured values only · Unavailable sensors are not estimated.")
        self.resource_note.setObjectName("resourceNote"); self.resource_note.setWordWrap(True); self.resource_note.setMaximumHeight(28)
        resources_outer.addWidget(self.resource_note)
        stage_row.addWidget(self.resources_panel, 1)
        main_layout.addWidget(stage,1)

        # Audio panel is an actual status view, not a dead navigation item.
        self.audio_panel = QFrame(); self.audio_panel.setObjectName("audioPanel"); self.audio_panel.hide()
        audio_layout = QHBoxLayout(self.audio_panel); audio_layout.setContentsMargins(16,12,16,12); audio_layout.setSpacing(16)
        audio_title = QLabel("AUDIO PIPELINE"); audio_title.setObjectName("audioTitle"); audio_layout.addWidget(audio_title)
        self.audio_info = QLabel(f"Input: Whisper · {WHISPER_DEVICE} · {WHISPER_COMPUTE_TYPE}     Output: XTTS streaming · localhost:5000")
        self.audio_info.setObjectName("audioInfo"); self.audio_info.setWordWrap(True); audio_layout.addWidget(self.audio_info,1)
        self.btn_close_audio = QPushButton("CLOSE"); self.btn_close_audio.setObjectName("quietButton"); audio_layout.addWidget(self.btn_close_audio)
        main_layout.addWidget(self.audio_panel)

        # Diagnostics are hidden by default and never leak into the chat.
        self.diagnostics_panel = QFrame(); self.diagnostics_panel.setObjectName("diagnosticsPanel"); self.diagnostics_panel.hide()
        diag = QVBoxLayout(self.diagnostics_panel); diag.setContentsMargins(0,8,0,0); diag.setSpacing(6)
        diag_top = QHBoxLayout(); diag_label = QLabel("DIAGNOSTICS  /  TECHNICAL OUTPUT"); diag_label.setObjectName("diagLabel")
        diag_top.addWidget(diag_label); diag_top.addStretch(1)
        self.btn_clear_logs = QPushButton("CLEAR VIEW"); self.btn_clear_logs.setObjectName("quietButton"); diag_top.addWidget(self.btn_clear_logs)
        diag.addLayout(diag_top)
        self.log_box = QTextEdit(); self.log_box.setObjectName("logView"); self.log_box.setReadOnly(True); self.log_box.setMinimumHeight(90); self.log_box.setMaximumHeight(150)
        self.btn_clear_logs.clicked.connect(self.log_box.clear); diag.addWidget(self.log_box)
        main_layout.addWidget(self.diagnostics_panel)

        footer = QHBoxLayout(); footer.setContentsMargins(2,2,2,0)
        self.footer_left = QLabel("NOVA DESKTOP  /  VOICE-FIRST"); self.footer_left.setObjectName("footerLabel")
        footer.addWidget(self.footer_left); footer.addStretch(1)
        self.footer_right = QLabel("LOCAL MODE · NO CLOUD ROUND-TRIP"); self.footer_right.setObjectName("footerMeta")
        footer.addWidget(self.footer_right); main_layout.addLayout(footer)

        # Functional navigation actions.
        self.btn_menu.clicked.connect(self.toggle_menu)
        self.btn_language.clicked.connect(self.toggle_language)
        self.btn_core.clicked.connect(self.show_core)
        self.btn_chat.clicked.connect(self.toggle_chat)
        self.btn_chat_top.clicked.connect(self.toggle_chat)
        self.btn_close_chat.clicked.connect(self.close_chat_panel)
        self.btn_audio.clicked.connect(self.toggle_audio_panel)
        self.btn_close_audio.clicked.connect(lambda: self.audio_panel.hide())
        self.btn_diag_top.clicked.connect(self.toggle_diagnostics_button)
        self.btn_menu_diag.clicked.connect(self.toggle_diagnostics_button)
        self.btn_resources.clicked.connect(self.toggle_resources_panel)

    def _translations(self):
        # English is the canonical key; only UI copy is translated, never model output.
        return {
            "PERSONAL AI · LOCAL": "KİŞİSEL YZ · YEREL",
            "◉   Core": "◉   Ana ekran", "▤   Conversation": "▤   Sohbet",
            "⌁   Audio status": "⌁   Ses durumu", "⌘   Diagnostics": "⌘   Tanılama",
            "▥   System resources": "▥   Sistem kaynakları",
            "SYSTEM RESOURCES / LIVE": "SİSTEM KAYNAKLARI / CANLI", "Refresh: —": "Ölçüm: —",
            "EXPORT JSON": "JSON DIŞA AKTAR",
            "Usage · Temperature": "Kullanım · Sıcaklık", "Used / total": "Kullanılan / toplam",
            "Load · Temperature": "Yük · Sıcaklık", "DISKS": "DİSKLER", "Capacity and I/O": "Doluluk ve G/Ç",
            "MODEL PROCESSES": "MODEL SÜREÇLERİ", "Whisper / NOVA process": "Whisper / NOVA süreci",
            "Includes the NOVA UI process": "NOVA arayüz sürecini de içerir",
            "Gemma / llama-server": "Gemma / llama-server", "Separate inference process": "Ayrı çıkarım süreci",
            "XTTS / Python server": "XTTS / Python sunucusu", "Separate speech server": "Ayrı ses sunucusu",
            "GEMMA INFERENCE SPEED": "GEMMA ÇIKARIM HIZI",
            "READ / PROMPT TOK/S": "OKUMA / PROMPT TOK/S",
            "GENERATE / TOK/S": "ÜRETİM / TOK/S",
            "Prompt evaluation speed": "Girdi değerlendirme hızı",
            "Output token generation speed": "Yanıt token üretim hızı",
            "Measured values only · Unavailable sensors are not estimated.": "Yalnızca ölçülen değerler · Okunamayan sensörler tahmin edilmez.",
            "ON-DEVICE PROCESSING\nPrivate by design": "CİHAZ ÜZERİNDE İŞLEME\nTasarım gereği gizli",
            "Toggle navigation menu": "Menüyü aç / kapat",
            "VOICE-FIRST PERSONAL INTELLIGENCE": "SES ODAKLI KİŞİSEL ASİSTAN",
            "LOCAL  /  PRIVATE SESSION": "YEREL  /  GİZLİ OTURUM",
            "CHAT  +": "SOHBET  +", "CHAT  −": "SOHBET  −",
            "DIAGNOSTICS": "TANILAMA", "DIAGNOSTICS  −": "TANILAMA  −",
            "A quieter interface for a smarter assistant.": "Daha akıllı bir asistan için sade bir arayüz.",
            "Speak naturally. Your conversation stays local, and the interface stays out of your way.": "Doğal konuş. Görüşmen cihazında kalır; arayüz önüne çıkmaz.",
            "INITIALIZING": "BAŞLATILIYOR", "NOVA  /  LIVE CORE": "NOVA  /  CANLI ÇEKİRDEK",
            "CORE 01": "ÇEKİRDEK 01", "CONNECTING TO LOCAL CORE": "YEREL ÇEKİRDEĞE BAĞLANIYOR",
            "Checking Gemma and audio services.": "Gemma ve ses servisi kontrol ediliyor.",
            "GEMMA  ·  OFFLINE": "GEMMA  ·  ÇEVRİMDIŞI", "XTTS  ·  OFFLINE": "XTTS  ·  ÇEVRİMDIŞI",
            "PRIVATE SESSION": "ÖZEL OTURUM", "Conversation": "Sohbet",
            "Conversation history will appear here when opened.": "Sohbet geçmişi açıldığında burada görünür.",
            "VOICE SESSION  ·  LOCAL HISTORY IN THIS RUN": "SES OTURUMU  ·  GEÇMİŞ BU OTURUMDA",
            "AUDIO PIPELINE": "SES AKIŞI", "CLOSE": "KAPAT", "CLEAR VIEW": "GÖRÜNÜMÜ TEMİZLE",
            "DIAGNOSTICS  /  TECHNICAL OUTPUT": "TANILAMA  /  TEKNİK ÇIKTI",
            "NOVA DESKTOP  /  VOICE-FIRST": "NOVA MASAÜSTÜ  /  SES ODAKLI",
            "LOCAL MODE · NO CLOUD ROUND-TRIP": "YEREL MOD · BULUTA GİDİŞ GELİŞ YOK",
            "Ready when you are": "Hazır olduğunda buradayım",
            "Listening will resume automatically.": "Dinleme otomatik olarak devam eder.",
            "Listening": "Dinliyorum", "I’m here. Speak naturally.": "Buradayım. Doğal konuşabilirsin.",
            "Thinking it through": "Düşünüyorum", "Your request is being processed locally.": "İsteğin cihazında işleniyor.",
            "Speaking": "Yanıtlıyorum", "Preparing and playing the response.": "Yanıt hazırlanıyor ve seslendiriliyor.",
            "Needs attention": "Kontrol gerekiyor", "Starting up": "Başlatılıyor",
            "Local voice session": "Yerel ses oturumu",
            "A local service is unavailable. Open diagnostics for details.": "Yerel bir hizmet kullanılamıyor. Ayrıntılar için tanılamayı aç.",
            "READY · LOCAL SESSION": "HAZIR · YEREL OTURUM", "CONNECTING": "BAĞLANIYOR",
        }

    def t(self, english):
        tr = self._translations()
        if self.language == "tr":
            return tr.get(english, english)
        reverse = {translated: original for original, translated in tr.items()}
        return reverse.get(english, english)

    def toggle_language(self):
        self.language = "en" if self.language == "tr" else "tr"
        self.apply_language()
        # Re-render dynamic state labels in the selected language.
        state = self.orb.state
        self.update_status("SİSTEM HAZIR" if state == "idle" else "", state)

    def apply_language(self):
        tr = self._translations()
        reverse = {translated: original for original, translated in tr.items()}
        mapping = tr if self.language == "tr" else reverse
        for widget in self.findChildren(QLabel) + self.findChildren(QPushButton):
            current = widget.text()
            if current in mapping:
                widget.setText(mapping[current])
            tooltip = widget.toolTip()
            if tooltip:
                if self.language == "tr" and tooltip == "Toggle navigation menu":
                    widget.setToolTip("Menüyü aç / kapat")
                elif self.language == "en" and tooltip == "Menüyü aç / kapat":
                    widget.setToolTip("Toggle navigation menu")
        self.btn_language.setText("EN" if self.language == "tr" else "TR")
        self.btn_chat_top.setText(self.t("CHAT  −" if self.chat_panel.isVisible() else "CHAT  +"))
        self.btn_diag_top.setText(self.t("DIAGNOSTICS  −" if self.diagnostics_panel.isVisible() else "DIAGNOSTICS"))
        self.update_resource_view(getattr(self, "_last_resource_snapshot", {"timestamp": "—", "gpu": {}, "models": {}, "disks": [], "inference_speed": {}}))
        if not self._has_chat_message:
            placeholder = self.t("Conversation history will appear here when opened.")
            self.chat_view.setHtml(f'<div style="color:#71849d;text-align:center;margin-top:120px;font-size:10pt;">{placeholder}</div>')
        self.audio_info.setText(self.t("Input: Whisper · DEVICE · COMPUTE     Output: XTTS streaming · localhost:5000").replace("DEVICE", WHISPER_DEVICE).replace("COMPUTE", WHISPER_COMPUTE_TYPE) if self.language == "en" else f"Giriş: Whisper · {WHISPER_DEVICE} · {WHISPER_COMPUTE_TYPE}     Çıkış: XTTS akış · localhost:5000")
        self.check_servers_health()

    def apply_stylesheet(self):
        self.setStyleSheet(r"""
            QWidget#appRoot { background:#050913; color:#e8f2ff; }
            QFrame#appShell { background:#080e19; border:1px solid #1a2a40; border-radius:24px; }
            QWidget#mainArea { background:transparent; }
            QFrame#navDrawer { background:#080d17; border-right:1px solid #203047; border-top-left-radius:24px; border-bottom-left-radius:24px; }
            QLabel#navBrand { color:#e8f7ff; font-size:22px; font-weight:900; letter-spacing:5px; }
            QLabel#navCaption { color:#607991; font-size:8px; font-weight:800; letter-spacing:1.5px; padding-bottom:10px; }
            QLabel#navNote { color:#536b83; font-size:9px; line-height:1.4; padding-top:14px; }
            QFrame#rule { background:#213149; border:none; }
            QPushButton#navButton { text-align:left; padding:12px 13px; background:transparent; border:1px solid transparent; border-radius:10px; color:#8da5bd; font-size:11px; font-weight:700; }
            QPushButton#navButton:hover { color:#dffaff; background:#102236; border-color:#23455e; }
            QWidget#titleBar { background:transparent; }
            QLabel#brandLabel { color:#e9f8ff; font-size:17px; font-weight:900; letter-spacing:4px; }
            QLabel#brandSubtitle { color:#607991; font-size:8px; font-weight:800; letter-spacing:1.6px; }
            QLabel#privateTag { color:#71d8bd; background:#0c211f; border:1px solid #1d4942; border-radius:9px; padding:8px 10px; font-size:8px; font-weight:900; letter-spacing:1px; }
            QPushButton#menuButton { min-width:38px; min-height:38px; background:#101b2b; border:1px solid #263950; border-radius:11px; color:#a7d9ea; font-size:18px; }
            QPushButton#menuButton:hover { background:#183047; color:#ffffff; }
            QPushButton#quietButton { background:#101b2b; color:#a6bfd5; border:1px solid #253950; border-radius:9px; padding:9px 12px; font-size:8px; font-weight:900; letter-spacing:1px; }
            QPushButton#quietButton:hover { background:#183149; color:#f1fbff; border-color:#37617c; }
            QPushButton#windowButton { min-width:31px; min-height:30px; background:#0d1725; color:#a8bfd3; border:1px solid #22334a; border-radius:8px; font-size:15px; }
            QPushButton#windowButton:hover { background:#20344a; color:white; }
            QPushButton#exitButton { min-width:32px; min-height:30px; background:#25151e; color:#f2a9bc; border:1px solid #583047; border-radius:8px; font-size:19px; }
            QPushButton#exitButton:hover { background:#76263e; color:white; }
            QFrame#introStrip { background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #0c1928,stop:1 #0b1422); border:1px solid #1d3147; border-radius:14px; }
            QLabel#introMark { color:#7de9f4; font-size:25px; }
            QLabel#introTitle { color:#dceeff; font-size:13px; font-weight:700; }
            QLabel#introDesc { color:#6f879f; font-size:10px; }
            QLabel#introStatus { color:#8adccf; background:#0e2928; border:1px solid #24524c; border-radius:8px; padding:8px 10px; font-size:8px; font-weight:900; letter-spacing:1px; }
            QFrame#stage { background:transparent; }
            QFrame#corePanel, QFrame#chatPanel, QFrame#resourcesPanel { background:qlineargradient(x1:0,y1:0,x2:0.8,y2:1,stop:0 #0b1523,stop:1 #080f1b); border:1px solid #1c2e44; border-radius:20px; }
            QFrame#resourceCard { background:#0a1422; border:1px solid #1c3046; border-radius:12px; }
            QLabel#resourceSectionTitle { color:#80dbe9; font-size:9px; font-weight:900; letter-spacing:1.2px; }
            QLabel#resourceCardTitle { color:#d9e9f7; font-size:10px; font-weight:800; }
            QLabel#resourceValue { color:#8fe3ed; font-size:19px; font-weight:900; }
            QLabel#resourceDetail { color:#8ba2b8; font-size:9px; }
            QLabel#resourceMeta, QLabel#resourceNote { color:#668199; font-size:8px; }
            QProgressBar#resourceBar { background:#101d2d; border:1px solid #20344b; border-radius:4px; height:7px; text-align:center; }
            QProgressBar#resourceBar::chunk { background:#4cb9c7; border-radius:3px; }
            QLabel#coreKicker, QLabel#chatKicker { color:#5f829f; font-size:8px; font-weight:900; letter-spacing:2px; }
            QLabel#coreIndex { color:#7db8ce; background:#101e2d; border:1px solid #263b51; border-radius:7px; padding:6px 8px; font-size:8px; font-weight:900; }
            QLabel#statusLabel { color:#a7e9f3; font-size:12px; font-weight:900; letter-spacing:2px; padding-top:4px; }
            QLabel#statusDescription { color:#6e879f; font-size:10px; padding-bottom:4px; }
            QFrame#waveBar { background:#3d819a; border:none; border-radius:2px; }
            QLabel#serviceOffline, QLabel#serviceOnline { border-radius:8px; padding:9px 10px; font-size:8px; font-weight:900; letter-spacing:1px; }
            QLabel#serviceOffline { background:#211721; border:1px solid #4b2d42; color:#e7a1b3; }
            QLabel#serviceOnline { background:#0d2624; border:1px solid #23534b; color:#7fe1c5; }
            QLabel#chatHeading { color:#e7f4ff; font-size:19px; font-weight:700; }
            QTextBrowser#chatView { background:transparent; color:#dfeaf6; border:none; font-size:10pt; padding:4px; }
            QLabel#chatFooter { color:#526c84; font-size:8px; font-weight:800; letter-spacing:1px; padding-top:8px; }
            QFrame#audioPanel { background:#0b1725; border:1px solid #20354c; border-radius:12px; }
            QLabel#audioTitle { color:#8fe3ed; font-size:9px; font-weight:900; letter-spacing:1.5px; }
            QLabel#audioInfo { color:#8ea6bc; font-size:10px; }
            QFrame#diagnosticsPanel { background:transparent; border:none; }
            QLabel#diagLabel { color:#6488a3; font-size:8px; font-weight:900; letter-spacing:1.5px; }
            QTextEdit#logView { background:#050a12; color:#8fc9d1; font-family:'Consolas'; font-size:9px; border:1px solid #1b2e43; border-radius:10px; padding:8px; }
            QLabel#footerLabel { color:#607b94; font-size:8px; font-weight:900; letter-spacing:1.4px; }
            QLabel#footerMeta { color:#3f586f; font-size:8px; font-weight:800; letter-spacing:1px; }
        """)

    def toggle_menu(self):
        self._menu_open = not self._menu_open
        self.nav_drawer.setVisible(self._menu_open)
        self.btn_menu.setText("×" if self._menu_open else "☰")

    def show_core(self):
        self.core_panel.show(); self.chat_panel.hide(); self.audio_panel.hide(); self.diagnostics_panel.hide(); self.resources_panel.hide()
        self.btn_diag_top.setText("DIAGNOSTICS")
        self._audio_panel_open = False
        self.nav_drawer.hide(); self._menu_open = False; self.btn_menu.setText("☰")

    def close_chat_panel(self):
        self.chat_panel.hide()
        self.btn_chat_top.setText("CHAT  +")

    def toggle_chat(self):
        visible = not self.chat_panel.isVisible()
        self.core_panel.show()
        self.chat_panel.setVisible(visible)
        if visible:
            self.audio_panel.hide(); self._audio_panel_open = False
            self.resources_panel.hide(); self.diagnostics_panel.hide()
            self.chat_view.verticalScrollBar().setValue(self.chat_view.verticalScrollBar().maximum())
        self.btn_chat_top.setText(self.t("CHAT  −" if visible else "CHAT  +"))
        self.nav_drawer.hide(); self._menu_open = False; self.btn_menu.setText("☰")

    def toggle_audio_panel(self):
        self._audio_panel_open = not self.audio_panel.isVisible()
        self.core_panel.show()
        self.audio_panel.setVisible(self._audio_panel_open)
        if self._audio_panel_open:
            self.chat_panel.hide(); self.resources_panel.hide(); self.diagnostics_panel.hide()
        self.nav_drawer.hide(); self._menu_open = False; self.btn_menu.setText("☰")

    def toggle_diagnostics_button(self):
        visible = not self.diagnostics_panel.isVisible()
        self.core_panel.show()
        self.diagnostics_panel.setVisible(visible)
        if visible:
            self.chat_panel.hide(); self.audio_panel.hide(); self.resources_panel.hide(); self._audio_panel_open = False
        self.btn_diag_top.setText(self.t("DIAGNOSTICS  −" if visible else "DIAGNOSTICS"))
        self.nav_drawer.hide(); self._menu_open = False; self.btn_menu.setText("☰")

    def _make_resource_card(self, key, title, subtitle, with_bar=False):
        card = QFrame(); card.setObjectName("resourceCard")
        layout = QVBoxLayout(card); layout.setContentsMargins(10,7,10,7); layout.setSpacing(3)
        title_label = QLabel(self.t(title)); title_label.setObjectName("resourceCardTitle")
        layout.addWidget(title_label)
        value = QLabel("—"); value.setObjectName("resourceValue")
        detail = QLabel(self.t(subtitle)); detail.setObjectName("resourceDetail"); detail.setWordWrap(True)
        layout.addWidget(value); layout.addWidget(detail)
        bar = None
        if with_bar:
            bar = QProgressBar(); bar.setObjectName("resourceBar"); bar.setRange(0,100); bar.setValue(0)
            bar.setTextVisible(False); bar.setFixedHeight(8); layout.addWidget(bar)
        return card, {"value": value, "detail": detail, "bar": bar}

    def toggle_resources_panel(self):
        visible = not self.resources_panel.isVisible()
        self.resources_panel.setVisible(visible)
        if visible:
            self.core_panel.hide(); self.chat_panel.hide(); self.audio_panel.hide(); self.diagnostics_panel.hide()
            self._audio_panel_open = False
        else:
            self.core_panel.show()
        self.nav_drawer.hide(); self._menu_open = False; self.btn_menu.setText("☰")

    def export_resource_snapshot(self):
        snapshot = getattr(self, "_last_resource_snapshot", None)
        if not snapshot:
            self.resource_note.setText("Henüz ölçüm alınmadı." if self.language == "tr" else "No measurement available yet.")
            return
        stamp = time.strftime("%Y%m%d_%H%M%S")
        default_name = f"nova_resource_snapshot_{stamp}.json"
        path, _ = QFileDialog.getSaveFileName(self, "NOVA kaynak ölçümünü dışa aktar", default_name, "JSON files (*.json)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(snapshot, handle, ensure_ascii=False, indent=2)
            self.resource_note.setText(("Ölçüm JSON olarak kaydedildi: " if self.language == "tr" else "Snapshot saved as JSON: ") + path)
        except Exception as exc:
            self.resource_note.setText(("Dışa aktarma başarısız: " if self.language == "tr" else "Export failed: ") + str(exc))

    @Slot(object)
    def update_resource_view(self, snapshot):
        """Render measured resource telemetry; never synthesize missing sensor values."""
        self._last_resource_snapshot = snapshot
        self.resource_time.setText(("Ölçüm: " if self.language == "tr" else "Updated: ") + str(snapshot.get("timestamp", "—")))
        widgets = self.resource_widgets
        def fmt_gb(value):
            return "—" if value is None else f"{value / 1024:.1f} GB"
        def set_bar(key, value):
            bar = widgets[key].get("bar")
            if bar is not None:
                bar.setValue(max(0, min(100, int(value or 0))))
                bar.setVisible(value is not None)
        cpu = snapshot.get("cpu_percent")
        cpu_temp = snapshot.get("cpu_temp_c")
        widgets["cpu"]["value"].setText("—" if cpu is None else f"{cpu:.0f}%")
        widgets["cpu"]["detail"].setText((f"CPU sıcaklığı: {cpu_temp:.0f} °C" if self.language == "tr" else f"CPU temperature: {cpu_temp:.0f} °C") if cpu_temp is not None else ("CPU sıcaklığı: sensör verisi yok" if self.language == "tr" else "CPU temperature: sensor unavailable"))
        set_bar("cpu", cpu)
        ram = snapshot.get("ram_percent")
        ru, rt = snapshot.get("ram_used_mb"), snapshot.get("ram_total_mb")
        widgets["ram"]["value"].setText("—" if ram is None else f"{ram:.0f}%")
        widgets["ram"]["detail"].setText(f"{fmt_gb(ru)} / {fmt_gb(rt)}" if ru is not None and rt is not None else "RAM: —")
        set_bar("ram", ram)
        gpu = snapshot.get("gpu") or {}
        gu = gpu.get("util_percent"); gt = gpu.get("temperature_c")
        widgets["gpu"]["value"].setText("—" if gu is None else f"{gu}%")
        gpu_detail = gpu.get("name") or ("GPU bulunamadı" if self.language == "tr" else "GPU not detected")
        gpu_detail += (f" · {gt} °C" if gt is not None else (" · sıcaklık yok" if self.language == "tr" else " · temp unavailable"))
        widgets["gpu"]["detail"].setText(gpu_detail)
        set_bar("gpu", gu)
        vu, vt = gpu.get("memory_used_mb"), gpu.get("memory_total_mb")
        vp = round(vu / vt * 100) if vu is not None and vt else None
        widgets["vram"]["value"].setText("—" if vp is None else f"{vp}%")
        widgets["vram"]["detail"].setText(f"{fmt_gb(vu)} / {fmt_gb(vt)}" if vu is not None and vt is not None else ("VRAM verisi alınamıyor" if self.language == "tr" else "VRAM unavailable"))
        set_bar("vram", vp)

        disks = snapshot.get("disks") or []
        read_rate, write_rate = snapshot.get("disk_read_mb_s"), snapshot.get("disk_write_mb_s")
        disk_lines = []
        for disk in disks:
            disk_lines.append(f"{disk['mount']} {disk['percent']:.0f}% · {disk['free_gb']:.1f}/{disk['total_gb']:.1f} GB boş")
        if read_rate is not None and write_rate is not None:
            disk_lines.append((f"Okuma {read_rate:.2f} MB/sn · Yazma {write_rate:.2f} MB/sn") if self.language == "tr" else (f"Read {read_rate:.2f} MB/s · Write {write_rate:.2f} MB/s"))
        if snapshot.get("disk_temp_c") is not None:
            disk_lines.append(f"Disk sıcaklığı: {snapshot['disk_temp_c']:.0f} °C" if self.language == "tr" else f"Disk temperature: {snapshot['disk_temp_c']:.0f} °C")
        if not disk_lines:
            disk_lines = ["Disk verisi alınamıyor" if self.language == "tr" else "Disk data unavailable"]
        widgets["disks"]["value"].setText(" · ".join(d["mount"] for d in disks) if disks else "—")
        widgets["disks"]["detail"].setText("  |  ".join(disk_lines))

        for key in ("whisper", "gemma", "xtts"):
            model = (snapshot.get("models") or {}).get(key, {})
            running = model.get("status") == "running"
            state = ("ÇALIŞIYOR" if running else "ÇALIŞMIYOR") if self.language == "tr" else ("RUNNING" if running else "OFFLINE")
            ram_mb, ram_private_mb, vram_mb = model.get("ram_mb"), model.get("ram_private_mb"), model.get("vram_mb")
            if ram_mb is not None:
                ram_text = f"RAM WS {ram_mb / 1024:.1f} GB"
                if ram_private_mb is not None:
                    ram_text += f" · özel {ram_private_mb / 1024:.1f} GB" if self.language == "tr" else f" · private {ram_private_mb / 1024:.1f} GB"
            else:
                ram_text = "RAM —"
            if vram_mb is not None:
                source = model.get("vram_source")
                source_label = "Windows" if source == "Windows GPU Process Memory counter" else ("NVIDIA" if source else "")
                vram_text = f"VRAM {vram_mb / 1024:.2f} GB" + (f" · {source_label}" if source_label else "")
            else:
                vram_text = "VRAM ölçülemiyor" if self.language == "tr" else "VRAM unavailable"
            detail = f"PID {model.get('pid') or '—'}\n{ram_text}\n{vram_text}"
            widgets[key]["value"].setText(state)
            widgets[key]["detail"].setText(detail)

        # These rates come only from new llama.cpp timing lines appended during this run.
        # If the active server does not write those lines to gemma_ui.log, show unavailable.
        speed = snapshot.get("inference_speed") or {}
        for key, value_key, tokens_key, time_key, stamp_key in (
            ("prompt_speed", "prompt_tok_s", "prompt_tokens", "prompt_time_ms", "prompt_updated_at"),
            ("generation_speed", "generation_tok_s", "generation_tokens", "generation_time_ms", "generation_updated_at"),
        ):
            value = speed.get(value_key)
            tokens = speed.get(tokens_key)
            elapsed_ms = speed.get(time_key)
            stamp = speed.get(stamp_key)
            widgets[key]["value"].setText("— tok/s" if value is None else f"{value:.1f} tok/s")
            if value is None:
                detail = ("Yeni Gemma ölçümü bekleniyor" if self.language == "tr" else "Waiting for a new Gemma measurement")
            else:
                pieces = []
                if tokens is not None:
                    pieces.append(f"{tokens} token")
                if elapsed_ms is not None:
                    pieces.append(f"{elapsed_ms / 1000.0:.2f} sn" if self.language == "tr" else f"{elapsed_ms / 1000.0:.2f} s")
                if stamp:
                    pieces.append(stamp)
                detail = " · ".join(pieces) or ("Son ölçülen hız" if self.language == "tr" else "Last measured speed")
            widgets[key]["detail"].setText(detail)

        gpu_note = gpu.get("note")
        if gpu_note:
            self.resource_note.setText(gpu_note + (" · Whisper RAM/VRAM değeri NOVA arayüzünü de kapsar." if self.language == "tr" else " · Whisper RAM/VRAM includes the NOVA UI process."))
        elif snapshot.get("errors"):
            self.resource_note.setText(" · ".join(snapshot["errors"]))
        else:
            self.resource_note.setText(self.t("Measured values only · Unavailable sensors are not estimated.") + (" Whisper RAM/VRAM değeri NOVA arayüzünü de kapsar." if self.language == "tr" else " Whisper RAM/VRAM includes the NOVA UI process."))

    def toggle_maximize(self):
        if self.isMaximized():
            self.showNormal(); self.btn_maximize.setText("□")
        else:
            self.showMaximized(); self.btn_maximize.setText("❐")

    def add_conversation_message(self, role, message):
        import html
        text = html.escape(str(message)).replace("\n", "<br>")
        if not text.strip(): return
        if not self._has_chat_message:
            self.chat_view.clear(); self._has_chat_message = True
        if role == "user":
            label, align, bg, fg, edge = ("YOU  /  VOICE INPUT" if self.language == "en" else "SEN  /  SES GİRDİSİ"), "right", "#102b3a", "#e4f8ff", "#2a718a"
        else:
            label, align, bg, fg, edge = ("NOVA  /  RESPONSE" if self.language == "en" else "NOVA  /  YANIT"), "left", "#111e2e", "#e5edf8", "#2a4057"
        block = (f'<table width="100%" cellspacing="0" cellpadding="0"><tr><td align="{align}">'
                 f'<table cellspacing="0" cellpadding="0"><tr><td bgcolor="{bg}" style="border:1px solid {edge}; border-radius:12px; padding:12px 14px; min-width:160px;">'
                 f'<p style="margin:0 0 7px 0; color:#81dff0; font-size:8pt; font-weight:bold; letter-spacing:1px;">{label}</p>'
                 f'<span style="color:{fg}; font-size:10pt; line-height:1.5;">{text}</span>'
                 f'</td></tr></table></td></tr></table><br>')
        self.chat_view.append(block)
        self.chat_view.verticalScrollBar().setValue(self.chat_view.verticalScrollBar().maximum())

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft(); event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            self.move(event.globalPosition().toPoint() - self.drag_position); event.accept()

    def check_servers_health(self):
        gemma_ok = gemma_ready(timeout=0.25)
        self._set_badge(self.gemma_badge, "GEMMA", gemma_ok)
        xtts_ok = tcp_ready("127.0.0.1", 5000, timeout=0.25)
        self._set_badge(self.xtts_badge, "XTTS", xtts_ok)
        if self.worker_started and not (gemma_ok and xtts_ok):
            self.status_desc.setText(self.t("A local service is unavailable. Open diagnostics for details."))

    def _set_badge(self, label, name, online):
        if self.language == "tr":
            state_text = "ÇEVRİMİÇİ" if online else "ÇEVRİMDIŞI"
        else:
            state_text = "ONLINE" if online else "OFFLINE"
        label.setText(f"{name}  ·  {state_text}")
        label.setObjectName("serviceOnline" if online else "serviceOffline")
        label.style().unpolish(label); label.style().polish(label); label.update()

    def poll_server_logs(self):
        error_terms = ("error", "fatal", "failed", "exception", "traceback", "not found", "cannot load", "could not", "cuda error", "unicodeencodeerror", "address already in use")
        for path in (self.server_manager.gemma_log_path, self.server_manager.xtts_log_path):
            try:
                if not path.exists(): continue
                key = str(path); size = path.stat().st_size; offset = self._log_offsets.get(key, 0)
                if size < offset: offset = 0
                if size == offset: continue
                with path.open("r", encoding="utf-8", errors="replace") as handle:
                    handle.seek(offset); chunk = handle.read(); self._log_offsets[key] = handle.tell()
                for line in chunk.splitlines():
                    stripped = line.strip()
                    if stripped and any(term in stripped.lower() for term in error_terms):
                        self.append_log(f"[SERVER WARNING] {path.stem}: {stripped}")
            except Exception as exc:
                self.append_log(f"[WARN] Could not read server log ({path.name}): {exc}")

    def auto_start_system(self):
        self.append_log("[START] Checking local inference and voice servers...")
        self.server_manager.start_servers()
        self.append_log("[INFO] Healthy services are reused; only missing services are started.")

    def try_start_worker(self):
        if self.worker_started:
            self.startup_timer.stop(); return
        gemma_ok = gemma_ready(timeout=0.2); xtts_ok = tcp_ready("127.0.0.1", 5000, timeout=0.2)
        if gemma_ok and xtts_ok:
            self.startup_timer.stop(); self.append_log("[READY] Gemma and XTTS are ready. Starting voice worker...")
            self.start_jarvis_worker()
        else:
            self.status_desc.setText((f"Waiting for local services · Gemma {'ready' if gemma_ok else 'loading'} / XTTS {'ready' if xtts_ok else 'loading'}") if self.language == "en" else (f"Yerel hizmetler bekleniyor · Gemma {'hazır' if gemma_ok else 'yükleniyor'} / XTTS {'hazır' if xtts_ok else 'yükleniyor'}"))
            self.intro_status.setText(self.t("CONNECTING"))

    def start_jarvis_worker(self):
        if self.worker_started: return
        self.worker_started = True
        self.worker = JarvisWorker()
        self.worker.log_signal.connect(self.append_log)
        self.worker.conversation_signal.connect(self.receive_conversation_message)
        self.worker.status_signal.connect(self.update_status)
        self.worker.start()

    @Slot(str)
    def append_log(self, text):
        self.log_box.append(str(text))

    @Slot(str, str)
    def receive_conversation_message(self, role, text):
        self.chat_view.setUpdatesEnabled(False)
        try: self.add_conversation_message(role, text)
        finally: self.chat_view.setUpdatesEnabled(True)

    @Slot(str, str)
    def update_status(self, text, state):
        self.orb.set_state(state)
        # Avoid shouty, awkward all-caps statuses; use short calm language.
        status_map = {
            "idle": (self.t("Ready when you are"), self.t("Listening will resume automatically.")),
            "listening": (self.t("Listening"), self.t("I’m here. Speak naturally.")),
            "thinking": (self.t("Thinking it through"), self.t("Your request is being processed locally.")),
            "speaking": (self.t("Speaking"), self.t("Preparing and playing the response.")),
        }
        if state == "idle" and "HATA" in text.upper():
            title, desc = (self.t("Needs attention"), text)
        elif state == "idle" and "HATA" not in text.upper() and "HAZIR" not in text.upper():
            title, desc = (self.t("Starting up"), text)
        else:
            title, desc = status_map.get(state, (text.title(), self.t("Local voice session")))
        self.status_label.setText(title)
        self.status_desc.setText(desc)
        self.intro_status.setText(title.upper())
        if state == "listening" and not self._first_listening_seen:
            self._first_listening_seen = True
            self.intro_strip.hide()
            self.brand_subtitle.hide()
            self.header_private.hide()
            self.footer_right.setText(self.t("READY · LOCAL SESSION"))
        # Give the minimal waveform a gentle state-based variation.
        for i, bar in enumerate(self.wave_bars):
            heights = {"idle": 5 + (i*7 % 10), "listening": 8 + (i*5 % 18), "thinking": 5 + (i*9 % 22), "speaking": 7 + (i*11 % 24)}
            bar.setFixedHeight(heights.get(state, 8))

    def closeEvent(self, event):
        if self._shutdown_complete:
            event.accept(); return
        if self.worker and self.worker.isRunning():
            event.ignore()
            if not self._close_pending:
                self._close_pending = True
                self.append_log("[SHUTDOWN] Stopping the microphone loop safely. Please wait a few seconds...")
                self.worker.stop(); self.worker.finished.connect(self._finish_close)
            return
        if hasattr(self, "resource_thread") and self.resource_thread.isRunning():
            self.resource_thread.stop(); self.resource_thread.wait(5000)
        self.server_manager.stop_servers(); self._shutdown_complete = True; event.accept()

    @Slot()
    def _finish_close(self):
        if hasattr(self, "resource_thread") and self.resource_thread.isRunning():
            self.resource_thread.stop(); self.resource_thread.wait(5000)
        self.server_manager.stop_servers(); self._shutdown_complete = True; self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NovaMainWindow(ServerManager())
    window.show()
    sys.exit(app.exec())
