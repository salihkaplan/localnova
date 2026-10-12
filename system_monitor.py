"""Read-only NOVA resource telemetry. Reports measured values only; unavailable metrics stay None.

Per-process GPU memory is best-effort: Windows WDDM/drivers may not expose it. The NOVA
process contains both Whisper and the UI, so its GPU-memory value is explicitly process-level,
not a Whisper-only allocation.
"""
from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any


def _safe_float(value: Any):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _creation_flags():
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def _apply_gemma_timing_line(line: str, speed: dict, updated_at: str | None = None) -> bool:
    """Update prompt/generation tok/s from llama.cpp's timing log lines only.

    The values are copied from measured server timings. This deliberately does not infer
    token speeds from whole-request latency or from model size.
    """
    stamp = updated_at or time.strftime("%H:%M:%S")
    prompt_eval = re.search(
        r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens.*?([\d.]+)\s*tokens per second",
        line, re.IGNORECASE,
    )
    if prompt_eval:
        speed.update({
            "prompt_tok_s": float(prompt_eval.group(3)),
            "prompt_tokens": int(prompt_eval.group(2)),
            "prompt_time_ms": float(prompt_eval.group(1)),
            "prompt_updated_at": stamp,
            "source": "llama.cpp timing log",
        })
        return True

    # Some llama.cpp builds emit an in-progress summary in this alternate format.
    prompt_progress = re.search(
        r"prompt processing.*?n_tokens\s*=\s*(\d+).*?t\s*=\s*([\d.]+)\s*s\s*/\s*([\d.]+)\s*tokens per second",
        line, re.IGNORECASE,
    )
    if prompt_progress:
        speed.update({
            "prompt_tok_s": float(prompt_progress.group(3)),
            "prompt_tokens": int(prompt_progress.group(1)),
            "prompt_time_ms": float(prompt_progress.group(2)) * 1000.0,
            "prompt_updated_at": stamp,
            "source": "llama.cpp timing log",
        })
        return True

    generation = re.search(
        r"(?<!prompt )eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens.*?([\d.]+)\s*tokens per second",
        line, re.IGNORECASE,
    )
    if generation:
        speed.update({
            "generation_tok_s": float(generation.group(3)),
            "generation_tokens": int(generation.group(2)),
            "generation_time_ms": float(generation.group(1)),
            "generation_updated_at": stamp,
            "source": "llama.cpp timing log",
        })
        return True
    return False


def _private_memory_mb(proc):
    """Return private resident memory (USS) where Windows/psutil exposes it."""
    try:
        full = proc.memory_full_info()
        value = getattr(full, "uss", None)
        return round(value / 1048576, 1) if value is not None and value >= 0 else None
    except Exception:
        return None


def _listener_pids(psutil, ports=(5000, 8080)):
    """Best-effort map of local listening ports to PIDs; process IDs change between runs."""
    result = {int(port): set() for port in ports}
    try:
        for connection in psutil.net_connections(kind="inet"):
            try:
                local = connection.laddr
                port = int(getattr(local, "port", 0) or 0)
                pid = _safe_int(getattr(connection, "pid", None))
                if (port in result and pid is not None and
                        connection.status == getattr(psutil, "CONN_LISTEN", "LISTEN")):
                    result[port].add(pid)
            except Exception:
                continue
    except Exception:
        pass
    return result


def _gpu_snapshot():
    result = {"available": False, "name": None, "util_percent": None,
              "memory_used_mb": None, "memory_total_mb": None, "temperature_c": None,
              "process_vram_mb": {}, "source": "unavailable", "note": None}
    try:
        import pynvml  # optional; do not require it for NOVA startup
        pynvml.nvmlInit()
        try:
            count = pynvml.nvmlDeviceGetCount()
            if count < 1:
                return result
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            result["available"] = True
            result["name"] = pynvml.nvmlDeviceGetName(handle)
            if isinstance(result["name"], bytes):
                result["name"] = result["name"].decode("utf-8", "replace")
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                result["util_percent"] = int(util.gpu)
            except Exception:
                pass
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                result["memory_used_mb"] = round(mem.used / 1048576, 1)
                result["memory_total_mb"] = round(mem.total / 1048576, 1)
            except Exception:
                pass
            try:
                result["temperature_c"] = int(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
            except Exception:
                pass
            process_functions = []
            for fn_name in ("nvmlDeviceGetComputeRunningProcesses_v3", "nvmlDeviceGetGraphicsRunningProcesses_v3",
                            "nvmlDeviceGetComputeRunningProcesses", "nvmlDeviceGetGraphicsRunningProcesses"):
                fn = getattr(pynvml, fn_name, None)
                if fn and fn not in process_functions:
                    process_functions.append(fn)
            for fn in process_functions:
                try:
                    for proc in fn(handle):
                        pid = _safe_int(getattr(proc, "pid", None))
                        used = getattr(proc, "usedGpuMemory", None)
                        # NVML_VALUE_NOT_AVAILABLE is a very large sentinel on some platforms.
                        if pid is not None and used is not None and 0 <= used < (1 << 50):
                            result["process_vram_mb"][pid] = max(
                                result["process_vram_mb"].get(pid, 0.0), round(used / 1048576, 1))
                except Exception:
                    continue
            result["source"] = "NVML"
            if not result["process_vram_mb"]:
                result["note"] = "Sürücü süreç bazında VRAM sunmuyor olabilir (WDDM)."
            return result
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    except Exception:
        pass

    # NVIDIA-SMI fallback for total GPU metrics and compute-process VRAM.
    try:
        cmd = ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
               "--format=csv,noheader,nounits"]
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=1.2,
                                   creationflags=_creation_flags())
        if completed.returncode == 0 and completed.stdout.strip():
            cols = [part.strip() for part in completed.stdout.splitlines()[0].split(",")]
            if len(cols) >= 5:
                result.update({"available": True, "name": cols[0], "util_percent": _safe_int(cols[1]),
                               "memory_used_mb": _safe_float(cols[2]), "memory_total_mb": _safe_float(cols[3]),
                               "temperature_c": _safe_int(cols[4]), "source": "nvidia-smi"})
        proc_cmd = ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]
        proc_result = subprocess.run(proc_cmd, capture_output=True, text=True, timeout=1.2,
                                     creationflags=_creation_flags())
        if proc_result.returncode == 0:
            for line in proc_result.stdout.splitlines():
                cols = [part.strip() for part in line.split(",")]
                if len(cols) >= 2:
                    pid, used = _safe_int(cols[0]), _safe_float(cols[1])
                    if pid is not None and used is not None and used >= 0:
                        result["process_vram_mb"][pid] = used
        if result["available"] and not result["process_vram_mb"]:
            result["note"] = "Sürücü süreç bazında VRAM sunmuyor olabilir (WDDM)."
    except Exception:
        pass
    return result


def _parse_windows_gpu_counter_rows(rows, target_pids, total_mb):
    """Parse Windows WDDM per-process Dedicated Usage rows with conservative validation.

    The Windows counter can over-report for some processes. Reject values that exceed the
    physical adapter capacity; never use these counters to calculate total adapter usage.
    """
    target_pids = {int(pid) for pid in target_pids if _safe_int(pid) is not None}
    result = {}
    if not target_pids or total_mb is None or total_mb <= 0:
        return result
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        instance = str(row.get("InstanceName", row.get("instance", "")))
        match = re.search(r"(?:^|_)pid_(\d+)_", instance, re.IGNORECASE)
        pid = _safe_int(row.get("pid")) if row.get("pid") is not None else (_safe_int(match.group(1)) if match else None)
        raw = _safe_float(row.get("CookedValue", row.get("value")))
        if pid not in target_pids or raw is None or raw < 0:
            continue
        # Get-Counter returns bytes for this counter. Discard known absurd values.
        mb = raw / 1048576.0
        if mb > total_mb * 1.05:
            continue
        result[pid] = max(result.get(pid, 0.0), round(mb, 1))
    return result


def _windows_gpu_process_vram(target_pids, total_mb, timeout=1.5):
    """Read process Dedicated Usage from Windows WDDM counters as a validated fallback.

    NVML/nvidia-smi cannot reliably report per-process memory under WDDM. Windows counters
    can expose PID values, but some versions report impossible values; validation is required.
    """
    if os.name != "nt" or not target_pids or total_mb is None:
        return {}, None
    import json
    # Only return the target process instances to keep the PowerShell payload small.
    ids = sorted({int(pid) for pid in target_pids if _safe_int(pid) is not None})
    id_list = ",".join(str(pid) for pid in ids)
    script = (
        f"$targets=@({id_list}); "
        "$rows=@(Get-Counter '\\GPU Process Memory(*)\\Dedicated Usage' -MaxSamples 1 -ErrorAction Stop "
        "| Select-Object -ExpandProperty CounterSamples | ForEach-Object { "
        "$inst=[string]$_.InstanceName; if ($inst -match 'pid_(\\d+)_') { "
        "$p=[int]$Matches[1]; if ($targets -contains $p) { "
        "[pscustomobject]@{pid=$p; InstanceName=$inst; CookedValue=[double]$_.CookedValue} } } }); "
        "$rows | ConvertTo-Json -Compress"
    )
    try:
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, timeout=timeout, creationflags=_creation_flags(),
        )
        if completed.returncode != 0 or not completed.stdout.strip():
            return {}, None
        rows = json.loads(completed.stdout.strip())
        if isinstance(rows, dict):
            rows = [rows]
        parsed = _parse_windows_gpu_counter_rows(rows, ids, total_mb)
        return parsed, ("Windows GPU Process Memory counter" if parsed else None)
    except Exception:
        return {}, None

def _temperature_snapshot(psutil):
    out = {"cpu_c": None, "disk_c": None, "source": "not available"}
    try:
        sensors = psutil.sensors_temperatures(fahrenheit=False) or {}
        cpu_values, disk_values = [], []
        for name, entries in sensors.items():
            lname = name.lower()
            for entry in entries or []:
                current = _safe_float(getattr(entry, "current", None))
                if current is None or current < -20 or current > 150:
                    continue
                label = (getattr(entry, "label", "") or "").lower()
                if any(token in lname + " " + label for token in ("cpu", "core", "k10temp", "package", "tctl", "tdie", "acpi")):
                    cpu_values.append(current)
                if any(token in lname + " " + label for token in ("nvme", "ssd", "disk", "drive")):
                    disk_values.append(current)
        if cpu_values:
            out["cpu_c"] = round(max(cpu_values), 1)
        if disk_values:
            out["disk_c"] = round(max(disk_values), 1)
        if cpu_values or disk_values:
            out["source"] = "psutil sensors"
    except Exception:
        pass
    return out


class ResourceCollector:
    """Collect system metrics and best-effort model-process metrics without changing system state."""
    def __init__(self):
        self._last_io = None
        self._last_io_time = None
        self._temp_cache = {"cpu_c": None, "disk_c": None, "source": "not available"}
        self._temp_cache_time = 0.0
        self._gemma_log_path = Path(__file__).resolve().parent / "gemma_ui.log"
        self._gemma_log_offset = None
        self._win_vram_cache = {}
        self._win_vram_cache_time = 0.0
        self._win_vram_cache_source = None
        self._inference_speed = {
            "prompt_tok_s": None, "prompt_tokens": None, "prompt_time_ms": None,
            "prompt_updated_at": None, "generation_tok_s": None,
            "generation_tokens": None, "generation_time_ms": None,
            "generation_updated_at": None, "source": "waiting for new Gemma timing data",
        }
        try:
            import psutil
            self._psutil = psutil
            psutil.cpu_percent(interval=None)  # prime non-blocking CPU measurement
        except Exception:
            self._psutil = None

    def _read_gemma_timings(self):
        """Read only newly appended Gemma log lines, avoiding stale historical speeds."""
        path = self._gemma_log_path
        try:
            if not path.is_file():
                return dict(self._inference_speed)
            size = path.stat().st_size
            # On first observation, skip pre-existing content: it may belong to an old run.
            if self._gemma_log_offset is None:
                self._gemma_log_offset = size
                return dict(self._inference_speed)
            if size < self._gemma_log_offset:
                # Log was truncated/recreated by a new server process.
                self._gemma_log_offset = 0
            with path.open("rb") as handle:
                handle.seek(self._gemma_log_offset)
                chunk = handle.read()
                self._gemma_log_offset = handle.tell()
            if chunk:
                for line in chunk.decode("utf-8", "replace").splitlines():
                    _apply_gemma_timing_line(line, self._inference_speed)
        except (OSError, PermissionError):
            pass
        return dict(self._inference_speed)

    def _optional_windows_hardware_sensors(self, now):
        """Read CPU/disk temps from an already-running LibreHardwareMonitor/OpenHardwareMonitor WMI provider.

        This is optional and cached for ten seconds. NOVA does not install, launch, or require the provider.
        """
        if now - self._temp_cache_time < 10.0:
            return dict(self._temp_cache)
        self._temp_cache_time = now
        result = {"cpu_c": None, "disk_c": None, "source": "not available"}
        script = ("$items=@(); foreach($ns in @('root/LibreHardwareMonitor','root/OpenHardwareMonitor')) { "
                  "try { $items += Get-CimInstance -Namespace $ns -ClassName Sensor -ErrorAction Stop | "
                  "Select-Object @{n='SensorType';e={$_.SensorType.ToString()}},Name,Value } catch {} }; "
                  "$items | ConvertTo-Json -Compress")
        try:
            completed = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                                       capture_output=True, text=True, timeout=1.5,
                                       creationflags=_creation_flags())
            if completed.returncode == 0 and completed.stdout.strip():
                import json
                rows = json.loads(completed.stdout.strip())
                if isinstance(rows, dict):
                    rows = [rows]
                cpu, disk = [], []
                for row in rows or []:
                    if str(row.get("SensorType", "")).lower() != "temperature":
                        continue
                    value = _safe_float(row.get("Value"))
                    if value is None or value < -20 or value > 150:
                        continue
                    name = str(row.get("Name", "")).lower()
                    if any(token in name for token in ("cpu package", "cpu core", "cpu tctl", "cpu tdie", "cpu (tctl", "cpu (tdie")):
                        cpu.append(value)
                    if any(token in name for token in ("nvme", "ssd", "hdd", "drive", "disk")):
                        disk.append(value)
                if cpu:
                    result["cpu_c"] = round(max(cpu), 1)
                if disk:
                    result["disk_c"] = round(max(disk), 1)
                if cpu or disk:
                    result["source"] = "LibreHardwareMonitor/OpenHardwareMonitor WMI"
        except Exception:
            pass
        self._temp_cache = dict(result)
        return result

    def _read_windows_process_vram(self, target_pids, total_mb, now):
        # PowerShell counter reads are comparatively expensive; refresh at most every 3 seconds.
        if now - self._win_vram_cache_time >= 3.0:
            values, source = _windows_gpu_process_vram(target_pids, total_mb)
            if values:
                self._win_vram_cache = values
                self._win_vram_cache_source = source
            elif source is None:
                # Do not erase a very recent valid snapshot on a transient counter failure.
                if now - self._win_vram_cache_time > 6.0:
                    self._win_vram_cache = {}
                    self._win_vram_cache_source = None
            self._win_vram_cache_time = now
        return ({pid: value for pid, value in self._win_vram_cache.items() if pid in set(target_pids)},
                self._win_vram_cache_source)

    def collect(self):
        now = time.monotonic()
        snap = {"timestamp": time.strftime("%H:%M:%S"), "cpu_percent": None, "ram_percent": None,
                "ram_used_mb": None, "ram_total_mb": None, "cpu_temp_c": None, "disk_temp_c": None,
                "temperature_source": "not available", "disks": [], "gpu": _gpu_snapshot(),
                "models": {"whisper": {}, "gemma": {}, "xtts": {}},
                "inference_speed": self._read_gemma_timings(), "errors": []}
        psutil = self._psutil
        if psutil is None:
            snap["errors"].append("psutil modülü kurulu değil; CPU/RAM/disk ölçümü alınamadı.")
            return snap
        try:
            snap["cpu_percent"] = round(psutil.cpu_percent(interval=None), 1)
            vm = psutil.virtual_memory()
            snap["ram_percent"] = round(vm.percent, 1)
            snap["ram_used_mb"] = round(vm.used / 1048576, 1)
            snap["ram_total_mb"] = round(vm.total / 1048576, 1)
        except Exception as exc:
            snap["errors"].append(f"CPU/RAM: {type(exc).__name__}")

        temps = _temperature_snapshot(psutil)
        if temps["cpu_c"] is None and temps["disk_c"] is None and os.name == "nt":
            temps = self._optional_windows_hardware_sensors(now)
        snap["cpu_temp_c"] = temps["cpu_c"]
        snap["disk_temp_c"] = temps["disk_c"]
        snap["temperature_source"] = temps["source"]

        try:
            seen = set()
            for part in psutil.disk_partitions(all=False):
                mount = part.mountpoint
                if not mount or mount in seen or (os.name == "nt" and "fixed" not in (part.opts or "").lower() and not mount[:1].isalpha()):
                    continue
                seen.add(mount)
                try:
                    usage = psutil.disk_usage(mount)
                    snap["disks"].append({"mount": mount, "percent": round(usage.percent, 1),
                        "used_gb": round(usage.used / (1024**3), 1), "total_gb": round(usage.total / (1024**3), 1),
                        "free_gb": round(usage.free / (1024**3), 1)})
                except Exception:
                    continue
        except Exception as exc:
            snap["errors"].append(f"Disk: {type(exc).__name__}")

        try:
            io = psutil.disk_io_counters()
            if io is not None:
                if self._last_io is not None and self._last_io_time is not None:
                    elapsed = max(0.1, now - self._last_io_time)
                    snap["disk_read_mb_s"] = round(max(0, io.read_bytes - self._last_io.read_bytes) / 1048576 / elapsed, 2)
                    snap["disk_write_mb_s"] = round(max(0, io.write_bytes - self._last_io.write_bytes) / 1048576 / elapsed, 2)
                else:
                    snap["disk_read_mb_s"] = None
                    snap["disk_write_mb_s"] = None
                self._last_io, self._last_io_time = io, now
        except Exception:
            snap["disk_read_mb_s"] = snap["disk_write_mb_s"] = None

        # Process memory is measurable in RAM. GPU memory is only assigned when the driver
        # reports PID-level values; it is never estimated by splitting total VRAM.
        current_pid = os.getpid()
        listener_pids = _listener_pids(psutil, ports=(5000, 8080))
        gemma_listener_pids = listener_pids.get(8080, set())
        xtts_listener_pids = listener_pids.get(5000, set())
        proc_rows = {}
        try:
            for proc in psutil.process_iter(["pid", "name", "cmdline", "memory_info"]):
                try:
                    info = proc.info
                    pid = int(info.get("pid") or 0)
                    name = (info.get("name") or "").lower()
                    cmdline = " ".join(info.get("cmdline") or []).lower()
                    mem_info = info.get("memory_info")
                    rss_mb = round(mem_info.rss / 1048576, 1) if mem_info else None
                    if pid == current_pid:
                        proc_rows["whisper"] = {"pid": pid, "process_name": "NOVA / Whisper + UI", "ram_mb": rss_mb,
                                                "ram_private_mb": _private_memory_mb(proc)}
                    if "llama-server" in name or "llama-server.exe" in cmdline:
                        existing = proc_rows.get("gemma")
                        candidate = {"pid": pid, "process_name": "llama-server", "ram_mb": rss_mb,
                                     "ram_private_mb": _private_memory_mb(proc)}
                        existing_is_listener = existing is not None and existing.get("pid") in gemma_listener_pids
                        candidate_is_listener = pid in gemma_listener_pids
                        if (existing is None or (candidate_is_listener and not existing_is_listener) or
                                (candidate_is_listener == existing_is_listener and (rss_mb or 0) > (existing.get("ram_mb") or 0))):
                            proc_rows["gemma"] = candidate
                    if "speech_service.py" in cmdline:
                        existing = proc_rows.get("xtts")
                        candidate = {"pid": pid, "process_name": "XTTS Python server", "ram_mb": rss_mb,
                                     "ram_private_mb": _private_memory_mb(proc)}
                        existing_is_listener = existing is not None and existing.get("pid") in xtts_listener_pids
                        candidate_is_listener = pid in xtts_listener_pids
                        if (existing is None or (candidate_is_listener and not existing_is_listener) or
                                (candidate_is_listener == existing_is_listener and (rss_mb or 0) > (existing.get("ram_mb") or 0))):
                            proc_rows["xtts"] = candidate
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, OSError):
                    continue
        except Exception as exc:
            snap["errors"].append(f"Processes: {type(exc).__name__}")

        vram_by_pid = snap["gpu"].get("process_vram_mb") or {}
        # NVIDIA NVML often cannot expose process memory on Windows WDDM. Try Windows' own
        # per-process counter for only the detected model PIDs, and reject values above adapter capacity.
        target_pids = [row.get("pid") for row in proc_rows.values() if row.get("pid")]
        total_mb = snap["gpu"].get("memory_total_mb")
        windows_vram, windows_vram_source = self._read_windows_process_vram(target_pids, total_mb, now)
        for pid, value in windows_vram.items():
            if pid not in vram_by_pid:
                vram_by_pid[pid] = value
        if windows_vram:
            snap["gpu"]["note"] = "Süreç VRAM değerleri Windows WDDM sayacından; kapasite dışı değerler elendi. Süreç değerleri toplam GPU belleğiyle toplanmamalıdır."
        for key in ("whisper", "gemma", "xtts"):
            model = proc_rows.get(key)
            if model:
                model["vram_mb"] = vram_by_pid.get(model["pid"])
                model["vram_status"] = "measured" if model["vram_mb"] is not None else "unavailable"
                model["vram_source"] = ("NVML/nvidia-smi" if model.get("pid") in (snap["gpu"].get("process_vram_mb") or {})
                                         else windows_vram_source if model.get("pid") in windows_vram else None)
                model["status"] = "running"
            else:
                model = {"pid": None, "process_name": {"whisper": "NOVA / Whisper + UI", "gemma": "llama-server", "xtts": "XTTS Python server"}[key],
                         "ram_mb": None, "ram_private_mb": None, "vram_mb": None, "vram_status": "unavailable", "vram_source": None, "status": "not detected"}
            snap["models"][key] = model
        snap["gpu"].pop("process_vram_mb", None)
        return snap
