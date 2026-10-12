import os
import time
import shutil
from pathlib import Path
import numpy as np
import torch
from flask import Flask, request, Response, jsonify
from TTS.api import TTS

# FFmpeg is optional at import time and can be configured per installation.
# Search order: NOVA_FFMPEG_BIN, project-local ffmpeg/bin, then existing PATH.
_PROJECT_DIR = Path(__file__).resolve().parent
_ffmpeg_override = os.environ.get("NOVA_FFMPEG_BIN", "").strip().strip('"')
_ffmpeg_candidates = []
if _ffmpeg_override:
    _candidate = Path(_ffmpeg_override).expanduser()
    if not _candidate.is_absolute():
        _candidate = _PROJECT_DIR / _candidate
    _ffmpeg_candidates.append(_candidate)
_ffmpeg_candidates.append(_PROJECT_DIR / "ffmpeg" / "bin")
_ffmpeg_path = shutil.which("ffmpeg")
if _ffmpeg_path:
    _ffmpeg_candidates.append(Path(_ffmpeg_path).resolve().parent)
FFMPEG_BIN = next((str(candidate.resolve()) for candidate in _ffmpeg_candidates
                   if (candidate / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")).is_file()), "")
if FFMPEG_BIN:
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(FFMPEG_BIN)
        except OSError:
            pass
    os.environ["PATH"] = FFMPEG_BIN + os.pathsep + os.environ.get("PATH", "")

app = Flask(__name__)

print("XTTS yükleniyor...")

tts = TTS(
    "tts_models/multilingual/multi-dataset/xtts_v2"
).to("cuda")

# TTS API'nin altında gerçek XTTS modeline erişiyoruz.
model = tts.synthesizer.tts_model

# Yerleşik speaker latent'larını model belleğinden bir kez alıyoruz.
speaker_data = model.speaker_manager.speakers["Viktor Eka"]
gpt_cond_latent = speaker_data["gpt_cond_latent"]
speaker_embedding = speaker_data["speaker_embedding"]

print("XTTS hazır.")
print("Viktor Eka latent'ları önbelleğe alındı.")

def xtts_cuda_memory():
    """PyTorch'un bu süreçte izlediği CUDA belleğini döndürür.
    reserved - allocated, PyTorch'un ayırdığı ama aktif tensorlarda
    kullanılmayan yaklaşık alanı gösterir; toplam GPU cache'i değildir.
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA kullanilabilir degil"}

    device = torch.cuda.current_device()
    torch.cuda.synchronize(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)

    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)

    try:
        device_name = torch.cuda.get_device_name(device)
    except Exception:
        device_name = f"cuda:{device}"

    mib = 1024 ** 2
    return {
        "device": device_name,
        "allocated_mb": round(allocated / mib, 1),
        "reserved_mb": round(reserved / mib, 1),
        "cache_estimate_mb": round(max(0, reserved - allocated) / mib, 1),
        "peak_allocated_mb": round(peak_allocated / mib, 1),
        "peak_reserved_mb": round(peak_reserved / mib, 1),
        "gpu_total_mb": round(total_bytes / mib, 1),
        "gpu_free_mb": round(free_bytes / mib, 1),
    }


def log_xtts_cuda_memory(stage, reset_peak=False):
    """Bellek ölçümünü konsola yazar; ölçüm hatası TTS akışını bozmaz."""
    try:
        if not torch.cuda.is_available():
            print(f"[XTTS VRAM][{stage}] CUDA kullanilabilir degil.", flush=True)
            return

        if reset_peak:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())

        stats = xtts_cuda_memory()
        print(f"[XTTS VRAM][{stage}] {stats}", flush=True)
    except Exception as exc:
        print(f"[XTTS VRAM][{stage}] Ölçüm alınamadı: {exc}", flush=True)


@app.route("/memory", methods=["GET"])
def memory_status():
    """Tarayıcıdan veya PowerShell'den anlık XTTS/PyTorch bellek durumu."""
    try:
        return jsonify(xtts_cuda_memory())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/memory_detail", methods=["GET"])
def memory_detail():
    """PyTorch allocator ayrıntıları; model üretimini değiştirmez."""
    try:
        if not torch.cuda.is_available():
            return jsonify({"error": "CUDA kullanilabilir degil"}), 503

        device = torch.cuda.current_device()
        torch.cuda.synchronize(device)
        stats = torch.cuda.memory_stats(device)

        # Yalnızca tanı için yararlı, sabit ve küçük bir alan kümesi döndür.
        keys = (
            "allocated_bytes.all.current",
            "allocated_bytes.all.peak",
            "active_bytes.all.current",
            "active_bytes.all.peak",
            "reserved_bytes.all.current",
            "reserved_bytes.all.peak",
            "inactive_split_bytes.all.current",
            "inactive_split_bytes.all.peak",
            "requested_bytes.all.current",
            "num_alloc_retries",
            "num_ooms",
        )
        mib = 1024 ** 2
        selected_stats = {}
        for key in keys:
            value = stats.get(key)
            if value is not None:
                selected_stats[key] = (
                    round(value / mib, 1)
                    if "bytes" in key
                    else int(value)
                )

        return jsonify({
            "memory": xtts_cuda_memory(),
            "allocator_stats": selected_stats,
            "memory_summary": torch.cuda.memory_summary(
                device=device, abbreviated=False
            ),
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


log_xtts_cuda_memory("MODEL_HAZIR")




def trim_trailing_silence(audio, sample_rate=24000, threshold=0.003,
                          min_keep_seconds=0.08, max_trim_seconds=0.35):
    """Sadece sesin en sonundaki gereksiz kuyruğu kırpar.
    Cümle içindeki duraklamalara dokunmaz.
    """
    if audio is None or len(audio) == 0:
        return audio

    abs_audio = np.abs(audio)
    peak = float(np.max(abs_audio))

    if peak <= 0:
        return audio

    # Eşiği kayda göre biraz ölçekle; çok sessiz konuşmaları da koru.
    effective_threshold = max(threshold, peak * 0.015)

    # Sondan itibaren eşik üstü son örneği bul.
    active = np.where(abs_audio >= effective_threshold)[0]
    if len(active) == 0:
        return audio

    last_active = int(active[-1])
    keep_samples = int(min_keep_seconds * sample_rate)
    max_trim_samples = int(max_trim_seconds * sample_rate)

    target_end = min(len(audio), last_active + keep_samples)
    trim_amount = len(audio) - target_end

    # Aşırı agresif kırpmayı engelle.
    if trim_amount > max_trim_samples:
        target_end = len(audio) - max_trim_samples

    return audio[:target_end]


@app.route("/tts_stream", methods=["POST"])
def tts_stream():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()

    if not text:
        return Response("Metin boş", status=400)

    speed = float(data.get("speed", 1.15))
    chunk_size = int(data.get("chunk_size", 5))

    def generate():
        start = time.perf_counter()
        first = True
        total_samples = 0
        chunk_count = 0

        log_xtts_cuda_memory("STREAM_BASLANGIC", reset_peak=True)
        try:
            with torch.inference_mode():
                chunks = model.inference_stream(
                    text=text,
                    language="tr",
                    gpt_cond_latent=gpt_cond_latent,
                    speaker_embedding=speaker_embedding,
                    stream_chunk_size=chunk_size,
                    speed=speed,
                    enable_text_splitting=True,
                )

                for chunk in chunks:
                    audio = chunk.detach().float().cpu().numpy().astype(np.float32)
                    chunk_count += 1
                    total_samples += len(audio)

                    if first:
                        print(
                            f"[XTTS] İlk chunk: "
                            f"{time.perf_counter() - start:.2f} sn",
                            flush=True,
                        )
                        first = False

                    # Gerçek streaming: chunk'ı hemen client'a gönder.
                    yield audio.tobytes()
        finally:
            log_xtts_cuda_memory("STREAM_BITIS")

        elapsed = time.perf_counter() - start
        audio_seconds = total_samples / 24000.0

        print(
            f"[XTTS] Üretim: {elapsed:.2f} sn | "
            f"ses: {audio_seconds:.2f}s | chunk: {chunk_count}"
        )


    return Response(
        generate(),
        mimetype="application/octet-stream",
        headers={
            "X-Sample-Rate": "24000",
            "X-Audio-Format": "float32",
        },
    )


@app.route("/tts", methods=["POST"])
def tts_full():
    """Eski yöntemi koruyan basit tam-WAV endpoint'i."""
    import uuid
    import soundfile as sf

    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()

    if not text:
        return {"error": "Metin boş"}, 400

    filename = f"jarvis_{uuid.uuid4().hex}.wav"

    log_xtts_cuda_memory("TTS_BASLANGIC", reset_peak=True)
    try:
        with torch.inference_mode():
            out = model.inference(
                text=text,
                language="tr",
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                speed=1.15,
            )
    finally:
        log_xtts_cuda_memory("TTS_BITIS")

    sf.write(filename, out["wav"], 24000)

    return {"success": True, "file": os.path.abspath(filename)}


app.run(host="127.0.0.1", port=5000, threaded=True)
