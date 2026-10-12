# Verify the exact Windows DLL preload sequence used by nova_app.py, then load Whisper on CUDA.
import ctypes, os, sys, traceback
from pathlib import Path
root = Path(sys.argv[1]).resolve()
log = root / 'logs' / 'whisper-smoke.log'
log.parent.mkdir(parents=True, exist_ok=True)
def out(s):
    print(s, flush=True)
    with log.open('a', encoding='utf-8') as f: f.write(s + '\n')
try:
    if os.name != 'nt': raise RuntimeError('Bu smoke test Windows icindir.')
    site = root / 'tts_env' / 'Lib' / 'site-packages'
    dirs = [site/'torch'/'lib', site/'nvidia'/'cuda_runtime'/'bin', site/'nvidia'/'cublas'/'bin', site/'nvidia'/'cudnn'/'bin']
    handles=[]; dll_handles=[]
    for d in dirs:
        if d.is_dir():
            handles.append(os.add_dll_directory(str(d)))
            out('[DLL_DIR] ' + str(d))
    for name in ('cublas64_12.dll','cublasLt64_12.dll'):
        candidates=[site/'torch'/'lib'/name, site/'nvidia'/'cublas'/'bin'/name]
        path=next((p for p in candidates if p.is_file()),None)
        if path is None: raise FileNotFoundError(f'{name} bulunamadi. Kontrol edilen: ' + ', '.join(map(str,candidates)))
        dll_handles.append(ctypes.WinDLL(str(path)))
        out('[DLL_LOAD_OK] ' + str(path))
    from faster_whisper import WhisperModel
    out('[WHISPER] small modeli CUDA/float16 ile yukleniyor; ilk calismada model indirilebilir.')
    model=WhisperModel('small', device='cuda', compute_type='float16', download_root=None)
    out('[WHISPER_MODEL_LOAD_OK] faster-whisper / CTranslate2 CUDA modeli basariyla yuklendi.')
except Exception as exc:
    out(f'[WHISPER_TEST_FAILED] {type(exc).__name__}: {exc}')
    out(traceback.format_exc())
    sys.exit(1)
