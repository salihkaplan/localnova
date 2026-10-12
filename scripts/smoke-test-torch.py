import sys
try:
    import torch
    print(f"TORCH_VERSION={torch.__version__}")
    print(f"TORCH_CUDA_BUILD={torch.version.cuda}")
    print(f"CUDA_AVAILABLE={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA kullanilabilir degil. NVIDIA surucusu ve cu126 wheel uyumunu kontrol edin.")
    print(f"GPU={torch.cuda.get_device_name(0)}")
    x = torch.ones((128, 128), device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print(f"CUDA_MATMUL_OK={float(y[0,0].item())}")
except Exception as exc:
    print(f"TORCH_CUDA_TEST_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
    raise
