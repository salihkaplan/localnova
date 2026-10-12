"""Windows master-volume actions for NOVA. Requires pycaw and comtypes."""


def _get_volume_interface():
    """Prefer pycaw's current AudioDevice.EndpointVolume API, then legacy COM."""
    from ctypes import POINTER, cast
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

    device = AudioUtilities.GetSpeakers()
    if device is None:
        raise RuntimeError("Windows varsayılan ses çıkış aygıtı bulunamadı.")

    # Current pycaw API (documented): GetSpeakers().EndpointVolume.
    try:
        endpoint = getattr(device, "EndpointVolume", None)
        if endpoint is not None:
            return endpoint
    except Exception:
        pass

    # Compatibility with wrappers that retain the endpoint pointer privately.
    try:
        endpoint = getattr(device, "_volume", None)
        if endpoint is not None:
            return endpoint
    except Exception:
        pass

    # Older pycaw versions returned an IMMDevice exposing Activate().
    activate = getattr(device, "Activate", None)
    if callable(activate):
        interface = activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        try:
            return interface.QueryInterface(IAudioEndpointVolume)
        except Exception:
            return cast(interface, POINTER(IAudioEndpointVolume))

    raise RuntimeError(
        "Pycaw ses arayüzü alınamadı: EndpointVolume, _volume ve Activate bulunamadı. "
        f"Aygıt türü: {type(device).__module__}.{type(device).__name__}"
    )


def execute_audio_action(command):
    """Returns (success, diagnostic). Volume values are percentages, 0..100.

    Initialize COM on the calling thread. NOVA invokes actions from a QThread; pycaw's
    endpoint interface is COM-based and must be created/used in an initialized apartment.
    """
    action = command.get("action")
    com_initialized = False
    try:
        from comtypes import CoInitialize, CoUninitialize
        CoInitialize()
        com_initialized = True
        endpoint = _get_volume_interface()

        if action == "set_volume":
            value = command.get("value")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return False, "Ses seviyesi sayısal değil."
            if not 0 <= value <= 100:
                return False, "Ses seviyesi 0-100 aralığında olmalı."
            endpoint.SetMasterVolumeLevelScalar(float(value) / 100.0, None)
            actual = round(float(endpoint.GetMasterVolumeLevelScalar()) * 100)
            if abs(actual - round(value)) > 1:
                return False, f"Ses seviyesi doğrulanamadı; okunan değer %{actual}."
            return True, f"Sistem ses seviyesi %{actual} olarak ayarlandı."

        if action in {"volume_up", "volume_down"}:
            current = float(endpoint.GetMasterVolumeLevelScalar()) * 100
            amount = command.get("amount", 10)
            if isinstance(amount, bool) or not isinstance(amount, (int, float)):
                amount = 10
            amount = max(1, min(25, int(amount)))
            target = min(100, current + amount) if action == "volume_up" else max(0, current - amount)
            endpoint.SetMasterVolumeLevelScalar(target / 100.0, None)
            actual = round(float(endpoint.GetMasterVolumeLevelScalar()) * 100)
            if abs(actual - round(target)) > 1:
                return False, f"Ses seviyesi değiştirilemedi; hedef %{round(target)}, okunan değer %{actual}."
            return True, f"Sistem ses seviyesi %{actual} oldu."

        if action == "mute_audio":
            endpoint.SetMute(1, None)
            muted = bool(endpoint.GetMute())
            if not muted:
                return False, "Sessize alma komutu gönderildi ancak GetMute() sessiz durumunu doğrulamadı."
            return True, "Sistem sesi kapatıldı ve durum doğrulandı."

        if action == "unmute_audio":
            endpoint.SetMute(0, None)
            muted = bool(endpoint.GetMute())
            if muted:
                return False, "Ses açma komutu gönderildi ancak GetMute() hâlâ sessiz durumu gösteriyor."
            return True, "Sistem sesi açıldı ve durum doğrulandı."

        return False, f"Bilinmeyen ses eylemi: {action!r}"
    except ImportError as exc:
        return False, (
            "Ses kontrolü için pycaw ve comtypes gerekli. "
            "Kurulum: python -m pip install pycaw comtypes. "
            f"Eksik modül: {exc}"
        )
    except Exception as exc:
        return False, f"Windows ses kontrolü başarısız: {type(exc).__name__}: {exc}"
    finally:
        if com_initialized:
            try:
                from comtypes import CoUninitialize
                CoUninitialize()
            except Exception:
                pass
