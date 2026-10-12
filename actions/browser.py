import subprocess
import webbrowser


def execute_browser_action(command):
    action = command.get("action")

    if action == "open_url":
        url = command.get("url")

        # Şimdilik açık URL listesini bilinçli olarak sınırlıyoruz.
        allowed_urls = {
            "https://www.youtube.com",
        }

        if url not in allowed_urls:
            return False, None

        try:
            opened = webbrowser.open(url)
        except Exception as exc:
            return False, f"YouTube açılamadı: {type(exc).__name__}: {exc}"
        if opened:
            return True, "YouTube açma isteği gönderildi."
        return False, "Tarayıcı YouTube adresini açamadı."

    if action == "open_incognito":
        app = command.get("app")

        if app == "chrome":
            subprocess.Popen([
                "cmd", "/c", "start", "", "chrome", "--incognito"
            ])
            return True, "✓ Chrome gizli pencere açıldı."

    return False, None
