"""NOVA için ortak Türkçe TTS metin hazırlama katmanı.

Bu modül ekrandaki yanıtı değiştirmez; yalnızca ses motoruna giden metni
konuşmaya daha uygun hâle getirir.
"""
from __future__ import annotations

import re

_ONES = ("sıfır", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz")
_TENS = ("", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan")
_SCALES = ((1_000_000_000_000, "trilyon"), (1_000_000_000, "milyar"), (1_000_000, "milyon"), (1_000, "bin"))
_MONTHS = {"01": "ocak", "02": "şubat", "03": "mart", "04": "nisan", "05": "mayıs", "06": "haziran", "07": "temmuz", "08": "ağustos", "09": "eylül", "10": "ekim", "11": "kasım", "12": "aralık"}


def _under_thousand(n: int) -> str:
    if n < 10:
        return _ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return _TENS[tens] + (" " + _ONES[ones] if ones else "")
    hundreds, remainder = divmod(n, 100)
    result = "yüz" if hundreds == 1 else _ONES[hundreds] + " yüz"
    return result + (" " + _under_thousand(remainder) if remainder else "")


def number_to_turkish(value: int) -> str:
    """Convert an integer to Turkish words, up to 999,999,999,999,999."""
    n = int(value)
    if n < 0:
        return "eksi " + number_to_turkish(-n)
    if n < 1000:
        return _under_thousand(n)
    if n >= 1_000_000_000_000_000:
        return str(n)
    parts = []
    for scale, label in _SCALES:
        amount, n = divmod(n, scale)
        if amount:
            if label == "bin" and amount == 1:
                parts.append("bin")
            else:
                parts.append(number_to_turkish(amount) + " " + label)
    if n:
        parts.append(_under_thousand(n))
    return " ".join(parts)


def _number_to_words(raw: str) -> str:
    # Binlik ayırıcı nokta/boşluklarını kaldır; ondalık virgül/nokta sesli okunur.
    value = raw.replace(" ", "")
    sign = ""
    if value.startswith("-"):
        sign, value = "eksi ", value[1:]
    elif value.startswith("+"):
        sign, value = "artı ", value[1:]
    # Türkçe fiyat biçimi: 1.250,50 -> 1250,50.
    if "." in value and "," in value and value.rfind(",") > value.rfind("."):
        value = value.replace(".", "")
    elif re.fullmatch(r"\d{1,3}(?:\.\d{3})+", value):
        value = value.replace(".", "")
    elif re.fullmatch(r"\d{1,3}(?:,\d{3})+", value):
        value = value.replace(",", "")
    if "." in value or "," in value:
        separator = "," if "," in value else "."
        left, right = value.split(separator, 1)
        if right.isdigit() and left.lstrip("+-").isdigit():
            whole = number_to_turkish(int(left))
            if not right.strip("0"):
                return sign + whole
            # Tek basamaklı/başı sıfırlı küsuratları basamak basamak; diğerlerini sayı olarak oku.
            if len(right) == 1 or right.startswith("0"):
                decimal = " ".join(_ONES[int(ch)] for ch in right)
            else:
                decimal = number_to_turkish(int(right))
            return sign + f"{whole} virgül {decimal}"
    if value.isdigit():
        return sign + number_to_turkish(int(value))
    return raw


def _strip_markup(text: str) -> str:
    # Kod, arama motoru kaynak işaretleri ve ham bağlantılar konuşmada anlamsız gürültü üretmesin.
    text = re.sub(r"[^]*", " ", text)
    text = re.sub(r"【[^】]*】", " ", text)
    text = re.sub(r"\[\d+\]", " ", text)
    text = re.sub(r"```[\s\S]*?```", " kod bloğu ", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"!?\[([^\]]*)\]\((https?://[^)]+)\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", text)
    text = re.sub(r"https?://\S+|www\.\S+", " bağlantı adresi ", text, flags=re.I)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*>\s?", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+[.)]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
    text = re.sub(r"(?<!\w)_{1,2}(.*?)_{1,2}(?!\w)", r"\1", text)
    text = re.sub(r"^\s*[-*_]{3,}\s*$", " ", text, flags=re.MULTILINE)
    # Emoji ve dekoratif karakterler.
    text = re.sub(r"[\U0001F000-\U0001FAFF\U0001FC00-\U0001FFFF\U00002600-\U000026FF\U00002700-\U000027BF\u200d\ufe0e\ufe0f\u20e3]", "", text)
    text = re.sub(r"[✓✔✕✖⚠❌⭕☑☒★☆♥♡♦♣♠•·→←↑↓]+", "", text)
    return text


def _expand_units(text: str) -> str:
    # Birimleri sayılardan önce aç ki sayıların nasıl okunacağı belli olsun.
    replacements = [
        (r"(?i)\bwi[\s-]?fi\b", "vayfay"),
        (r"(?i)\bCPU\b", "işlemci"),
        (r"(?i)\bGPU\b", "ekran kartı"),
        (r"(?i)\bRAM\b", "bellek"),
        (r"(?i)\bVRAM\b", "ekran belleği"),
        (r"(?i)\bSSD\b", "katı hâl diski"),
        (r"(?i)\bHDD\b", "sabit disk"),
        (r"(?i)\bURL\b", "bağlantı adresi"),
        (r"(?i)\bAPI\b", "uygulama arayüzü"),
        (r"(?i)\bAI\b", "yapay zekâ"),
        (r"(?i)\bGB/s\b", "saniyede gigabayt"),
        (r"(?i)\bMB/s\b", "saniyede megabayt"),
        (r"(?i)\bGbps\b", "gigabit bölü saniye"),
        (r"(?i)\bMbps\b", "megabit bölü saniye"),
        (r"(?i)\bKbps\b", "kilobit bölü saniye"),
        (r"(?i)\bGB\b", "gigabayt"),
        (r"(?i)\bMB\b", "megabayt"),
        (r"(?i)\bKB\b", "kilobayt"),
        (r"(?i)\bTB\b", "terabayt"),
        (r"(?i)\bGHz\b", "gigahertz"),
        (r"(?i)\bMHz\b", "megahertz"),
        (r"(?i)\bkHz\b", "kilohertz"),
        (r"(?i)\bHz\b", "hertz"),
        (r"(?i)\bms\b", "milisaniye"),
        (r"(?i)\bkm/h\b", "kilometre bölü saat"),
        (r"(?i)\bkm\b", "kilometre"),
        (r"(?i)\bcm\b", "santimetre"),
        (r"(?i)\bmm\b", "milimetre"),
        (r"(?i)\bkg\b", "kilogram"),
        (r"(?i)\bmg\b", "miligram"),
        (r"(?i)\bml\b", "mililitre"),
        (r"(?i)\bMHz\b", "megahertz"),
        (r"(?i)°\s*C\b", " derece"),
        (r"(?i)\bCelsius\b", "santigrat"),
        (r"(?i)\bTL\b", "Türk lirası"),
        (r"(?i)\bUSD\b", "Amerikan doları"),
        (r"(?i)\bEUR\b", "avro"),
    ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)
    # Yüzdeleri tek parça ele al: hem 23% hem %23 -> yüzde yirmi üç.
    text = re.sub(
        r"(?<![\w/])%\s*(\d+(?:[.,]\d+)?)|(?<![\w/])(\d+(?:[.,]\d+)?)\s*%",
        lambda m: "yüzde " + _number_to_words(m.group(1) or m.group(2)),
        text,
    )
    # Tarih ve saatleri sayıların genel dönüşümünden önce daha doğal biçime getir.
    def date_repl(m):
        year, month, day = m.group(1), m.group(2), m.group(3)
        month_name = _MONTHS.get(month, month)
        return f"{day} {month_name} {year}"
    text = re.sub(r"\b(20\d{2})-(\d{2})-(\d{2})\b", date_repl, text)
    text = re.sub(r"\b(\d{1,2}):(\d{2})\b", lambda m: f"{m.group(1)} {m.group(2)}", text)
    # Sürüm numaraları ve sayısal aralıklar, genel sayı dönüşümünden önce korunur.
    text = re.sub(
        r"(?<![\w])v?(\d+)\.(\d+)\.(\d+)(?![\w])",
        lambda m: ("sürüm " if m.group(0).lower().startswith("v") else "") + " nokta ".join(number_to_turkish(int(part)) for part in m.groups()),
        text,
        flags=re.I,
    )
    text = re.sub(
        r"(?<![\w])(-?\d+)\s*[-–]\s*(-?\d+)(?![\w])",
        lambda m: number_to_turkish(int(m.group(1))) + " ile " + number_to_turkish(int(m.group(2))),
        text,
    )
    # Noktalama öncesinde duran ondalıklı/büyük/küçük sayılar.
    text = re.sub(r"(?<![\w/])[-+]?\d[\d.,]*\d|(?<![\w/])[-+]?\d", lambda m: _number_to_words(m.group(0).strip()), text)
    # Sembollerin kalan yaygın okunuşları.
    text = text.replace("&", " ve ").replace("+", " artı ").replace("=", " eşittir ")
    text = re.sub(r"\s*/\s*", " bölü ", text)
    return text


def prepare_speech_text(text: object) -> str:
    """Clean and normalize arbitrary assistant output for Turkish speech."""
    result = _strip_markup(str(text or "").strip())
    result = _expand_units(result)
    result = re.sub(r"[ \t]{2,}", " ", result)
    result = re.sub(r" *\n[ \t]*", ". ", result)
    result = re.sub(r"\n{2,}", ". ", result)
    result = re.sub(r"\s+([,.;:!?])", r"\1", result)
    result = re.sub(r"([,;:])(?=\S)", r"\1 ", result)
    result = re.sub(r"([.!?]){2,}", r"\1", result)
    result = re.sub(r"\s{2,}", " ", result)
    result = re.sub(r"([,;:])\s*([.!?])", r"\2", result)
    return result.strip(" \t\n,;:")
