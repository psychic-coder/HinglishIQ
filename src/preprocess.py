"""Text normalization for noisy Hinglish and code-mix ratio estimation."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

#: Common romanized-Hindi words (canonical spellings, after slang normalization).
HINDI_LEXICON: frozenset[str] = frozenset(
    """
    aap aapka aapke aapki aaj aaya aayi aaye aayega aayegi aage abhi accha acha
    agar aisa aur bahut bahar baar bata batao batana bekaar bhai bhej bhejo bheja
    bhi bilkul bol bolo chahiye chal chalega chuka dekho dekh de do diya diye din
    dino dinon dobara dono ek fir gaya gayi gaye ghatiya ghante hafte hafta hai
    hain ham hamara ho hoga hogi hona hua hui hue hum hi isliye iska iske itna
    jaldi jab jaise jo jaunga ka kab kaha kahan kaise kal kam kar karna karne
    karo karke karte kat ke ki kisi ko koi kuch kya kyun le lekin liye lo
    mahina mahine maine manga mat me mein mera mere meri mil mila mili milega
    mujhe na nahi naya ne nikla nikal pahuncha pe pehle par paisa paise pata
    phati raha rahi rahe sab sahi sakta sakte se tak tha thi the theek thik thoda
    toota tuta tum turant unka usne wala wali wale wapas wo woh ya yaar ye yeh
    abtak bekar barbaad dikha dikh dikhta hi hu hoon jayega gya kaafi kitna
    kyunki mast mazaak raho rakho samajh sasta shaadi waqt
    """.split()
)

#: Spelling / slang variants mapped to a canonical form.
SLANG_MAP: dict[str, str] = {
    "nhi": "nahi", "nai": "nahi", "nahin": "nahi", "nahee": "nahi", "ni": "nahi",
    "kr": "kar", "krna": "karna", "kro": "karo", "krke": "karke", "krdo": "kar do",
    "h": "hai", "hy": "hai", "he": "hai",
    "plz": "please", "pls": "please", "plss": "please", "plzz": "please",
    "mjhe": "mujhe", "muje": "mujhe", "mujhko": "mujhe",
    "kb": "kab", "ky": "kya", "kyu": "kyun", "kyon": "kyun",
    "abi": "abhi", "jldi": "jaldi", "jaldii": "jaldi",
    "thx": "thanks", "thnx": "thanks", "tq": "thanks", "ty": "thanks",
    "bro": "bhai", "bhaiya": "bhai",
    "pese": "paise", "pesa": "paisa", "rply": "reply", "msg": "message",
    "yr": "yaar", "yrr": "yaar", "ok": "okay", "k": "okay",
    "u": "you", "ur": "your", "r": "are",
    "mei": "me", "tk": "tak", "gya": "gaya", "gyi": "gayi",
}

#: Emoji -> textual cue (kept because they carry sentiment / urgency signal).
EMOJI_MAP: dict[str, str] = {
    "😡": " [angry] ", "🤬": " [angry] ", "😤": " [angry] ", "😠": " [angry] ",
    "😭": " [sad] ", "😢": " [sad] ", "😞": " [sad] ", "😔": " [sad] ",
    "🙏": " [please] ",
    "👍": " [positive] ", "😊": " [positive] ", "😀": " [positive] ", "❤️": " [positive] ",
    "❤": " [positive] ", "🙂": " [positive] ", "😍": " [positive] ",
    "⚠️": " [urgent] ", "‼️": " [urgent] ", "🆘": " [urgent] ",
}

_ID_RE = re.compile(r"\b(?:od|ord)\s?-?\d{5,12}\b", re.IGNORECASE)
_ELONGATION_RE = re.compile(r"([a-z])\1{2,}")
_PUNCT_REPEAT_RE = re.compile(r"([!?.])\1{2,}")
_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0000FE0F\U0000200D]+",
)
_TOKEN_RE = re.compile(r"(?:od|ord)\d{5,12}|\[[a-z]+\]|₹?\d[\d,]*|[a-z]+(?:'[a-z]+)?")


@dataclass(frozen=True)
class PreprocessResult:
    """Normalized text plus code-mix statistics."""

    original: str
    text: str
    tokens: tuple[str, ...]
    code_mix_ratio: float


def _replace_emojis(text: str) -> str:
    for emoji, cue in EMOJI_MAP.items():
        text = text.replace(emoji, cue)
    return _EMOJI_RE.sub(" ", text)


def _normalize_slang(text: str) -> str:
    def swap(match: re.Match[str]) -> str:
        word = match.group(0)
        return SLANG_MAP.get(word, word)

    return re.sub(r"(?<![\w\[])[a-z]+(?![\w\]])", swap, text)


def _restore_ids(text: str) -> str:
    return _ID_RE.sub(lambda m: re.sub(r"[\s\-]", "", m.group(0)).upper(), text)


def code_mix_ratio(tokens: tuple[str, ...] | list[str]) -> float:
    """Fraction of alphabetic word tokens found in the romanized-Hindi lexicon."""
    words = [t for t in tokens if t.isalpha()]
    if not words:
        return 0.0
    return round(sum(w in HINDI_LEXICON for w in words) / len(words), 4)


def preprocess(text: str) -> PreprocessResult:
    """Normalize a raw Hinglish message.

    Steps: Unicode NFKC, lowercase, emoji -> cue tokens, collapse letter
    elongations ("pleaseeee" -> "pleasee"), collapse repeated punctuation,
    canonicalize slang spellings, restore order IDs to upper case, and squeeze
    whitespace. Numbers, amounts and IDs are never altered.
    """
    normalized = unicodedata.normalize("NFKC", text)
    normalized = _replace_emojis(normalized).lower()
    normalized = _ELONGATION_RE.sub(r"\1\1", normalized)
    normalized = _PUNCT_REPEAT_RE.sub(r"\1\1", normalized)
    normalized = _normalize_slang(normalized)
    normalized = _restore_ids(normalized)
    normalized = " ".join(normalized.split())
    tokens = tuple(_TOKEN_RE.findall(normalized.lower()))
    return PreprocessResult(
        original=text,
        text=normalized,
        tokens=tokens,
        code_mix_ratio=code_mix_ratio(tokens),
    )
