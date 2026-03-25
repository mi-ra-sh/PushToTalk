"""
PushToTalk Text Processing
Auto-replace, custom vocabulary, hallucination detection
"""

import re
import logging

from config import LANG_CONFIGS

logger = logging.getLogger("ptt")

# ─── Hallucination Detection ───

HALLUCINATION_PATTERNS = [
    "дякую за перегляд",
    "підписуйтесь",
    "subscribe",
    "thanks for watching",
    "thank you for watching",
    "like and subscribe",
    "не забудьте підписатися",
    "ставте лайки",
    "дякую за увагу",
    "продовження наступне",
    "субтитри зроблені",
    "subtitles by",
    "translated by",
    "www.",
    "http",
]


def is_prompt_leak(text):
    """Check if text is a prompt leak (Whisper hallucinates prompt phrases on silence)."""
    text_lower = text.lower().strip()

    prompt_markers = [
        "deploy to server",
        "docker container",
        "kubernetes cluster",
        "commit і push",
        "commit and push",
        "make a commit",
        "merge this branch",
        "merge цей branch",
        "check the docker",
        "проаналізуй цей код",
        "відкрий файл",
        "закрий термінал",
        "запусти deploy",
        "перевір docker",
        "push на remote",
        "push to remote",
    ]

    matches = sum(1 for marker in prompt_markers if marker in text_lower)
    if matches >= 2:
        return True

    for lang_cfg in LANG_CONFIGS.values():
        if text_lower in lang_cfg["prompt"].lower():
            return True

    return False


def has_repeated_phrase(text, min_phrase_len=2, max_phrase_len=6, max_repeats=2):
    """Check if text contains a repeated phrase (anti-loop safety net).

    Looks for phrases of min..max words repeated max_repeats+ times consecutively.
    Returns the first found phrase or None.
    """
    words = text.lower().split()
    for phrase_len in range(min_phrase_len, min(max_phrase_len + 1, len(words) // max_repeats + 1)):
        for start in range(len(words) - phrase_len * max_repeats + 1):
            phrase = tuple(words[start : start + phrase_len])
            count = 1
            pos = start + phrase_len
            while pos + phrase_len <= len(words):
                if tuple(words[pos : pos + phrase_len]) == phrase:
                    count += 1
                    pos += phrase_len
                else:
                    break
            if count >= max_repeats:
                return " ".join(phrase)
    return None


def is_hallucination(text):
    """Check if text is a Whisper hallucination."""
    lower = text.lower().strip()

    for pattern in HALLUCINATION_PATTERNS:
        if pattern in lower:
            return True

    words = lower.split()
    if len(words) >= 3:
        for i in range(len(words) - 2):
            if words[i] == words[i + 1] == words[i + 2]:
                return True

    if len(words) >= 2 and len(set(words)) == 1:
        return True

    repeated = has_repeated_phrase(lower)
    if repeated:
        logger.info(f'   [!!] Repeated phrase: "{repeated}"')
        return True

    return False


# ─── Auto-Replace ───

AUTO_REPLACE = {
    # Russian → Ukrainian
    "что": "що",
    "это": "це",
    "привет": "привіт",
    "как": "як",
    "хорошо": "добре",
    "пожалуйста": "будь ласка",
    "спасибо": "дякую",
    "здесь": "тут",
    "сейчас": "зараз",
    "тоже": "також",
    "потому що": "тому що",
    "конечно": "звісно",
    "может": "може",
    "нужно": "потрібно",
    "если": "якщо",
    "когда": "коли",
    "где": "де",
    "почему": "чому",
    "который": "який",
    "только": "тільки",
    "очень": "дуже",
    "сегодня": "сьогодні",
    "завтра": "завтра",
    "вчера": "вчора",
    "всегда": "завжди",
    "никогда": "ніколи",
    "ничего": "нічого",
    "делать": "робити",
    "работать": "працювати",
    "смотреть": "дивитися",
    "знать": "знати",
    "думать": "думати",
    "говорить": "говорити",
    "понимать": "розуміти",
    "хотеть": "хотіти",
    "давай": "давай",
    "ладно": "гаразд",
}


def post_process(text: str, language: str, custom_vocabulary: dict = None) -> str:
    """Apply auto-replace and custom vocabulary to transcribed text."""
    if not text:
        return text

    # Auto-replace for Ukrainian (Russian words → Ukrainian)
    if language == "uk":
        for rus, ukr in AUTO_REPLACE.items():
            pattern = re.compile(r"\b" + re.escape(rus) + r"\b", re.IGNORECASE)
            if pattern.search(text):
                def replace_match(m, _ukr=ukr):
                    matched = m.group(0)
                    if matched[0].isupper():
                        return _ukr[0].upper() + _ukr[1:]
                    return _ukr
                text = pattern.sub(replace_match, text)

    # Custom vocabulary (case-insensitive → correct form)
    if custom_vocabulary:
        for key, replacement in custom_vocabulary.items():
            pattern = re.compile(re.escape(key), re.IGNORECASE)
            text = pattern.sub(replacement, text)

    return text
