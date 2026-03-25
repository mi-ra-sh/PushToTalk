"""
PushToTalk Text Processing
Auto-replace dictionary, custom vocabulary, post-processing
"""

import re
import logging

logger = logging.getLogger("ptt")

# Auto-replace: Russian→Ukrainian, tech terms, common errors
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
            # Case-insensitive word boundary replace
            pattern = re.compile(r"\b" + re.escape(rus) + r"\b", re.IGNORECASE)
            if pattern.search(text):
                # Preserve case of first letter
                def replace_match(m):
                    matched = m.group(0)
                    if matched[0].isupper():
                        return ukr[0].upper() + ukr[1:]
                    return ukr
                text = pattern.sub(replace_match, text)

    # Custom vocabulary (case-insensitive → correct form)
    if custom_vocabulary:
        for key, replacement in custom_vocabulary.items():
            pattern = re.compile(re.escape(key), re.IGNORECASE)
            text = pattern.sub(replacement, text)

    return text
