"""Соответствие кодов Whisper и FLORES-200 (NLLB)."""

_WHISPER_TO_FLORES = {
    "uz": "uzn_Latn",   # узбекский (латиница)
    "az": "azj_Latn",   # азербайджанский (северный, латиница)
    "tg": "tgk_Cyrl",   # таджикский (кириллица)
}


def flores_code(whisper_lang):
    """Код FLORES-200 для языка Whisper или None, если языка нет в NLLB."""
    if whisper_lang is None:
        return None
    return _WHISPER_TO_FLORES.get(whisper_lang)
