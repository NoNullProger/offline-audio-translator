from pathlib import Path

import pytest

from app.translate import NllbTranslator


def test_unmapped_language_returns_nones(tmp_path):
    t = NllbTranslator(tmp_path)   # модель даже не нужна: маппинг проверяется до загрузки
    assert t.translate_texts(["Текст"], "ru") == [None]
    assert t.translate_texts(["Текст"], None) == [None]
    assert t.translate_texts([], "uz") == []


MODELS_DIR = Path(__file__).resolve().parent.parent / "models-dev" / "nllb-600m-ct2"


@pytest.mark.skipif(not MODELS_DIR.exists(), reason="модель NLLB не скачана")
def test_smoke_uzbek_to_russian():
    t = NllbTranslator(MODELS_DIR)
    out = t.translate_texts(["Assalomu alaykum"], "uz")
    assert out and isinstance(out[0], str) and len(out[0]) > 0
