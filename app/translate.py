"""Перевод на русский: NLLB-200 (CTranslate2) + sentencepiece-токенизатор."""

import itertools
from pathlib import Path

import ctranslate2
import sentencepiece as spm

from .langmap import flores_code

TARGET_CODE = "rus_Cyrl"
_CHUNK = 32  # текстов за один вызов translate_batch


def _chunks(seq, size):
    it = iter(seq)
    while batch := list(itertools.islice(it, size)):
        yield batch


class NllbTranslator:
    def __init__(self, model_dir: Path):
        self._dir = Path(model_dir)
        self._translator: ctranslate2.Translator | None = None
        self._spm: spm.SentencePieceProcessor | None = None

    def _ensure_loaded(self) -> None:
        if self._translator is None:
            if not self._dir.exists():
                raise RuntimeError(f"модель перевода не найдена: {self._dir}")
            self._translator = ctranslate2.Translator(
                str(self._dir), device="cpu", compute_type="int8"
            )
            self._spm = spm.SentencePieceProcessor(
                model_file=str(self._dir / "sentencepiece.bpe.model")
            )

    def translate_texts(self, texts, src_whisper_lang):
        """Переводит список текстов в русский. None — перевод недоступен."""
        if not texts:
            return []
        if flores_code(src_whisper_lang) is None:
            return [None] * len(texts)
        self._ensure_loaded()
        result: list[str | None] = []
        for chunk in _chunks(texts, _CHUNK):
            try:
                result.extend(self._translate_chunk(chunk, src_whisper_lang))
            except Exception:  # чанк не должен валить всю задачу
                result.extend([None] * len(chunk))
        return result

    def _translate_chunk(self, chunk, src_whisper_lang):
        src_code = flores_code(src_whisper_lang)
        batches = [
            [src_code] + self._spm.encode(text, out_type=str)
            for text in chunk
        ]
        results = self._translator.translate_batch(
            batches,
            target_prefix=[[TARGET_CODE]],
            beam_size=4,
            max_decoding_length=256,
            max_batch_size=_CHUNK,
        )
        out = []
        for r in results:
            hyp = r.hypotheses[0]
            tokens = [t for t in hyp if t not in (TARGET_CODE, "</s>")]
            out.append(self._spm.decode(tokens))
        return out
