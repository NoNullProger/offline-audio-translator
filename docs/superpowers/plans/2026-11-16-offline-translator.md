# Офлайн-расшифровка аудио (uz/az/tg) → русский: план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Портативный офлайн-сервис: расшифровка аудио (узбекский/таджикский/азербайджанский) через faster-whisper + перевод на русский через NLLB-200, локальный веб-интерфейс, развёртывание с флешки на Astra Linux без интернета.

**Architecture:** Один Python-сервис (FastAPI + uvicorn) с фоновым последовательным обработчиком задач (threading). Модели — faster-whisper (CTranslate2, int8) и NLLB-600M (CTranslate2, int8), загружаются лениво из локальных каталогов. Развёртывание: бандл с портативным CPython 3.11 + site-packages + модели, запуск `bash start.sh`.

**Tech Stack:** Python 3.11, faster-whisper, CTranslate2, sentencepiece, FastAPI, uvicorn, python-docx, PyAV (декодирование аудио), pytest.

**Spec:** `docs/superpowers/specs/2026-11-16-astra-offline-translator-design.md`

## Global Constraints

- Рантайм без интернета: `HF_HUB_OFFLINE=1` в start.sh, модели только из локальных каталогов.
- Память ≤ 8 ГБ: задачи строго последовательно, модели int8, `cpu_threads` ≤ 4.
- Код совместим с Python 3.11 (без `itertools.batched` и прочего 3.12+).
- Никаких сборщиков JS; веб-интерфейс — статичные HTML/JS/CSS из каталога `web/`.
- Запись на диск — только под корнем бандла (`work/`, `output/`, `logs/`); исключение — fallback `/tmp` в start.sh при noexec.
- Модели Whisper: локальные каталоги `whisper-large-v3-turbo`, `whisper-medium`, `whisper-small` (источники `Systran/faster-whisper-*`); NLLB: `models/nllb-600m-ct2` (конвертация из `facebook/nllb-200-distilled-600M`).
- Коды FLORES: `uz→uzn_Latn`, `az→azj_Latn`, `tg→tgk_Cyrl`, выход `rus_Cyrl`.
- Порты/пути: порт `${PORT:-8080}`, хост `${HOST:-127.0.0.1}`; структура бандла — как в спеке (§6).
- Ограничение загрузки файла: 2 ГБ.

## Структура файлов проекта

```
app/
  __init__.py
  config.py       # пути: root, models, work, output, web, logs (env-переопределения)
  langmap.py      # коды Whisper → FLORES-200
  asr.py          # Transcriber (faster-whisper), validate_audio (PyAV)
  translate.py    # NllbTranslator (CTranslate2 + sentencepiece)
  outputs.py      # запись transcript.txt / translation.txt / result.docx
  jobs.py         # Segment, Job, JobManager (очередь + воркер)
  main.py         # FastAPI: create_app(manager), маршруты, статика
web/
  index.html, app.js, style.css
scripts/
  prepare-models.py      # скачивание whisper-моделей + конвертация NLLB в CT2
  fetch-fleurs.py        # тестовые клипы FLEURS (uz/az/tg) с эталонами
  eval-wer.py            # WER моделей на клипах FLEURS
  build-offline-bundle.sh
tests/
  test_langmap.py, test_outputs.py, test_jobs.py, test_api.py, test_config.py
requirements.txt, requirements-dev.txt, requirements-convert.txt
start.sh               # шаблон лаунчера (копируется в бандл)
.gitignore, README.md
```

Каталоги дев-артефактов (в `.gitignore`): `.venv/`, `models-dev/`, `data/`, `translator-bundle*/`.

---

### Task 0: Dev-окружение и каркас проекта

**Files:**
- Create: `requirements.txt`, `requirements-dev.txt`, `requirements-convert.txt`, `.gitignore`, `app/__init__.py`, `web/` (пустой), `scripts/`, `tests/`

**Interfaces:**
- Produces: venv `.venv` (Python 3.11), устанавливаемые пакеты для всех последующих задач.

- [ ] **Step 1: Установить uv и создать venv 3.11**

```bash
pip3 install --user uv
python3 -m uv venv .venv --python 3.11
```

- [ ] **Step 2: Написать requirements-файлы**

`requirements.txt` (рантайм, пойдёт в бандл):
```
faster-whisper>=1.1.0,<2
ctranslate2>=4.4.0,<5
fastapi>=0.115,<1
uvicorn>=0.30,<1
python-multipart>=0.0.9,<0.1
python-docx>=1.1,<2
sentencepiece>=0.2,<0.3
```

`requirements-dev.txt` (только разработка/сборка, в бандл не попадает):
```
-r requirements.txt
pytest>=8
httpx>=0.27
huggingface_hub>=0.26
pyarrow>=17
```

`requirements-convert.txt` (только для конвертации NLLB, в бандл не попадает):
```
torch
transformers>=4.40,<5
```

- [ ] **Step 3: Установить зависимости**

```bash
python3 -m uv pip install -r requirements-dev.txt -p .venv
```

- [ ] **Step 4: Написать .gitignore**

```
.venv/
__pycache__/
*.pyc
models-dev/
data/
translator-bundle*/
.pytest_cache/
```

- [ ] **Step 5: Создать пустые пакеты/каталоги и коммит**

```bash
mkdir -p app web scripts tests
printf '' > app/__init__.py
git add requirements*.txt .gitignore app/__init__.py
git commit -m "chore: каркас проекта и зависимости"
```

---

### Task 1: config.py и langmap.py

**Files:**
- Create: `app/config.py`, `app/langmap.py`
- Test: `tests/test_config.py`, `tests/test_langmap.py`

**Interfaces:**
- Produces:
  - `config.root() -> Path` — корень бандла/репо (env `TRANSLATOR_ROOT`, иначе репо).
  - `config.models_dir() -> Path` (env `TRANSLATOR_MODELS_DIR`, иначе `<root>/models`, в деве без env — `models-dev/`).
  - `config.work_dir() -> Path`, `config.output_dir() -> Path`, `config.logs_dir() -> Path` (под `root()/data` в деве, под корнем бандла в проде — см. код).
  - `config.web_dir() -> Path`.
  - `langmap.flores_code(whisper_lang: str | None) -> str | None` — `'uz'→'uzn_Latn'`, `'az'→'azj_Latn'`, `'tg'→'tgk_Cyrl'`, прочее/None → `None`.

- [ ] **Step 1: Написать тесты**

`tests/test_langmap.py`:
```python
from app.langmap import flores_code

def test_known_codes():
    assert flores_code("uz") == "uzn_Latn"
    assert flores_code("az") == "azj_Latn"
    assert flores_code("tg") == "tgk_Cyrl"

def test_unknown_and_none():
    assert flores_code("ru") is None
    assert flores_code(None) is None
```

`tests/test_config.py`:
```python
from pathlib import Path
import importlib, pytest
import app.config as cfg

def test_models_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("TRANSLATOR_MODELS_DIR", str(tmp_path / "m"))
    importlib.reload(cfg)
    assert cfg.models_dir() == tmp_path / "m"

def test_bundle_root_layout(monkeypatch, tmp_path):
    monkeypatch.delenv("TRANSLATOR_MODELS_DIR", raising=False)
    monkeypatch.setenv("TRANSLATOR_ROOT", str(tmp_path))
    importlib.reload(cfg)
    assert cfg.root() == tmp_path
    assert cfg.models_dir() == tmp_path / "models"
    assert cfg.work_dir() == tmp_path / "work"
    assert cfg.output_dir() == tmp_path / "output"
    assert cfg.web_dir() == tmp_path / "web"

def test_dev_defaults(monkeypatch):
    monkeypatch.delenv("TRANSLATOR_ROOT", raising=False)
    monkeypatch.delenv("TRANSLATOR_MODELS_DIR", raising=False)
    importlib.reload(cfg)
    assert cfg.models_dir().name == "models-dev"
    importlib.reload(cfg)  # вернуть состояние другим тестам
```

- [ ] **Step 2: Запустить тесты, убедиться в падении**

Run: `.venv/bin/python -m pytest tests/test_config.py tests/test_langmap.py -v`
Expected: FAIL (`ModuleNotFoundError: app.config` / `app.langmap`)

- [ ] **Step 3: Реализовать app/langmap.py**

```python
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
```

- [ ] **Step 4: Реализовать app/config.py**

```python
"""Пути приложения. В бандле всё живёт под TRANSLATOR_ROOT, в деве — под репо."""

import os
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent

def root() -> Path:
    env = os.environ.get("TRANSLATOR_ROOT")
    return Path(env) if env else _REPO

def models_dir() -> Path:
    env = os.environ.get("TRANSLATOR_MODELS_DIR")
    if env:
        return Path(env)
    if os.environ.get("TRANSLATOR_ROOT"):
        return root() / "models"
    return _REPO / "models-dev"   # дев-режим: модели качаются в models-dev/

def work_dir() -> Path:
    return _data_or_root() / "work"

def output_dir() -> Path:
    return _data_or_root() / "output"

def logs_dir() -> Path:
    return _data_or_root() / "logs"

def web_dir() -> Path:
    return root() / "web"

def _data_or_root() -> Path:
    # в деве рабочие каталоги прячем в data/, в бандле — прямо под корнем
    return root() / "data" if not os.environ.get("TRANSLATOR_ROOT") else root()
```

- [ ] **Step 5: Запустить тесты — все зелёные**

Run: `.venv/bin/python -m pytest tests/test_config.py tests/test_langmap.py -v`
Expected: PASS

- [ ] **Step 6: Коммит**

```bash
git add app/config.py app/langmap.py tests/test_config.py tests/test_langmap.py
git commit -m "feat: конфигурация путей и маппинг кодов языков"
```

---

### Task 2: outputs.py — запись результатов (txt, docx)

**Files:**
- Create: `app/outputs.py`
- Test: `tests/test_outputs.py`

**Interfaces:**
- Consumes: сегменты — объекты с атрибутами `.start` (float), `.end` (float), `.text` (str); канонический тип `asr.Segment` определяется в Task 3.
- Produces:
  - `outputs.fmt_time(sec: float) -> str` — `"MM:SS.ss"` (например `"01:07.42"`).
  - `outputs.write_transcript(path: Path, segments: list) -> None` — строки `[MM:SS.ss] текст`.
  - `outputs.write_translation(path: Path, segments: list, translations: list[str | None]) -> None` — для `None` строка `[MM:SS.ss] [перевод недоступен]`.
  - `outputs.write_docx(path: Path, title: str, segments: list, translations: list[str | None]) -> None`.

- [ ] **Step 1: Написать тесты**

`tests/test_outputs.py`:
```python
from docx import Document
from app.outputs import fmt_time, write_transcript, write_translation, write_docx

class Seg:
    def __init__(self, start, text):
        self.start, self.end, self.text = start, start + 2.0, text

def test_fmt_time():
    assert fmt_time(67.42) == "01:07.42"
    assert fmt_time(0) == "00:00.00"

def test_transcript_txt(tmp_path):
    p = tmp_path / "t.txt"
    write_transcript(p, [Seg(0.0, "Салом"), Seg(5.0, "Хуш")])
    assert p.read_text(encoding="utf-8") == "[00:00.00] Салом\n[00:05.00] Хуш\n"

def test_translation_missing_marker(tmp_path):
    p = tmp_path / "r.txt"
    write_translation(p, [Seg(0.0, "Салом")], [None])
    assert "[перевод недоступен]" in p.read_text(encoding="utf-8")

def test_docx(tmp_path):
    p = tmp_path / "r.docx"
    write_docx(p, "запись.wav", [Seg(0.0, "Салом")], ["Привет"])
    d = Document(str(p))
    text = "\n".join(par.text for par in d.paragraphs)
    assert "запись.wav" in text and "Привет" in text and "[00:00.00] Салом" in text
```

- [ ] **Step 2: Запустить, убедиться в падении**

Run: `.venv/bin/python -m pytest tests/test_outputs.py -v`
Expected: FAIL (`No module named app.outputs`)

- [ ] **Step 3: Реализовать app/outputs.py**

```python
"""Запись результатов: расшифровка, перевод, docx."""

from pathlib import Path

from docx import Document

def fmt_time(sec: float) -> str:
    m = int(sec // 60)
    s = sec - 60 * m
    return f"{m:02d}:{s:05.2f}"

def _lines(segments, translations=None):
    lines = []
    for i, seg in enumerate(segments):
        text = seg.text
        if translations is not None:
            t = translations[i] if i < len(translations) else None
            text = t if t else "[перевод недоступен]"
        lines.append(f"[{fmt_time(seg.start)}] {text}")
    return lines

def write_transcript(path: Path, segments) -> None:
    path.write_text("\n".join(_lines(segments)) + "\n", encoding="utf-8")

def write_translation(path: Path, segments, translations) -> None:
    path.write_text("\n".join(_lines(segments, translations)) + "\n", encoding="utf-8")

def write_docx(path: Path, title: str, segments, translations) -> None:
    doc = Document()
    doc.add_heading(f"Расшифровка и перевод: {title}", 0)
    doc.add_heading("Расшифровка (оригинал)", level=1)
    for line in _lines(segments):
        doc.add_paragraph(line)
    doc.add_heading("Перевод на русский", level=1)
    for line in _lines(segments, translations):
        doc.add_paragraph(line)
    doc.save(str(path))
```

- [ ] **Step 4: Запустить тесты — зелёные**

Run: `.venv/bin/python -m pytest tests/test_outputs.py -v`
Expected: PASS

- [ ] **Step 5: Коммит**

```bash
git add app/outputs.py tests/test_outputs.py
git commit -m "feat: запись расшифровки, перевода и docx"
```

---

### Task 3: asr.py — обёртка faster-whisper

**Files:**
- Create: `app/asr.py`
- Test: `tests/test_asr.py`

**Interfaces:**
- Consumes: `config.models_dir()`.
- Produces:
  - `asr.Segment(start: float, end: float, text: str)` — dataclass, используется всеми слоями.
  - `asr.validate_audio(path: Path) -> None` — `ValueError` при невозможности открыть/нет аудиодорожки.
  - `asr.Transcriber(models_root: Path)` с методом
    `transcribe(audio_path: Path, language: str | None, model_size: str, on_progress=None) -> tuple[list[Segment], str | None]`
    (`language` — `'uz'|'az'|'tg'|'auto'|None`; возвращает сегменты и определённый Whisper код языка при auto; `on_progress(p: float)` вызывается по сегментам).
  - Константа `asr.MODEL_DIRS = {"turbo": "whisper-large-v3-turbo", "medium": "whisper-medium", "small": "whisper-small"}`.

- [ ] **Step 1: Написать тесты**

`tests/test_asr.py`:
```python
import struct, wave, pytest
from pathlib import Path
from app.asr import Segment, Transcriber, validate_audio, MODEL_DIRS

def make_wav(path: Path, seconds: float = 0.5):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        frames = b"".join(struct.pack("<h", 0) for _ in range(int(16000 * seconds)))
        w.writeframes(frames)

def test_model_dirs():
    assert MODEL_DIRS["turbo"] == "whisper-large-v3-turbo"

def test_validate_audio_bad_file(tmp_path):
    p = tmp_path / "bad.wav"; p.write_bytes(b"not audio at all")
    with pytest.raises(ValueError):
        validate_audio(p)

def test_validate_audio_ok(tmp_path):
    p = tmp_path / "ok.wav"; make_wav(p)
    validate_audio(p)  # не бросает

def test_unknown_model_size(tmp_path):
    t = Transcriber(tmp_path)
    with pytest.raises(ValueError):
        t._model("huge")

def test_missing_model_dir(tmp_path):
    t = Transcriber(tmp_path)
    with pytest.raises(RuntimeError, match="модель не найдена"):
        t._model("small")
```

- [ ] **Step 2: Запустить, убедиться в падении**

Run: `.venv/bin/python -m pytest tests/test_asr.py -v`
Expected: FAIL (`No module named app.asr`)

- [ ] **Step 3: Реализовать app/asr.py**

```python
"""Расшифровка речи: faster-whisper на CPU, int8, VAD."""

import os
from dataclasses import dataclass
from pathlib import Path

import av
from faster_whisper import WhisperModel

MODEL_DIRS = {
    "turbo": "whisper-large-v3-turbo",
    "medium": "whisper-medium",
    "small": "whisper-small",
}

@dataclass
class Segment:
    start: float
    end: float
    text: str

def validate_audio(path: Path) -> None:
    """Бросает ValueError, если файл не аудио или без аудиодорожки."""
    try:
        with av.open(str(path)) as container:
            if not container.streams.audio:
                raise ValueError("в файле нет аудиодорожки")
    except av.FFmpegError as e:
        raise ValueError(f"повреждённый или неподдерживаемый файл: {e}") from e

class Transcriber:
    def __init__(self, models_root: Path):
        self._root = Path(models_root)
        self._cache: dict[str, WhisperModel] = {}

    def _model(self, model_size: str) -> WhisperModel:
        if model_size not in MODEL_DIRS:
            raise ValueError(f"неизвестный размер модели: {model_size}")
        if model_size not in self._cache:
            path = self._root / MODEL_DIRS[model_size]
            if not path.exists():
                raise RuntimeError(f"модель не найдена: {path}")
            self._cache[model_size] = WhisperModel(
                str(path),
                device="cpu",
                compute_type="int8",
                cpu_threads=min(4, os.cpu_count() or 4),
            )
        return self._cache[model_size]

    def transcribe(self, audio_path, language, model_size, on_progress=None):
        """Возвращает (сегменты, определённый язык|None). language: uz/az/tg/auto/None."""
        validate_audio(Path(audio_path))
        model = self._model(model_size)
        lang = None if language in (None, "auto") else language
        segments_iter, info = model.transcribe(
            str(audio_path),
            language=lang,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        segments: list[Segment] = []
        duration = info.duration or 0.0
        for s in segments_iter:
            segments.append(Segment(start=s.start, end=s.end, text=(s.text or "").strip()))
            if on_progress and duration > 0:
                on_progress(min(1.0, s.end / duration))
        detected = getattr(info, "language", None) if lang is None else lang
        return segments, detected
```

- [ ] **Step 4: Запустить тесты — зелёные (без моделей — smoke-тесты скипаются)**

Run: `.venv/bin/python -m pytest tests/test_asr.py -v`
Expected: PASS

- [ ] **Step 5: Коммит**

```bash
git add app/asr.py tests/test_asr.py
git commit -m "feat: обёртка faster-whisper с VAD и прогрессом"
```

---

### Task 4: translate.py — NLLB через CTranslate2

**Files:**
- Create: `app/translate.py`, `scripts/prepare-models.py`, `requirements-convert.txt` (создан в Task 0)
- Test: `tests/test_translate.py`

**Interfaces:**
- Consumes: `langmap.flores_code()`.
- Produces:
  - `translate.NllbTranslator(model_dir: Path)` с методом
    `translate_texts(texts: list[str], src_whisper_lang: str | None) -> list[str | None]`
    — `None` в позиции = перевод недоступен (нет языка в маппинге или ошибка чанка);
    чанками по 32 текста, beam_size=4.
  - `scripts/prepare-models.py` — скачивает в `models-dev/`:
    `whisper-{large-v3-turbo,medium,small}` (snapshot_download из `Systran/*`) и
    `nllb-600m-ct2` (конвертация `facebook/nllb-200-distilled-600M` через `ct2-transformers-converter`).

- [ ] **Step 1: Написать тесты**

`tests/test_translate.py`:
```python
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
```

- [ ] **Step 2: Запустить, убедиться в падении**

Run: `.venv/bin/python -m pytest tests/test_translate.py -v`
Expected: FAIL (`No module named app.translate`)

- [ ] **Step 3: Реализовать app/translate.py**

```python
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
```

- [ ] **Step 4: Запустить тесты — зелёные (smoke скипается без модели)**

Run: `.venv/bin/python -m pytest tests/test_translate.py -v`
Expected: PASS (1 skipped)

- [ ] **Step 5: Написать scripts/prepare-models.py**

```python
"""Скачивание и подготовка моделей в models-dev/ (нужен интернет).

Whisper: snapshot_download из Systran/* (уже в формате CTranslate2).
NLLB: скачивание facebook/nllb-200-distilled-600M + конвертация
ct2-transformers-converter (требует torch+transformers из requirements-convert.txt).
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS = REPO_ROOT / "models-dev"

WHISPER_REPOS = {
    "whisper-large-v3-turbo": "Systran/faster-whisper-large-v3-turbo",
    "whisper-medium": "Systran/faster-whisper-medium",
    "whisper-small": "Systran/faster-whisper-small",
}
NLLB_HF = "facebook/nllb-200-distilled-600M"
NLLB_OUT = MODELS / "nllb-600m-ct2"

def fetch_whisper() -> None:
    for name, repo in WHISPER_REPOS.items():
        dest = MODELS / name
        if dest.exists():
            print(f"[skip] {name}")
            continue
        print(f"[fetch] {repo} -> {dest}")
        snapshot_download(repo_id=repo, local_dir=dest)

def fetch_and_convert_nllb(venv_bin: Path) -> None:
    if NLLB_OUT.exists():
        print("[skip] nllb-600m-ct2")
        return
    hf_cache = MODELS / "_nllb_hf"
    if not (hf_cache / "config.json").exists():
        print(f"[fetch] {NLLB_HF}")
        snapshot_download(repo_id=NLLB_HF, local_dir=hf_cache)
    cmd = [
        str(venv_bin / "ct2-transformers-converter"),
        "--model", str(hf_cache),
        "--output_dir", str(NLLB_OUT),
        "--quantization", "int8",
        "--force",
    ]
    print("[convert]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    # конвертер не копирует токенизатор — переносим вручную
    shutil.copy(hf_cache / "sentencepiece.bpe.model", NLLB_OUT / "sentencepiece.bpe.model")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--venv-bin", default=str(REPO_ROOT / ".venv" / "bin"),
                    help="каталог с bin/ venv, где лежит ct2-transformers-converter")
    args = ap.parse_args()
    MODELS.mkdir(parents=True, exist_ok=True)
    fetch_whisper()
    fetch_and_convert_nllb(Path(args.venv_bin))
    print("Готово:", MODELS)

if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Установить зависимости конвертации и подготовить модели (разовый шаг, ~30–60 мин: torch ~2 ГБ, whisper-large-v3-turbo ~1.6 ГБ, medium ~3 ГБ, small ~0.5 ГБ, NLLB ~2.5 ГБ + конвертация)**

```bash
uv pip install -r requirements-convert.txt -p .venv
.venv/bin/python scripts/prepare-models.py
ls models-dev/
```
Expected: каталоги `whisper-large-v3-turbo`, `whisper-medium`, `whisper-small`, `nllb-600m-ct2` (в последнем — `model.bin`, `sentencepiece.bpe.model`, `shared_vocabulary*.json`)

- [ ] **Step 7: Smoke-тест перевода с реальной моделью**

Run: `.venv/bin/python -m pytest tests/test_translate.py -v`
Expected: PASS, 0 skipped (перевод «Assalomu alaykum» → непустая русская строка)

- [ ] **Step 8: Коммит**

```bash
git add app/translate.py tests/test_translate.py scripts/prepare-models.py requirements-convert.txt
git commit -m "feat: перевод NLLB-600M через CTranslate2 и подготовка моделей"
```

---

### Task 5: jobs.py — очередь задач и конвейер

**Files:**
- Create: `app/jobs.py`
- Test: `tests/test_jobs.py`

**Interfaces:**
- Consumes: `outputs.write_*` (Task 2), `Segment` (Task 3).
- Produces:
  - `jobs.Job` — dataclass: `id: str, filename: str, language: str, model_size: str, stage: str, progress: float, error: str | None, output_dir: str | None`. Значения stage: `queued → asr → translate → write → done | error`.
  - `jobs.JobManager(work_root: Path, output_root: Path, transcribe, translate)`, где
    `transcribe(path, language, model_size, on_progress) -> (segments, detected_lang)`,
    `translate(texts, detected_lang) -> list[str | None]`.
    Методы: `submit(src_path: Path, orig_name: str, language: str, model_size: str) -> Job`,
    `get(job_id) -> Job | None`, `all() -> list[Job]`.
  - Статус задачи дублируется в `work_root/<id>/status.json`.

- [ ] **Step 1: Написать тесты**

`tests/test_jobs.py`:
```python
import json, time
from pathlib import Path
from app.asr import Segment
from app.jobs import JobManager

def fake_transcribe(path, language, model_size, on_progress=None):
    on_progress and on_progress(1.0)
    return [Segment(start=0.0, end=2.0, text="Салом")], "uz"

def fake_translate(texts, lang):
    return ["Привет" for _ in texts]

def wait_stage(mgr, job_id, stage, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        j = mgr.get(job_id)
        if j and j.stage == stage:
            return j
        time.sleep(0.05)
    raise AssertionError(f"stage={stage} не достигнута, последний: {mgr.get(job_id)}")

def make_mgr(tmp_path, transcribe=fake_transcribe, translate=fake_translate):
    return JobManager(tmp_path / "work", tmp_path / "output", transcribe, translate)

def submit(mgr, tmp_path):
    src = tmp_path / "in.wav"; src.write_bytes(b"fake")
    return mgr.submit(src, "запись.wav", "uz", "turbo")

def test_pipeline_happy_path(tmp_path):
    mgr = make_mgr(tmp_path)
    job = submit(mgr, tmp_path)
    j = wait_stage(mgr, job.id, "done")
    out = Path(j.output_dir)
    assert (out / "transcript.txt").read_text(encoding="utf-8").strip().endswith("Салом")
    assert "Привет" in (out / "translation.txt").read_text(encoding="utf-8")
    assert (out / "result.docx").exists()
    status = json.loads((tmp_path / "work" / job.id / "status.json").read_text())
    assert status["stage"] == "done"

def test_error_is_reported(tmp_path):
    def boom(path, language, model_size, on_progress=None):
        raise ValueError("плохой файл")
    mgr = make_mgr(tmp_path, transcribe=boom)
    job = submit(mgr, tmp_path)
    j = wait_stage(mgr, job.id, "error")
    assert "плохой файл" in j.error

def test_translate_failure_marks_unavailable(tmp_path):
    def fail_translate(texts, lang):
        return [None for _ in texts]
    mgr = make_mgr(tmp_path, translate=fail_translate)
    job = submit(mgr, tmp_path)
    j = wait_stage(mgr, job.id, "done")
    assert "[перевод недоступен]" in (Path(j.output_dir) / "translation.txt").read_text(encoding="utf-8")
```

- [ ] **Step 2: Запустить, убедиться в падении**

Run: `.venv/bin/python -m pytest tests/test_jobs.py -v`
Expected: FAIL (`No module named app.jobs`)

- [ ] **Step 3: Реализовать app/jobs.py**

```python
"""Очередь задач: строго последовательная обработка в фоновом потоке."""

import json
import queue
import shutil
import threading
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from .asr import Segment
from .outputs import write_docx, write_translation, write_transcript

@dataclass
class Job:
    id: str
    filename: str
    language: str
    model_size: str
    stage: str = "queued"
    progress: float = 0.0
    error: str | None = None
    output_dir: str | None = None

class JobManager:
    def __init__(self, work_root: Path, output_root: Path, transcribe, translate):
        self.work_root = Path(work_root)
        self.output_root = Path(output_root)
        self._transcribe = transcribe
        self._translate = translate
        self._queue: queue.Queue[str] = queue.Queue()
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self.work_root.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)
        threading.Thread(target=self._worker, daemon=True).start()

    def submit(self, src_path: Path, orig_name: str, language: str, model_size: str) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], filename=orig_name,
                  language=language, model_size=model_size)
        job_dir = self.work_root / job.id
        job_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(orig_name).suffix or ".audio"
        dst = job_dir / f"input{suffix}"
        shutil.copyfile(src_path, dst)   # стрим-копия: не держим файл в RAM
        with self._lock:
            self._jobs[job.id] = job
        self._persist(job)
        self._queue.put(job.id)
        return job

    def get(self, job_id: str):
        with self._lock:
            return self._jobs.get(job_id)

    def all(self):
        with self._lock:
            return list(self._jobs.values())

    def _worker(self) -> None:
        while True:
            job_id = self._queue.get()
            job = self.get(job_id)
            if job is not None:
                self._run(job)

    def _set(self, job: Job, **fields) -> None:
        with self._lock:
            for k, v in fields.items():
                setattr(job, k, v)
            snapshot = asdict(job)
        self._persist_snapshot(job.id, snapshot)

    def _persist(self, job: Job) -> None:
        self._persist_snapshot(job.id, asdict(job))

    def _persist_snapshot(self, job_id: str, snapshot: dict) -> None:
        path = self.work_root / job_id / "status.json"
        path.write_text(json.dumps(snapshot, ensure_ascii=False), encoding="utf-8")

    def _run(self, job: Job) -> None:
        try:
            job_dir = self.work_root / job.id
            src = next(p for p in job_dir.iterdir() if p.name.startswith("input"))
            self._set(job, stage="asr", progress=0.0)

            def on_progress(p: float) -> None:
                self._set(job, progress=round(p, 4))

            segments, detected = self._transcribe(src, job.language, job.model_size, on_progress)
            self._set(job, stage="translate", progress=0.0)
            texts = [s.text for s in segments]
            translations = self._translate(texts, detected) if texts else []
            self._set(job, stage="write")
            out_dir = self.output_root / job.id
            out_dir.mkdir(parents=True, exist_ok=True)
            write_transcript(out_dir / "transcript.txt", segments)
            write_translation(out_dir / "translation.txt", segments, translations)
            write_docx(out_dir / "result.docx", job.filename, segments, translations)
            self._set(job, stage="done", output_dir=str(out_dir))
        except Exception as e:  # любая ошибка не валит сервис
            self._set(job, stage="error", error=f"{type(e).__name__}: {e}")
```

- [ ] **Step 4: Запустить тесты — зелёные**

Run: `.venv/bin/python -m pytest tests/test_jobs.py -v`
Expected: PASS

- [ ] **Step 5: Коммит**

```bash
git add app/jobs.py tests/test_jobs.py
git commit -m "feat: очередь задач с конвейером asr-перевод-запись"
```

---

### Task 6: main.py — FastAPI-сервис

**Files:**
- Create: `app/main.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `JobManager`, `Transcriber`, `NllbTranslator`, `config.*`.
- Produces:
  - `main.create_app(manager: JobManager | None = None) -> FastAPI` (DI для тестов).
  - Маршруты: `GET /` (index.html), `GET /static/*`, `GET /api/health`,
    `POST /api/jobs` (multipart: `file`, `language`=`uz|az|tg|auto`, `model_size`=`turbo|medium|small`; лимит 2 ГБ → 413),
    `GET /api/jobs`, `GET /api/jobs/{id}`, `GET /api/jobs/{id}/files/{name}` (`name` ∈ transcript.txt/translation.txt/result.docx, иначе 404).
  - Модульный `app = create_app()` для `uvicorn app.main:app`.

- [ ] **Step 1: Написать тесты**

`tests/test_api.py`:
```python
import time
from pathlib import Path
from fastapi.testclient import TestClient
from app.jobs import JobManager, Segment
from app.main import create_app

def fake_transcribe(path, language, model_size, on_progress=None):
    return [Segment(start=0.0, end=1.0, text="Салом")], "uz"

def fake_translate(texts, lang):
    return ["Привет" for _ in texts]

def make_client(tmp_path):
    mgr = JobManager(tmp_path / "work", tmp_path / "output", fake_transcribe, fake_translate)
    return TestClient(create_app(mgr)), mgr

def wait_done(mgr, job_id, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        j = mgr.get(job_id)
        if j and j.stage == "done":
            return j
        time.sleep(0.05)
    raise AssertionError("задача не завершена")

def test_health(tmp_path):
    client, _ = make_client(tmp_path)
    assert client.get("/api/health").json() == {"ok": True}

def test_full_job_flow(tmp_path):
    client, mgr = make_client(tmp_path)
    r = client.post("/api/jobs", data={"language": "uz", "model_size": "turbo"},
                    files={"file": ("запись.wav", b"fake wav", "audio/wav")})
    assert r.status_code == 200
    job_id = r.json()["id"]
    wait_done(mgr, job_id)
    detail = client.get(f"/api/jobs/{job_id}").json()
    assert detail["stage"] == "done"
    for name in ("transcript.txt", "translation.txt", "result.docx"):
        assert client.get(f"/api/jobs/{job_id}/files/{name}").status_code == 200
    assert client.get(f"/api/jobs/{job_id}/files/..%2Fstatus.json").status_code in (400, 404)

def test_upload_limit(tmp_path, monkeypatch):
    import app.main as m
    monkeypatch.setattr(m, "MAX_UPLOAD", 8)
    client, _ = make_client(tmp_path)
    r = client.post("/api/jobs", data={"language": "uz", "model_size": "small"},
                    files={"file": ("big.wav", b"x" * 100, "audio/wav")})
    assert r.status_code == 413
```

- [ ] **Step 2: Запустить, убедиться в падении**

Run: `.venv/bin/python -m pytest tests/test_api.py -v`
Expected: FAIL (`No module named app.main`)

- [ ] **Step 3: Реализовать app/main.py**

```python
"""HTTP-сервис: загрузка аудио, статусы задач, выдача результатов."""

import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .asr import Transcriber
from .config import models_dir, output_dir, web_dir, work_dir
from .jobs import JobManager
from .translate import NllbTranslator

MAX_UPLOAD = 2 * 1024 ** 3  # 2 ГБ
RESULT_FILES = {"transcript.txt", "translation.txt", "result.docx"}

def build_manager() -> JobManager:
    transcriber = Transcriber(models_dir())
    translator = NllbTranslator(models_dir() / "nllb-600m-ct2")
    return JobManager(work_dir(), output_dir(), transcriber.transcribe, translator.translate_texts)

def create_app(manager: JobManager | None = None) -> FastAPI:
    mgr = manager or build_manager()

    def job_dto(job):
        return {"id": job.id, "filename": job.filename, "language": job.language,
                "model_size": job.model_size, "stage": job.stage,
                "progress": job.progress, "error": job.error}

    app = FastAPI(title="Офлайн-расшифровка", docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=str(web_dir()), check_dir=False), name="static")

    @app.get("/")
    def index():
        return FileResponse(web_dir() / "index.html")

    @app.get("/api/health")
    def health():
        return {"ok": True}

    @app.post("/api/jobs")
    def create_job(file: UploadFile = File(...),
                   language: str = Form("auto"),
                   model_size: str = Form("turbo")):
        if language not in ("uz", "az", "tg", "auto"):
            raise HTTPException(400, "язык должен быть uz, az, tg или auto")
        if model_size not in ("turbo", "medium", "small"):
            raise HTTPException(400, "model_size должен быть turbo, medium или small")
        job_id = uuid.uuid4().hex[:12]
        tmp = work_dir() / f"upload-{job_id}"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        total = 0
        try:
            with tmp.open("wb") as out:
                while chunk := file.file.read(4 * 1024 * 1024):
                    total += len(chunk)
                    if total > MAX_UPLOAD:
                        raise HTTPException(413, "файл больше 2 ГБ")
                    out.write(chunk)
            job = mgr.submit(tmp, file.filename or "audio", language, model_size)
        finally:
            tmp.unlink(missing_ok=True)
        return {"id": job.id}

    @app.get("/api/jobs")
    def list_jobs():
        return [job_dto(j) for j in reversed(mgr.all())]

    @app.get("/api/jobs/{job_id}")
    def job_detail(job_id: str):
        job = mgr.get(job_id)
        if job is None:
            raise HTTPException(404, "задача не найдена")
        return job_dto(job)

    @app.get("/api/jobs/{job_id}/files/{name}")
    def job_file(job_id: str, name: str):
        if name not in RESULT_FILES:
            raise HTTPException(404, "файл не найден")
        job = mgr.get(job_id)
        if job is None or not job.output_dir:
            raise HTTPException(404, "задача не найдена")
        path = Path(job.output_dir) / name
        if not path.exists():
            raise HTTPException(404, "файл ещё не готов")
        return FileResponse(str(path), filename=name)

    return app

app = create_app()
```

- [ ] **Step 4: Запустить тесты — зелёные**

Run: `.venv/bin/python -m pytest tests/test_api.py -v`
Expected: PASS

- [ ] **Step 5: Коммит**

```bash
git add app/main.py tests/test_api.py
git commit -m "feat: HTTP API сервиса (загрузка, статусы, файлы)"
```

---

### Task 7: Веб-интерфейс

**Files:**
- Create: `web/index.html`, `web/app.js`, `web/style.css`
- Test: ручной smoke (в Task 6 тест `test_index_served` уже проверяет выдачу index.html)

**Interfaces:**
- Consumes: API из Task 6.
- Produces: страница загрузки со списком задач и скачиванием результатов.

- [ ] **Step 1: Написать web/index.html**

```html
<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Расшифровка аудио</title>
<link rel="stylesheet" href="/static/style.css">
</head>
<body>
<main>
  <h1>Расшифровка аудио и перевод на русский</h1>
  <section class="card">
    <form id="upload-form">
      <label>Аудиофайл (mp3, m4a, wav, ogg, flac, mp4 — до 2 ГБ)
        <input type="file" id="file" accept="audio/*,video/mp4" required>
      </label>
      <label>Язык записи
        <select id="language">
          <option value="uz">Узбекский</option>
          <option value="az">Азербайджанский</option>
          <option value="tg">Таджикский</option>
          <option value="auto">Определить автоматически</option>
        </select>
      </label>
      <label>Модель расшифровки
        <select id="model_size">
          <option value="turbo">Крупная (точнее, медленнее)</option>
          <option value="medium">Средняя</option>
          <option value="small">Небольшая (быстрее, грубее)</option>
        </select>
      </label>
      <button type="submit">Обработать</button>
      <span id="upload-status"></span>
    </form>
  </section>
  <section class="card">
    <h2>Задачи</h2>
    <table id="jobs">
      <thead><tr><th>Файл</th><th>Язык</th><th>Модель</th><th>Статус</th><th>Результат</th></tr></thead>
      <tbody></tbody>
    </table>
  </section>
  <section class="card" id="result" hidden>
    <h2>Результат: <span id="result-name"></span></h2>
    <div class="cols">
      <div><h3>Расшифровка</h3><pre id="transcript"></pre></div>
      <div><h3>Перевод</h3><pre id="translation"></pre></div>
    </div>
    <p><a id="dl-transcript" href="#" download>Скачать расшифровку (.txt)</a> ·
       <a id="dl-translation" href="#" download>Скачать перевод (.txt)</a> ·
       <a id="dl-docx" href="#" download>Скачать документ (.docx)</a></p>
  </section>
</main>
<script src="/static/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Написать web/app.js**

```javascript
const STAGES = {
  queued: "в очереди", asr: "расшифровка", translate: "перевод",
  write: "запись результата", done: "готово", error: "ошибка",
};

async function refresh() {
  const jobs = await (await fetch("/api/jobs")).json();
  const tbody = document.querySelector("#jobs tbody");
  tbody.innerHTML = "";
  for (const j of jobs) {
    const tr = document.createElement("tr");
    const status = j.stage === "error"
      ? `ошибка: ${j.error}`
      : `${STAGES[j.stage] || j.stage}${j.stage === "asr" ? " " + Math.round(j.progress * 100) + "%" : ""}`;
    tr.innerHTML = `<td>${esc(j.filename)}</td><td>${esc(j.language)}</td>` +
      `<td>${esc(j.model_size)}</td><td>${esc(status)}</td>` +
      `<td>${j.stage === "done" ? `<button data-id="${j.id}" data-name="${esc(j.filename)}">Открыть</button>` : ""}</td>`;
    const btn = tr.querySelector("button");
    if (btn) btn.onclick = () => showResult(j.id, j.filename);
    tbody.appendChild(tr);
  }
}

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, c =>
    ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
}

async function showResult(id, filename) {
  document.getElementById("result").hidden = false;
  document.getElementById("result-name").textContent = filename;
  for (const [elId, file] of [["transcript", "transcript.txt"], ["translation", "translation.txt"]]) {
    const r = await fetch(`/api/jobs/${id}/files/${file}`);
    document.getElementById(elId).textContent = r.ok ? await r.text() : "(нет данных)";
  }
  for (const [linkId, file] of [["dl-transcript", "transcript.txt"],
       ["dl-translation", "translation.txt"], ["dl-docx", "result.docx"]]) {
    document.getElementById(linkId).href = `/api/jobs/${id}/files/${file}`;
  }
}

document.getElementById("upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const status = document.getElementById("upload-status");
  const f = document.getElementById("file").files[0];
  if (!f) return;
  status.textContent = "Загрузка файла…";
  const body = new FormData();
  body.append("file", f);
  body.append("language", document.getElementById("language").value);
  body.append("model_size", document.getElementById("model_size").value);
  const r = await fetch("/api/jobs", { method: "POST", body });
  status.textContent = r.ok ? "Файл принят, задача в очереди." : `Ошибка: ${r.status} ${await r.text()}`;
  if (r.ok) refresh();
});

refresh();
setInterval(refresh, 2000);
```

- [ ] **Step 3: Написать web/style.css**

```css
body { font-family: system-ui, sans-serif; margin: 0; background: #f5f6f8; }
main { max-width: 900px; margin: 0 auto; padding: 16px; }
.card { background: #fff; border-radius: 8px; padding: 16px 20px; margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,.12); }
label { display: block; margin: 10px 0; }
input, select, button { font: inherit; padding: 6px 10px; margin-top: 4px; }
button { background: #2563eb; color: #fff; border: 0; border-radius: 6px; cursor: pointer; }
table { width: 100%; border-collapse: collapse; }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid #e5e7eb; }
.cols { display: flex; gap: 16px; }
.cols > div { flex: 1; min-width: 0; }
pre { white-space: pre-wrap; word-break: break-word; background: #f8f9fb;
      padding: 10px; border-radius: 6px; max-height: 400px; overflow: auto; }
#upload-status { margin-left: 10px; color: #555; }
```

- [ ] **Step 4: Добавить тест выдачи страницы в tests/test_api.py**

```python
def test_index_served(tmp_path):
    client, _ = make_client(tmp_path)
    r = client.get("/")
    assert r.status_code == 200 and "Загрузка" in r.text
```

Run: `.venv/bin/python -m pytest tests/test_api.py -v`
Expected: PASS (index.html отдаётся с `<link rel="stylesheet" href="/static/style.css">`, слово «Загрузка» присутствует)

- [ ] **Step 5: Ручной smoke**

```bash
mkdir -p data/work data/output
TRANSLATOR_MODELS_DIR=models-dev .venv/bin/python -m uvicorn app.main:app --port 8080 &
sleep 2 && curl -s localhost:8080/api/health && curl -s -o /dev/null -w "%{http_code}\n" localhost:8080/
kill %1
```
Expected: `{"ok":true}` и `200`. Открыть `http://localhost:8080` в браузере и глазами проверить форму (на dev-машине с моделями можно прогнать короткий wav целиком).

- [ ] **Step 6: Коммит**

```bash
git add web/ tests/test_api.py
git commit -m "feat: веб-интерфейс загрузки и просмотра результатов"
```

---

### Task 8: FLEURS-тесты и измерение WER

**Files:**
- Create: `scripts/fetch-fleurs.py`, `scripts/eval-wer.py`
- Create (по результатам): `docs/benchmark.md`

**Interfaces:**
- Consumes: `asr.Transcriber` (Task 3), модели из `models-dev/` (Task 4).
- Produces:
  - `data/fleurs/<cfg>/NNN.<ext>` + `NNN.ref.txt` (клипы + эталонные транскрипты), cfg ∈ `uz_uz, az_az, tg_tg`.
  - `docs/benchmark.md` — WER `turbo`/`medium`/`small` по каждому языку, время обработки → выбор модели по умолчанию подтверждается или меняется.

- [ ] **Step 1: Написать scripts/fetch-fleurs.py**

```python
"""Скачивание тестовых клипов FLEURS (ethal-транскрипты) для uz/az/tg.

Клипы и эталоны кладутся в data/fleurs/<cfg>/NNN.<ext> и NNN.ref.txt.
Нужен интернет. Используются parquet-шарды датасета google/fleurs с HF.
"""

import argparse
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

REPO = "google/fleurs"
AUDIO_COL_CANDIDATES = ("audio",)
TEXT_COL_CANDIDATES = ("raw_transcription", "transcription")

def fetch(cfg: str, n: int, out_root: Path) -> None:
    out_dir = out_root / cfg
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = list(out_dir.glob("*.ref.txt"))
    if len(existing) >= n:
        print(f"[skip] {cfg}: уже {len(existing)} клипов")
        return
    api = HfApi()
    files = [f for f in api.list_repo_files(REPO, repo_type="dataset")
             if f.startswith(f"data/{cfg}/") and "test" in f and f.endswith(".parquet")]
    if not files:
        raise SystemExit(f"не найдены test-шарды для {cfg}; файлы: "
                         f"{[f for f in api.list_repo_files(REPO, repo_type='dataset') if cfg in f][:20]}")
    saved = len(existing)
    for shard in sorted(files):
        local = hf_hub_download(REPO, shard, repo_type="dataset")
        table = pq.read_table(local)
        audio_col = next(c for c in table.column_names if c in AUDIO_COL_CANDIDATES or "audio" in c)
        text_col = next((c for c in TEXT_COL_CANDIDATES if c in table.column_names),
                        table.column_names[-1])
        for row in table.select([audio_col, text_col]).to_pylist():
            if saved >= n:
                return
            a = row[audio_col]
            data = a["bytes"] if isinstance(a, dict) and a.get("bytes") else None
            ext = "mp3"
            if isinstance(a, dict) and a.get("path"):
                ext = Path(a["path"]).suffix.lstrip(".") or ext
            idx = f"{saved:03d}"
            if data:
                (out_dir / f"{idx}.{ext}").write_bytes(data)
            else:  # аудио лежит отдельным файлом
                shutil.copy(a["path"], out_dir / f"{idx}.{ext}")
            (out_dir / f"{idx}.ref.txt").write_text(row[text_col] or "", encoding="utf-8")
            saved += 1
            print(f"[{cfg}] {idx} сохранён")

if __name__ == "__main__":
    import shutil
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--out", default="data/fleurs")
    args = ap.parse_args()
    for cfg in ("uz_uz", "az_az", "tg_tg"):
        fetch(cfg, args.n, Path(args.out))
```

- [ ] **Step 2: Скачать клипы**

```bash
.venv/bin/python scripts/fetch-fleurs.py --n 15
ls data/fleurs/uz_uz | head
```
Expected: 15 клипов + 15 эталонов на каждый язык (если для какого-то cfg шарды называются иначе — скрипт подскажет список файлов и правится фильтр).

- [ ] **Step 3: Написать scripts/eval-wer.py**

```python
"""WER моделей Whisper на клипах FLEURS + время обработки.

Запуск: .venv/bin/python scripts/eval-wer.py --model turbo [--langs uz_uz az_az tg_tg]
Результаты печатаются и дописываются в docs/benchmark.md.
"""

import argparse
import resource
import time
from pathlib import Path

from app.asr import Transcriber

def lang_of(cfg: str) -> str:
    return cfg.split("_")[0]          # uz_uz -> uz, az_az -> az, tg_tg -> tg

def norm_words(text: str) -> list[str]:
    words = []
    for raw in text.lower().split():
        clean = "".join(ch for ch in raw if ch.isalnum())
        if clean:
            words.append(clean)
    return words

def wer(ref: str, hyp: str) -> float:
    ref_words, hyp_words = norm_words(ref), norm_words(hyp)
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    prev = list(range(len(hyp_words) + 1))
    for i, r in enumerate(ref_words, 1):
        cur = [i]
        for j, h in enumerate(hyp_words, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1] / len(ref_words)

if __name__ == "__main__":
    assert wer("салом алейкум", "салом алейкум") == 0.0
    assert wer("салом алейкум", "салом") == 0.5
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="turbo", choices=["turbo", "medium", "small"])
    ap.add_argument("--models-root", default="models-dev")
    ap.add_argument("--data", default="data/fleurs")
    ap.add_argument("--langs", nargs="*", default=["uz_uz", "az_az", "tg_tg"])
    args = ap.parse_args()
    t = Transcriber(args.models_root)
    lines = [f"## Модель {args.model}", "",
             "| язык | клипов | WER | время, с |", "|---|---|---|---|"]
    for cfg in args.langs:
        d = Path(args.data) / cfg
        refs = sorted(d.glob("*.ref.txt"))
        scores, seconds = [], 0.0
        for ref_file in refs:
            audio = next(p for p in d.glob(f"{ref_file.stem}.*")
                         if not p.name.endswith(".ref.txt"))
            ref = ref_file.read_text(encoding="utf-8")
            t0 = time.time()
            segs, _ = t.transcribe(audio, lang_of(cfg), args.model)
            seconds += time.time() - t0
            hyp = " ".join(s.text for s in segs)
            scores.append(wer(ref, hyp))
        mean = sum(scores) / max(1, len(scores))
        print(f"{cfg}: WER={mean:.2%} (n={len(scores)}), {seconds:.1f} c")
        lines.append(f"| {cfg} | {len(scores)} | {mean:.2%} | {seconds:.1f} |")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    lines += ["", f"max RSS: {rss / 1e6:.2f} ГБ"]
    Path("docs").mkdir(exist_ok=True)
    with open("docs/benchmark.md", "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
```

- [ ] **Step 4: Прогнать eval для трёх моделей (на dev-машине, интернет не нужен — модели локальные)**

```bash
for m in turbo medium small; do .venv/bin/python scripts/eval-wer.py --model $m; done
cat docs/benchmark.md
```
Expected: таблица WER по 3 языкам × 3 модели; max RSS < 8 ГБ. Если WER turbo заметно лучше — модель по умолчанию остаётся `turbo` (ожидаемо).

- [ ] **Step 5: Коммит**

```bash
git add scripts/fetch-fleurs.py scripts/eval-wer.py docs/benchmark.md
git commit -m "test: измерение WER моделей на FLEURS (uz/az/tg)"
```

---

### Task 9: Портативный бандл (start.sh + сборка)

**Files:**
- Create: `start.sh` (шаблон в репо), `scripts/build-offline-bundle.sh`

**Interfaces:**
- Consumes: всё приложение (app/, web/), requirements.txt, модели из `models-dev/` (или скачивает через prepare-models.py).
- Produces: `translator-bundle-<дата>.tar.gz` со структурой спеки §6 и встроенным `selftest.sh`.

- [ ] **Step 1: Написать start.sh**

```bash
#!/bin/sh
# Портативный запуск. Не требует исполняемого бита: bash start.sh / sh start.sh
# Переменные: PORT (по умолчанию 8080), HOST (по умолчанию 127.0.0.1)
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PY="$ROOT/python/bin/python3.11"

# Проверка, что интерпретатор исполняется (на noexec-монтировании упадёт)
if ! "$PY" -c "pass" >/dev/null 2>&1; then
  cp "$PY" /tmp/.offline-translator-python || exit 1
  PY=/tmp/.offline-translator-python
  export PYTHONHOME="$ROOT/python"   # копия вне родного префикса
fi

export HF_HUB_OFFLINE=1
export TRANSLATOR_ROOT="$ROOT"
export PYTHONPATH="$ROOT/site-packages"

exec "$PY" -m uvicorn app.main:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8080}"
```

- [ ] **Step 2: Написать scripts/build-offline-bundle.sh**

```bash
#!/bin/bash
# Сборка офлайн-бандла. Запускается на машине с ИНТЕРНЕТОМ (любая ОС с bash+python).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATE=$(date +%Y%m%d)
BUNDLE="$REPO/translator-bundle-$DATE"
PYURL="https://github.com/astral-sh/python-build-standalone/releases/download/20241016/cpython-3.11.10+20241016-x86_64-unknown-linux-gnu-install_only.tar.gz"

echo "==> Каталог бандла: $BUNDLE"
mkdir -p "$BUNDLE"/{app,web,scripts,models,site-packages,output,work,logs,testdata}
cp -r "$REPO/app"/*.py "$BUNDLE/app/"
cp -r "$REPO/web"/* "$BUNDLE/web/"
cp "$REPO/start.sh" "$BUNDLE/start.sh"
cp "$REPO/requirements.txt" "$BUNDLE/requirements.txt"

echo "==> Портативный CPython 3.11 (linux x86_64)"
mkdir -p "$BUNDLE/python"
curl -L "$PYURL" | tar xz -C "$BUNDLE/python" --strip-components=1

echo "==> Linux-колёса в site-packages (manylinux)"
pip3 install \
  --platform manylinux2014_x86_64 --platform manylinux_2_27_x86_64 --platform manylinux_2_28_x86_64 \
  --python-version 3.11 --only-binary=:all: \
  --target "$BUNDLE/site-packages" -r "$REPO/requirements.txt"

echo "==> Модели"
if [ -d "$REPO/models-dev/whisper-large-v3-turbo" ] && [ -d "$REPO/models-dev/nllb-600m-ct2" ]; then
  cp -r "$REPO/models-dev/whisper-large-v3-turbo" "$REPO/models-dev/whisper-medium" \
        "$REPO/models-dev/whisper-small" "$REPO/models-dev/nllb-600m-ct2" "$BUNDLE/models/"
else
  echo "Моделей нет в models-dev/ — запускаю prepare-models.py"
  "$REPO/.venv/bin/python" "$REPO/scripts/prepare-models.py"
  cp -r "$REPO/models-dev/whisper-large-v3-turbo" "$REPO/models-dev/whisper-medium" \
        "$REPO/models-dev/whisper-small" "$REPO/models-dev/nllb-600m-ct2" "$BUNDLE/models/"
fi

echo "==> Тестовый клип для selftest"
TESTCLIP=$(find "$REPO/data/fleurs" -name "001.*" ! -name "*.ref.txt" | head -n1 || true)
if [ -n "$TESTCLIP" ]; then
  cp "$TESTCLIP" "$BUNDLE/testdata/test.${TESTCLIP##*.}"
else
  (cd "$BUNDLE" && "$REPO/.venv/bin/python" - <<'EOF'
import wave
w = wave.open("testdata/test.wav", "wb")
w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
w.writeframes(b"\x00\x00" * (16000 * 3))   # 3 секунды тишины
w.close()
EOF
  )
fi

echo "==> selftest.sh"
cat > "$BUNDLE/selftest.sh" <<'EOF'
#!/bin/sh
# Проверка бандла на целевой машине: bash selftest.sh
set -e
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PORT=8099
TRANSLATOR_ROOT="$ROOT" HF_HUB_OFFLINE=1 PYTHONPATH="$ROOT/site-packages" \
  "$ROOT/python/bin/python3.11" -m uvicorn app.main:app --host 127.0.0.1 --port $PORT &
SRV=$!
trap "kill $SRV 2>/dev/null" EXIT
for i in $(seq 1 60); do
  curl -s "http://127.0.0.1:$PORT/api/health" | grep -q ok && break
  sleep 2
done
curl -s "http://127.0.0.1:$PORT/api/health" | grep -q ok || { echo "FAIL: сервис не поднялся"; exit 1; }
echo "Сервис поднялся. Отправляю тестовый файл (small-модель)..."
CLIP="$ROOT/testdata/test.wav"
[ -f "$CLIP" ] || CLIP=$(find "$ROOT/testdata" -type f | head -n1)
JOB=$(curl -s -F "file=@$CLIP" -F language=auto -F model_size=small "http://127.0.0.1:$PORT/api/jobs")
ID=$(printf '%s' "$JOB" | sed -n 's/.*"id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')
for i in $(seq 1 150); do
  ST=$(curl -s "http://127.0.0.1:$PORT/api/jobs/$ID" | sed -n 's/.*"stage"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')
  [ "$ST" = "done" ] && break
  [ "$ST" = "error" ] && { echo "FAIL: задача завершилась ошибкой"; exit 1; }
  sleep 2
done
for f in transcript.txt translation.txt result.docx; do
  curl -sf "http://127.0.0.1:$PORT/api/jobs/$ID/files/$f" -o /dev/null || { echo "FAIL: нет $f"; exit 1; }
done
echo "PASS: бандл работает офлайн."
EOF

echo "==> Упаковка"
tar czf "$BUNDLE.tar.gz" -C "$REPO" "$(basename "$BUNDLE")"
du -sh "$BUNDLE.tar.gz"
echo "Готово: $BUNDLE.tar.gz — скопировать содержимое на флешку (ext4), затем на Astra: bash selftest.sh"
```

- [ ] **Step 3: Собрать бандл**

```bash
chmod +x scripts/build-offline-bundle.sh
bash scripts/build-offline-bundle.sh
tar tzf translator-bundle-*.tar.gz | head -20
```
Expected: в архиве есть `start.sh`, `selftest.sh`, `python/bin/python3.11`,
`site-packages/` (ctranslate2, faster_whisper, fastapi, uvicorn, av, ...),
`models/whisper-large-v3-turbo`, `models/nllb-600m-ct2`, `testdata/`.

- [ ] **Step 4: Статические проверки (на dev-машине)**

```bash
# синтаксис shell-скриптов
bash -n start.sh && bash -n scripts/build-offline-bundle.sh && echo OK
# все .py компилируются целевой версией
.venv/bin/python -m compileall -q app scripts
```
Expected: OK, без ошибок компиляции. Полная проверка бандла — на целевой машине Astra (`bash selftest.sh`, она офлайн по определению); пользователь запускает сам или присылает вывод.

- [ ] **Step 5: Коммит**

```bash
git add start.sh scripts/build-offline-bundle.sh
git commit -m "feat: сборка портативного офлайн-бандла и selftest"
```

---

### Task 10: README и передача пользователю

**Files:**
- Create: `README.md`, `docs/deployment.md`

**Interfaces:**
- Produces: инструкция по сборке, переносу на флешку, запуску и диагностике.

- [ ] **Step 1: Написать README.md**

```markdown
# Офлайн-расшифровка аудио (узбекский/таджикский/азербайджанский) с переводом на русский

Локальный веб-сервис для Astra Linux: расшифровка faster-whisper + перевод NLLB-200,
полностью без интернета, развёртывание с флешки без установки.

## Быстрый старт (разработка, машина с интернетом)
1. `pip3 install --user uv && uv venv .venv --python 3.11`
2. `uv pip install -r requirements-dev.txt -p .venv`
3. Конвертация NLLB (разово, ~30–60 мин): `uv pip install -r requirements-convert.txt -p .venv && .venv/bin/python scripts/prepare-models.py`
4. Тестовые клипы: `.venv/bin/python scripts/fetch-fleurs.py --n 15`
5. `.venv/bin/python -m pytest -v` — все тесты.
6. Сервис: `TRANSLATOR_MODELS_DIR=models-dev .venv/bin/python -m uvicorn app.main:app --port 8080` → http://localhost:8080

## Сборка бандла для Astra
`bash scripts/build-offline-bundle.sh` → `translator-bundle-<дата>.tar.gz` (~6–7 ГБ)

## Развёртывание на Astra (без интернета)
1. Флешка ext4 (или exFAT/FAT32 — тоже работает), ≥ 16 ГБ.
2. Распаковать архив в корень флешки: получится каталог `translator-bundle-...`.
3. На Astra: `cd /media/.../translator-bundle-... && bash selftest.sh` — должно напечатать `PASS`.
4. Работа: `bash start.sh`, открыть в браузере `http://127.0.0.1:8080`
   (для доступа из локальной сети: `HOST=0.0.0.0 bash start.sh`).

## Диагностика
- «Permission denied» при запуске — носитель смонтирован с noexec: start.sh сам
  копирует python в /tmp (нужно место ~100 МБ).
- Порт занят: `PORT=8081 bash start.sh`.
- Не хватает памяти: выберите модель medium/small в интерфейсе.
```

- [ ] **Step 2: Написать docs/deployment.md** — развернуть спеку §6 в пошаговую инструкцию:
  сборка → копирование на флешку (ext4 рекомендация, команды `mkfs.ext4`/`rsync`) →
  selftest на Astra → штатная работа → known issues (noexec/PYTHONHOME, FAT32 4 ГБ
  на файл — наши файлы меньше, exFAT поддержка, PARSEC-политики съёмных носителей —
  к администраторам).

- [ ] **Step 3: Финальный прогон всех тестов**

```bash
.venv/bin/python -m pytest -v
```
Expected: все PASS (кроме model-dependent skip, если модели не готовы).

- [ ] **Step 4: Коммит и завершение**

```bash
git add README.md docs/deployment.md
git commit -m "docs: инструкции по разработке, сборке и развёртыванию"
```

Передать пользователю: архив бандла + просьбу выполнить `bash selftest.sh` на целевой машине и прислать вывод; при появлении реальных записей — проверить качество по спеке (§8).
