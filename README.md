# Офлайн-расшифровка аудио (узбекский/таджикский/азербайджанский) с переводом на русский

Локальный веб-сервис для Astra Linux: гибридная расшифровка (узбекский и
таджикский — специализированные ONNX-модели Vosk через sherpa-onnx;
азербайджанский — Whisper large-v3) + перевод NLLB-200, полностью без
интернета, развёртывание с флешки без установки.

Качество (WER на FLEURS, docs/benchmark.md): узбекский 22.5%, таджикский 8.1%,
азербайджанский 18.2%. Язык выбирается вручную (автоопределение ненадёжно
различает узбекский и азербайджанский).

Спека: `docs/superpowers/specs/2026-11-16-astra-offline-translator-design.md`
План: `docs/superpowers/plans/2026-11-16-offline-translator.md`

## Быстрый старт (разработка, машина с интернетом)

1. `pip3 install --user uv && uv venv .venv --python 3.11`
2. `uv pip install -r requirements-dev.txt -p .venv`
3. Модели (разово, ~30–60 мин, ~10 ГБ):
   `uv pip install -r requirements-convert.txt -p .venv && .venv/bin/python scripts/prepare-models.py`
4. Тестовые клипы: `.venv/bin/python scripts/fetch-fleurs.py --n 15`
5. `.venv/bin/python -m pytest -v` — все тесты.
6. Сервис: `TRANSLATOR_MODELS_DIR=models-dev .venv/bin/python -m uvicorn app.main:app --port 8080`
   → http://localhost:8080

## Сборка бандла для Astra

```bash
bash scripts/build-offline-bundle.sh    # → translator-bundle-<дата>.tar.gz (~5.4 ГБ)
```

## Развёртывание на Astra (без интернета)

Подробно — в [docs/deployment.md](docs/deployment.md). Кратко:

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
