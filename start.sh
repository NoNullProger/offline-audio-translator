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
