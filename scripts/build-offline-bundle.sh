#!/bin/bash
# Сборка офлайн-бандла. Запускается на машине с ИНТЕРНЕТОМ (любая ОС с bash+python3+pip).
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
PIP_FLAGS=(--platform manylinux2014_x86_64 --platform manylinux_2_27_x86_64 --platform manylinux_2_28_x86_64
           --python-version 3.11 --only-binary=:all: --target "$BUNDLE/site-packages")
if ! pip3 install "${PIP_FLAGS[@]}" -r "$REPO/requirements.txt"; then
  echo "Официальный pypi недоступен — пробую зеркало tuna"
  pip3 install "${PIP_FLAGS[@]}" --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    -r "$REPO/requirements.txt"
fi

echo "==> Модели"
if [ -d "$REPO/models-dev/whisper-large-v3" ] && [ -d "$REPO/models-dev/nllb-600m-ct2" ]; then
  cp -r "$REPO/models-dev/whisper-large-v3" "$REPO/models-dev/whisper-medium" \
        "$REPO/models-dev/whisper-small" "$REPO/models-dev/nllb-600m-ct2" \
        "$REPO/models-dev/vosk-onnx" "$BUNDLE/models/"
else
  echo "Моделей нет в models-dev/ — запускаю prepare-models.py"
  "$REPO/.venv/bin/python" "$REPO/scripts/prepare-models.py"
  cp -r "$REPO/models-dev/whisper-large-v3" "$REPO/models-dev/whisper-medium" \
        "$REPO/models-dev/whisper-small" "$REPO/models-dev/nllb-600m-ct2" \
        "$REPO/models-dev/vosk-onnx" "$BUNDLE/models/"
fi

echo "==> Тестовый клип для selftest"
TESTCLIP=$(find "$REPO/data/fleurs" -name "001.*" ! -name "*.ref.txt" -print -quit || true)
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
