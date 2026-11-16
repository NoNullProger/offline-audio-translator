"""WER моделей Vosk на тех же клипах FLEURS, что и у Whisper (сравнение).

Модели Vosk (Kaldi-формат) качаются с alphacephei.com, если их ещё нет:
- uz: vosk-model-small-uz-0.22 (~49 МБ)
- tg: vosk-model-tg-0.22 (~327 МБ)

Запуск: .venv/bin/python scripts/eval-vosk-wer.py
Результаты дописываются в docs/benchmark.md.
"""

import argparse
import json
import shutil
import subprocess
import sys
import wave
from pathlib import Path

from vosk import Model, KaldiRecognizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODELS_DIR = Path(__file__).resolve().parent.parent / "models-dev" / "vosk"

VOSK_MODELS = {
    "uz": ("vosk-model-small-uz-0.22",
           "https://alphacephei.com/vosk/models/vosk-model-small-uz-0.22.zip"),
    "tg": ("vosk-model-tg-0.22",
           "https://alphacephei.com/vosk/models/vosk-model-tg-0.22.zip"),
}


def ensure_model(lang: str) -> Path:
    name, url = VOSK_MODELS[lang]
    dest = MODELS_DIR / name
    if dest.exists():
        return dest
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = MODELS_DIR / f"{name}.zip"
    print(f"[fetch] {url}")
    subprocess.run(
        ["curl", "-sL", "--fail", "-A",
         "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
         "-o", str(zip_path), "--max-time", "900", url],
        check=True,
    )
    shutil.unpack_archive(zip_path, MODELS_DIR)
    zip_path.unlink()
    return dest


def decode_pcm16k(path: Path) -> bytes:
    """PCM 16 кГц моно int16 через PyAV (вход — любой формат)."""
    import av
    import numpy as np
    resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
    chunks = []
    with av.open(str(path)) as c:
        for frame in c.decode(audio=0):
            for rf in resampler.resample(frame):
                chunks.append(rf.to_ndarray())
    if not chunks:
        return b""
    pcm = np.concatenate(chunks).astype("int16")
    return pcm.tobytes()


def transcribe_vosk(model: Model, pcm: bytes) -> str:
    rec = KaldiRecognizer(model, 16000)
    rec.SetWords(False)
    texts = []
    for i in range(0, len(pcm), 8000):
        if rec.AcceptWaveform(pcm[i:i + 8000]):
            texts.append(json.loads(rec.Result()).get("text", ""))
    texts.append(json.loads(rec.FinalResult()).get("text", ""))
    return " ".join(t for t in texts if t)


def norm_words(text: str) -> list:
    out = []
    for raw in text.lower().split():
        clean = "".join(ch for ch in raw if ch.isalnum())
        if clean:
            out.append(clean)
    return out


def wer(ref: str, hyp: str) -> float:
    r, h = norm_words(ref), norm_words(hyp)
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i]
        for j, hw in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw)))
        prev = cur
    return prev[-1] / len(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="*", default=["uz", "tg"])
    ap.add_argument("--data", default="data/fleurs")
    ap.add_argument("--cfgs", nargs="*",
                    default=["uz_uz", "az_az", "tg_tj"])
    args = ap.parse_args()
    cfg_of = dict(zip(["uz_uz", "az_az", "tg_tj"], ["uz", "az", "tg"]))
    lines = ["## Vosk (те же клипы FLEURS)", "",
             "| модель | язык | клипов | WER |", "|---|---|---|---|"]
    for lang in args.langs:
        model_path = ensure_model(lang)
        model = Model(str(model_path))
        cfg = next(c for c in args.cfgs if cfg_of.get(c) == lang)
        d = Path(args.data) / cfg
        scores = []
        for ref_file in sorted(d.glob("*.ref.txt")):
            base = ref_file.name[:-len(".ref.txt")]
            audio = next(p for p in d.glob(f"{base}.*") if not p.name.endswith(".ref.txt"))
            ref = ref_file.read_text(encoding="utf-8")
            hyp = transcribe_vosk(model, decode_pcm16k(audio))
            scores.append(wer(ref, hyp))
        mean = sum(scores) / max(1, len(scores))
        print(f"vosk-{lang} ({cfg}): WER={mean:.2%} (n={len(scores)})")
        lines.append(f"| vosk-model-{lang} | {cfg} | {len(scores)} | {mean:.2%} |")
    with open("docs/benchmark.md", "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
