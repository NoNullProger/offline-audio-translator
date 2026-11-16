"""WER Vosk/sherpa-onnx (ONNX) на тех же клипах FLEURS — сравнение с Whisper.

Модели качаются с зеркал alphacep на Hugging Face:
- uz: alphacep/vosk-model-small-streaming-uz
- tg: alphacep/vosk-model-tajik

Запуск: .venv/bin/python scripts/eval-vosk-wer.py
Результаты дописываются в docs/benchmark.md.
"""

import argparse
import onnx
import sys
import time
from pathlib import Path

import av
import numpy as np
import sherpa_onnx
from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODELS_DIR = Path(__file__).resolve().parent.parent / "models-dev" / "vosk-onnx"

VOSK_ONNX = {
    "uz": "alphacep/vosk-model-small-streaming-uz",
    "tg": "alphacep/vosk-model-tajik",
}


def ensure_model(lang: str) -> Path:
    dest = MODELS_DIR / lang
    if not (dest / "am-onnx" / "encoder.onnx").exists():
        print(f"[fetch] {VOSK_ONNX[lang]} -> {dest}")
        snapshot_download(repo_id=VOSK_ONNX[lang], local_dir=dest)
    return dest


def pcm16k(path: Path) -> np.ndarray:
    resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
    chunks = []
    with av.open(str(path)) as c:
        for frame in c.decode(audio=0):
            for rf in resampler.resample(frame):
                chunks.append(rf.to_ndarray().reshape(-1))
    return np.concatenate(chunks).astype("int16") if chunks else np.zeros(1, dtype="int16")


def transcribe(rec, pcm: np.ndarray) -> str:
    s = rec.create_stream()
    s.accept_waveform(16000, pcm.astype("float32") / 32768.0)
    if isinstance(rec, sherpa_onnx.OnlineRecognizer):
        s.input_finished()
        while rec.is_ready(s):
            rec.decode_stream(s)
        return rec.get_result(s)
    rec.decode_stream(s)
    return s.result.text


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
    args = ap.parse_args()
    cfg_of = {"uz": "uz_uz", "tg": "tg_tj"}
    lines = ["## Vosk ONNX (sherpa-onnx), те же клипы FLEURS", "",
             "| модель | язык | клипов | WER | время, с |", "|---|---|---|---|---|"]
    for lang in args.langs:
        model_path = ensure_model(lang)
        # uz-модель — стриминговый zipformer2, tg — non-streaming:
        # у streaming-энкодера в сигнатуре есть cached_* состояния,
        # поэтому тип загрузчика определяем по наличию входа x_lens
        enc = onnx.load(str(model_path / "am-onnx" / "encoder.onnx"), load_external_data=False)
        is_streaming = not any(i.name == "x_lens" for i in enc.graph.input)
        kwargs = dict(
            encoder=str(model_path / "am-onnx" / "encoder.onnx"),
            decoder=str(model_path / "am-onnx" / "decoder.onnx"),
            joiner=str(model_path / "am-onnx" / "joiner.onnx"),
            tokens=str(model_path / "lang" / "tokens.txt"),
            sample_rate=16000,
            feature_dim=80,
        )
        if is_streaming:
            rec = sherpa_onnx.OnlineRecognizer.from_transducer(**kwargs)
        else:
            rec = sherpa_onnx.OfflineRecognizer.from_transducer(**kwargs)
        d = Path(args.data) / cfg_of[lang]
        scores, seconds = [], 0.0
        for ref_file in sorted(d.glob("*.ref.txt")):
            base = ref_file.name[:-len(".ref.txt")]
            audio = next(p for p in d.glob(f"{base}.*") if not p.name.endswith(".ref.txt"))
            ref = ref_file.read_text(encoding="utf-8")
            t0 = time.time()
            hyp = transcribe(rec, pcm16k(audio))
            seconds += time.time() - t0
            scores.append(wer(ref, hyp))
            print(f"  {cfg_of[lang]}/{base}: WER={scores[-1]:.0%}")
        mean = sum(scores) / max(1, len(scores))
        print(f"vosk-{lang}: WER={mean:.2%} (n={len(scores)}), {seconds:.1f} c")
        lines.append(f"| vosk-{lang} | {cfg_of[lang]} | {len(scores)} | {mean:.2%} | {seconds:.1f} |")
    with open("docs/benchmark.md", "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
