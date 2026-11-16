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
