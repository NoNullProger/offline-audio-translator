"""Скачивание тестовых клипов FLEURS (эталонные транскрипты) для uz/az/tg.

Клипы и эталоны кладутся в data/fleurs/<cfg>/NNN.<ext> и NNN.ref.txt.
Нужен интернет. Используются parquet-шарды датасета google/fleurs с HF.
"""

import argparse
import shutil
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

REPO = "google/fleurs"


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
        all_cfg = [f for f in api.list_repo_files(REPO, repo_type="dataset") if cfg in f][:20]
        raise SystemExit(f"не найдены test-шарды для {cfg}; файлы: {all_cfg}")
    saved = len(existing)
    for shard in sorted(files):
        local = hf_hub_download(REPO, shard, repo_type="dataset")
        table = pq.read_table(local)
        audio_col = next(c for c in table.column_names if "audio" in c)
        text_col = next((c for c in ("raw_transcription", "transcription")
                         if c in table.column_names), table.column_names[-1])
        for row in table.select([audio_col, text_col]).to_pylist():
            if saved >= n:
                return
            a = row[audio_col]
            ext = "mp3"
            data = None
            if isinstance(a, dict) and a.get("bytes"):
                data = a["bytes"]
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--out", default="data/fleurs")
    args = ap.parse_args()
    for cfg in ("uz_uz", "az_az", "tg_tg"):
        fetch(cfg, args.n, Path(args.out))
