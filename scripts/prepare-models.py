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
