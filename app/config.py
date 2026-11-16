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
    return root() if os.environ.get("TRANSLATOR_ROOT") else root() / "data"
