from pathlib import Path
import importlib
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
    assert cfg.work_dir() == cfg.root() / "data" / "work"
    importlib.reload(cfg)  # вернуть состояние другим тестам
