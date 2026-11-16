import struct
import wave
from pathlib import Path

import pytest

from app.asr import Segment, Transcriber, validate_audio, MODEL_DIRS


def make_wav(path: Path, seconds: float = 0.5):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        frames = b"".join(struct.pack("<h", 0) for _ in range(int(16000 * seconds)))
        w.writeframes(frames)


def test_model_dirs():
    assert MODEL_DIRS["turbo"] == "whisper-large-v3-turbo"


def test_segment_fields():
    s = Segment(start=1.0, end=2.5, text="привет")
    assert (s.start, s.end, s.text) == (1.0, 2.5, "привет")


def test_validate_audio_bad_file(tmp_path):
    p = tmp_path / "bad.wav"
    p.write_bytes(b"not audio at all")
    with pytest.raises(ValueError):
        validate_audio(p)


def test_validate_audio_ok(tmp_path):
    p = tmp_path / "ok.wav"
    make_wav(p)
    validate_audio(p)  # не бросает


def test_unknown_model_size(tmp_path):
    t = Transcriber(tmp_path)
    with pytest.raises(ValueError):
        t._model("huge")


def test_missing_model_dir(tmp_path):
    t = Transcriber(tmp_path)
    with pytest.raises(RuntimeError, match="модель не найдена"):
        t._model("small")
