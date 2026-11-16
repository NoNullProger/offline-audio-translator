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
    assert MODEL_DIRS["large-v3"] == "whisper-large-v3"


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


class FakeWhisper:
    def __init__(self, detected="az"):
        self.detected = detected
        self.calls = []

    def transcribe(self, path, language, model_size, on_progress=None):
        self.calls.append(("transcribe", language))
        return [], "az"

    def detect_language(self, path, model_size=None):
        self.calls.append(("detect", model_size))
        return self.detected


class FakeVosk:
    def __init__(self):
        self.calls = []

    def transcribe(self, path, language, on_progress=None):
        self.calls.append(("vosk", language))
        return [Segment(start=0.0, end=1.0, text="салом")], language


def test_hybrid_routes_uz_to_vosk(tmp_path):
    from app.asr import HybridTranscriber
    w, v = FakeWhisper(), FakeVosk()
    h = HybridTranscriber(tmp_path, whisper=w, vosk=v)
    segs, det = h.transcribe("x.wav", "uz", "large-v3")
    assert det == "uz" and v.calls == [("vosk", "uz")] and w.calls == []
    assert segs[0].text == "салом"


def test_hybrid_routes_az_to_whisper(tmp_path):
    from app.asr import HybridTranscriber
    w, v = FakeWhisper(), FakeVosk()
    h = HybridTranscriber(tmp_path, whisper=w, vosk=v)
    h.transcribe("x.wav", "az", "large-v3")
    assert w.calls == [("transcribe", "az")] and v.calls == []


def test_hybrid_auto_detects_then_routes(tmp_path):
    from app.asr import HybridTranscriber
    w, v = FakeWhisper(detected="tg"), FakeVosk()
    h = HybridTranscriber(tmp_path, whisper=w, vosk=v)
    segs, det = h.transcribe("x.wav", "auto", "large-v3")
    assert w.calls == [("detect", "large-v3")]
    assert v.calls == [("vosk", "tg")] and det == "tg"
    assert segs[0].start == 0.0
