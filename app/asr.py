"""Расшифровка речи: гибрид — Whisper (az) и Vosk ONNX (uz/tg) через sherpa-onnx."""

import os
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
from faster_whisper import WhisperModel

MODEL_DIRS = {
    "large-v3": "whisper-large-v3",
    "medium": "whisper-medium",
    "small": "whisper-small",
}


@dataclass
class Segment:
    start: float
    end: float
    text: str


def validate_audio(path: Path) -> None:
    """Бросает ValueError, если файл не аудио или без аудиодорожки."""
    try:
        with av.open(str(path)) as container:
            if not container.streams.audio:
                raise ValueError("в файле нет аудиодорожки")
    except av.FFmpegError as e:
        raise ValueError(f"повреждённый или неподдерживаемый файл: {e}") from e


class Transcriber:
    def __init__(self, models_root: Path):
        self._root = Path(models_root)
        self._cache: dict[str, WhisperModel] = {}

    def _model(self, model_size: str) -> WhisperModel:
        if model_size not in MODEL_DIRS:
            raise ValueError(f"неизвестный размер модели: {model_size}")
        if model_size not in self._cache:
            path = self._root / MODEL_DIRS[model_size]
            if not path.exists():
                raise RuntimeError(f"модель не найдена: {path}")
            self._cache[model_size] = WhisperModel(
                str(path),
                device="cpu",
                compute_type="int8",
                cpu_threads=min(4, os.cpu_count() or 4),
            )
        return self._cache[model_size]

    def transcribe(self, audio_path, language, model_size, on_progress=None):
        """Возвращает (сегменты, язык|None). language: uz/az/tg/auto/None."""
        validate_audio(Path(audio_path))
        model = self._model(model_size)
        lang = None if language in (None, "auto") else language
        segments_iter, info = model.transcribe(
            str(audio_path),
            language=lang,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        segments: list[Segment] = []
        duration = info.duration or 0.0
        for s in segments_iter:
            segments.append(Segment(start=s.start, end=s.end, text=(s.text or "").strip()))
            if on_progress and duration > 0:
                on_progress(min(1.0, s.end / duration))
        detected = getattr(info, "language", None) if lang is None else lang
        return segments, detected

    def detect_language(self, audio_path, model_size=None):
        """Определяет язык по первым ~30 секундам (для маршрутизации auto)."""
        model = self._model(model_size or "large-v3")
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        samples: list[np.ndarray] = []
        total = 0
        with av.open(str(audio_path)) as container:
            for frame in container.decode(audio=0):
                for rf in resampler.resample(frame):
                    arr = rf.to_ndarray().reshape(-1)
                    samples.append(arr)
                    total += len(arr)
                if total >= 30 * 16000:
                    break
        if not samples:
            raise ValueError("в файле нет аудиоданных")
        pcm = np.concatenate(samples)[:30 * 16000].astype("float32") / 32768.0
        _, info = model.transcribe(pcm, language=None, vad_filter=False)
        return getattr(info, "language", None)


class HybridTranscriber:
    """Маршрутизация по языку: uz/tg → Vosk ONNX, az → Whisper (docs/benchmark.md)."""

    def __init__(self, models_root: Path, whisper=None, vosk=None):
        self._whisper = whisper or Transcriber(models_root)
        if vosk is None:
            from .vosk_asr import VoskTranscriber  # лениво, чтобы не зациклить импорт
            vosk = VoskTranscriber(Path(models_root) / "vosk-onnx")
        self._vosk = vosk

    def transcribe(self, audio_path, language, model_size, on_progress=None):
        if language in ("uz", "tg"):
            return self._vosk.transcribe(audio_path, language, on_progress)
        if language in (None, "auto"):
            detected = self._whisper.detect_language(audio_path, model_size)
            if detected in ("uz", "tg"):
                return self._vosk.transcribe(audio_path, detected, on_progress)
        return self._whisper.transcribe(audio_path, language, model_size, on_progress)
