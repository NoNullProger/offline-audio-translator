"""Расшифровка речи: faster-whisper на CPU, int8, VAD."""

import os
from dataclasses import dataclass
from pathlib import Path

import av
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
