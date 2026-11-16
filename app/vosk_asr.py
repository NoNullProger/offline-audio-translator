"""ASR для узбекского и таджикского: ONNX-модели Vosk через sherpa-onnx.

Модели alphacep с Hugging Face:
- uz: alphacep/vosk-model-small-streaming-uz (стриминговый zipformer2);
- tg: alphacep/vosk-model-tajik (non-streaming zipformer2).

Бенчмарк FLEURS: uz WER 22.5%, tg WER 8.1% (Whisper large-v3: 92% и 76%).
"""

from pathlib import Path

import av
import numpy as np
import sherpa_onnx
from faster_whisper.vad import VadOptions, get_speech_timestamps

from .asr import Segment

# uz-модель стриминговая (сигнатура без x_lens, с cached_* состояниями),
# tg — non-streaming (x, x_lens)
_STREAMING = {"uz"}
_CHUNK_SEC = 30          # максимум длины сегмента
_SR = 16000


def load_pcm16k(path: Path) -> np.ndarray:
    """Декодирует любой аудио-файл в mono int16 16 кГц (PyAV)."""
    resampler = av.AudioResampler(format="s16", layout="mono", rate=_SR)
    chunks = []
    try:
        with av.open(str(path)) as container:
            if not container.streams.audio:
                raise ValueError("в файле нет аудиодорожки")
            for frame in container.decode(audio=0):
                for rf in resampler.resample(frame):
                    chunks.append(rf.to_ndarray().reshape(-1))
    except av.FFmpegError as e:
        raise ValueError(f"повреждённый или неподдерживаемый файл: {e}") from e
    if not chunks:
        raise ValueError("в файле нет аудиоданных")
    return np.concatenate(chunks).astype("int16")


def _windows(speech: list, max_samples: int) -> list:
    """Объединяет соседние речевые куски VAD в окна длиной до max_samples."""
    merged = []
    for s in speech:
        if merged and s["end"] - merged[-1][0] <= max_samples:
            merged[-1][1] = s["end"]
        else:
            merged.append([s["start"], s["end"]])
    # слишком длинные окна режем пополам по середине
    out = []
    for start, end in merged:
        while end - start > max_samples:
            mid = (start + end) // 2
            out.append([start, mid])
            start = mid
        out.append([start, end])
    return out


class VoskTranscriber:
    LANGS = ("uz", "tg")

    def __init__(self, models_root: Path):
        self._root = Path(models_root)
        self._cache: dict[str, object] = {}

    def _model(self, lang: str):
        if lang not in self.LANGS:
            raise ValueError(f"Vosk-модель недоступна для языка: {lang}")
        if lang not in self._cache:
            model_dir = self._root / lang
            enc = model_dir / "am-onnx" / "encoder.onnx"
            if not enc.exists():
                raise RuntimeError(f"модель не найдена: {enc}")
            kwargs = dict(
                encoder=str(enc),
                decoder=str(model_dir / "am-onnx" / "decoder.onnx"),
                joiner=str(model_dir / "am-onnx" / "joiner.onnx"),
                tokens=str(model_dir / "lang" / "tokens.txt"),
                sample_rate=_SR,
                feature_dim=80,
            )
            if lang in _STREAMING:
                self._cache[lang] = sherpa_onnx.OnlineRecognizer.from_transducer(**kwargs)
            else:
                self._cache[lang] = sherpa_onnx.OfflineRecognizer.from_transducer(**kwargs)
        return self._cache[lang]

    def transcribe(self, audio_path, language: str, on_progress=None):
        """Возвращает (сегменты, язык). Декодирует pcm один раз."""
        pcm = load_pcm16k(Path(audio_path))
        return self.transcribe_pcm(pcm, language, on_progress)

    def transcribe_pcm(self, pcm: np.ndarray, language: str, on_progress=None):
        rec = self._model(language)
        speech = get_speech_timestamps(
            pcm.astype("float32") / 32768.0,
            vad_options=VadOptions(min_silence_duration_ms=500),
            sampling_rate=_SR,
        )
        windows = _windows(speech, _CHUNK_SEC * _SR) or [[0, len(pcm)]]
        total = max(1, len(pcm))
        segments: list[Segment] = []
        for start, end in windows:
            piece = pcm[start:end]
            if not len(piece):
                continue
            s = rec.create_stream()
            s.accept_waveform(_SR, piece.astype("float32") / 32768.0)
            if isinstance(rec, sherpa_onnx.OnlineRecognizer):
                s.input_finished()
                while rec.is_ready(s):
                    rec.decode_stream(s)
                text = rec.get_result(s).strip()
            else:
                rec.decode_stream(s)
                text = s.result.text.strip()
            if text:
                segments.append(Segment(start=start / _SR, end=end / _SR, text=text))
            if on_progress:
                on_progress(min(1.0, end / total))
        return segments, language
