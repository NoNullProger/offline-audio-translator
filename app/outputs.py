"""Запись результатов: расшифровка, перевод, docx."""

from pathlib import Path

from docx import Document


def fmt_time(sec: float) -> str:
    m = int(sec // 60)
    s = sec - 60 * m
    return f"{m:02d}:{s:05.2f}"


def _lines(segments, translations=None):
    lines = []
    for i, seg in enumerate(segments):
        text = seg.text
        if translations is not None:
            t = translations[i] if i < len(translations) else None
            text = t if t else "[перевод недоступен]"
        lines.append(f"[{fmt_time(seg.start)}] {text}")
    return lines


def write_transcript(path: Path, segments) -> None:
    path.write_text("\n".join(_lines(segments)) + "\n", encoding="utf-8")


def write_translation(path: Path, segments, translations) -> None:
    path.write_text("\n".join(_lines(segments, translations)) + "\n", encoding="utf-8")


def write_docx(path: Path, title: str, segments, translations) -> None:
    doc = Document()
    doc.add_heading(f"Расшифровка и перевод: {title}", 0)
    doc.add_heading("Расшифровка (оригинал)", level=1)
    for line in _lines(segments):
        doc.add_paragraph(line)
    doc.add_heading("Перевод на русский", level=1)
    for line in _lines(segments, translations):
        doc.add_paragraph(line)
    doc.save(str(path))
