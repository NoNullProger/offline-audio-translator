from docx import Document
from app.outputs import fmt_time, write_transcript, write_translation, write_docx


class Seg:
    def __init__(self, start, text):
        self.start, self.end, self.text = start, start + 2.0, text


def test_fmt_time():
    assert fmt_time(67.42) == "01:07.42"
    assert fmt_time(0) == "00:00.00"


def test_transcript_txt(tmp_path):
    p = tmp_path / "t.txt"
    write_transcript(p, [Seg(0.0, "Салом"), Seg(5.0, "Хуш")])
    assert p.read_text(encoding="utf-8") == "[00:00.00] Салом\n[00:05.00] Хуш\n"


def test_translation_missing_marker(tmp_path):
    p = tmp_path / "r.txt"
    write_translation(p, [Seg(0.0, "Салом")], [None])
    assert "[перевод недоступен]" in p.read_text(encoding="utf-8")


def test_docx(tmp_path):
    p = tmp_path / "r.docx"
    write_docx(p, "запись.wav", [Seg(0.0, "Салом")], ["Привет"])
    d = Document(str(p))
    text = "\n".join(par.text for par in d.paragraphs)
    assert "запись.wav" in text and "Привет" in text and "[00:00.00] Салом" in text
