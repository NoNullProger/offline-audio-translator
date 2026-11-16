import json
import time
from pathlib import Path

from app.asr import Segment
from app.jobs import JobManager


def fake_transcribe(path, language, model_size, on_progress=None):
    if on_progress:
        on_progress(1.0)
    return [Segment(start=0.0, end=2.0, text="Салом")], "uz"


def fake_translate(texts, lang):
    return ["Привет" for _ in texts]


def wait_stage(mgr, job_id, stage, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        j = mgr.get(job_id)
        if j and j.stage == stage:
            return j
        time.sleep(0.05)
    raise AssertionError(f"stage={stage} не достигнута, последний: {mgr.get(job_id)}")


def make_mgr(tmp_path, transcribe=fake_transcribe, translate=fake_translate):
    return JobManager(tmp_path / "work", tmp_path / "output", transcribe, translate)


def submit(mgr, tmp_path):
    src = tmp_path / "in.wav"
    src.write_bytes(b"fake")
    return mgr.submit(src, "запись.wav", "uz", "turbo")


def test_pipeline_happy_path(tmp_path):
    mgr = make_mgr(tmp_path)
    job = submit(mgr, tmp_path)
    j = wait_stage(mgr, job.id, "done")
    out = Path(j.output_dir)
    assert (out / "transcript.txt").read_text(encoding="utf-8").strip().endswith("Салом")
    assert "Привет" in (out / "translation.txt").read_text(encoding="utf-8")
    assert (out / "result.docx").exists()
    status = json.loads((tmp_path / "work" / job.id / "status.json").read_text())
    assert status["stage"] == "done"


def test_error_is_reported(tmp_path):
    def boom(path, language, model_size, on_progress=None):
        raise ValueError("плохой файл")

    mgr = make_mgr(tmp_path, transcribe=boom)
    job = submit(mgr, tmp_path)
    j = wait_stage(mgr, job.id, "error")
    assert "плохой файл" in j.error


def test_translate_failure_marks_unavailable(tmp_path):
    def fail_translate(texts, lang):
        return [None for _ in texts]

    mgr = make_mgr(tmp_path, translate=fail_translate)
    job = submit(mgr, tmp_path)
    j = wait_stage(mgr, job.id, "done")
    assert "[перевод недоступен]" in (Path(j.output_dir) / "translation.txt").read_text(encoding="utf-8")
