import time

from fastapi.testclient import TestClient

from app.asr import Segment
from app.jobs import JobManager
from app.main import create_app


def fake_transcribe(path, language, model_size, on_progress=None):
    return [Segment(start=0.0, end=1.0, text="Салом")], "uz"


def fake_translate(texts, lang):
    return ["Привет" for _ in texts]


def make_client(tmp_path):
    mgr = JobManager(tmp_path / "work", tmp_path / "output", fake_transcribe, fake_translate)
    return TestClient(create_app(mgr)), mgr


def wait_done(mgr, job_id, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        j = mgr.get(job_id)
        if j and j.stage == "done":
            return j
        time.sleep(0.05)
    raise AssertionError("задача не завершена")


def test_health(tmp_path):
    client, _ = make_client(tmp_path)
    assert client.get("/api/health").json() == {"ok": True}


def test_full_job_flow(tmp_path):
    client, mgr = make_client(tmp_path)
    r = client.post("/api/jobs", data={"language": "uz", "model_size": "turbo"},
                    files={"file": ("запись.wav", b"fake wav", "audio/wav")})
    assert r.status_code == 200
    job_id = r.json()["id"]
    wait_done(mgr, job_id)
    detail = client.get(f"/api/jobs/{job_id}").json()
    assert detail["stage"] == "done"
    for name in ("transcript.txt", "translation.txt", "result.docx"):
        assert client.get(f"/api/jobs/{job_id}/files/{name}").status_code == 200
    assert client.get(f"/api/jobs/{job_id}/files/..%2Fstatus.json").status_code in (400, 404)


def test_unknown_job_404(tmp_path):
    client, _ = make_client(tmp_path)
    assert client.get("/api/jobs/nonexistent").status_code == 404
    assert client.get("/api/jobs/nonexistent/files/transcript.txt").status_code == 404


def test_upload_limit(tmp_path, monkeypatch):
    import app.main as m
    monkeypatch.setattr(m, "MAX_UPLOAD", 8)
    client, _ = make_client(tmp_path)
    r = client.post("/api/jobs", data={"language": "uz", "model_size": "small"},
                    files={"file": ("big.wav", b"x" * 100, "audio/wav")})
    assert r.status_code == 413


def test_bad_language_rejected(tmp_path):
    client, _ = make_client(tmp_path)
    r = client.post("/api/jobs", data={"language": "de", "model_size": "small"},
                    files={"file": ("a.wav", b"x", "audio/wav")})
    assert r.status_code == 400
