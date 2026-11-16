"""HTTP-сервис: загрузка аудио, статусы задач, выдача результатов."""

import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .asr import Transcriber
from .config import models_dir, output_dir, web_dir, work_dir
from .jobs import JobManager
from .translate import NllbTranslator

MAX_UPLOAD = 2 * 1024 ** 3  # 2 ГБ
RESULT_FILES = {"transcript.txt", "translation.txt", "result.docx"}


def build_manager() -> JobManager:
    transcriber = Transcriber(models_dir())
    translator = NllbTranslator(models_dir() / "nllb-600m-ct2")
    return JobManager(work_dir(), output_dir(), transcriber.transcribe, translator.translate_texts)


def create_app(manager: JobManager | None = None) -> FastAPI:
    mgr = manager or build_manager()

    def job_dto(job):
        return {"id": job.id, "filename": job.filename, "language": job.language,
                "model_size": job.model_size, "stage": job.stage,
                "progress": job.progress, "error": job.error}

    app = FastAPI(title="Офлайн-расшифровка", docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=str(web_dir()), check_dir=False), name="static")

    @app.get("/")
    def index():
        return FileResponse(web_dir() / "index.html")

    @app.get("/api/health")
    def health():
        return {"ok": True}

    @app.post("/api/jobs")
    def create_job(file: UploadFile = File(...),
                   language: str = Form("auto"),
                   model_size: str = Form("turbo")):
        if language not in ("uz", "az", "tg", "auto"):
            raise HTTPException(400, "язык должен быть uz, az, tg или auto")
        if model_size not in ("turbo", "medium", "small"):
            raise HTTPException(400, "model_size должен быть turbo, medium или small")
        job_id = uuid.uuid4().hex[:12]
        tmp = work_dir() / f"upload-{job_id}"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        total = 0
        try:
            with tmp.open("wb") as out:
                while chunk := file.file.read(4 * 1024 * 1024):
                    total += len(chunk)
                    if total > MAX_UPLOAD:
                        raise HTTPException(413, "файл больше 2 ГБ")
                    out.write(chunk)
            job = mgr.submit(tmp, file.filename or "audio", language, model_size)
        finally:
            tmp.unlink(missing_ok=True)
        return {"id": job.id}

    @app.get("/api/jobs")
    def list_jobs():
        return [job_dto(j) for j in reversed(mgr.all())]

    @app.get("/api/jobs/{job_id}")
    def job_detail(job_id: str):
        job = mgr.get(job_id)
        if job is None:
            raise HTTPException(404, "задача не найдена")
        return job_dto(job)

    @app.get("/api/jobs/{job_id}/files/{name}")
    def job_file(job_id: str, name: str):
        if name not in RESULT_FILES:
            raise HTTPException(404, "файл не найден")
        job = mgr.get(job_id)
        if job is None or not job.output_dir:
            raise HTTPException(404, "задача не найдена")
        path = Path(job.output_dir) / name
        if not path.exists():
            raise HTTPException(404, "файл ещё не готов")
        return FileResponse(str(path), filename=name)

    return app


app = create_app()
