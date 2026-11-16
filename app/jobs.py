"""Очередь задач: строго последовательная обработка в фоновом потоке."""

import json
import queue
import shutil
import threading
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from .asr import Segment
from .outputs import write_docx, write_translation, write_transcript


@dataclass
class Job:
    id: str
    filename: str
    language: str
    model_size: str
    stage: str = "queued"
    progress: float = 0.0
    error: str | None = None
    output_dir: str | None = None


class JobManager:
    def __init__(self, work_root: Path, output_root: Path, transcribe, translate):
        self.work_root = Path(work_root)
        self.output_root = Path(output_root)
        self._transcribe = transcribe
        self._translate = translate
        self._queue: queue.Queue[str] = queue.Queue()
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self.work_root.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)
        threading.Thread(target=self._worker, daemon=True).start()

    def submit(self, src_path: Path, orig_name: str, language: str, model_size: str) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], filename=orig_name,
                  language=language, model_size=model_size)
        job_dir = self.work_root / job.id
        job_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(orig_name).suffix or ".audio"
        dst = job_dir / f"input{suffix}"
        shutil.copyfile(src_path, dst)   # стрим-копия: не держим файл в RAM
        with self._lock:
            self._jobs[job.id] = job
        self._persist(job)
        self._queue.put(job.id)
        return job

    def get(self, job_id: str):
        with self._lock:
            return self._jobs.get(job_id)

    def all(self):
        with self._lock:
            return list(self._jobs.values())

    def _worker(self) -> None:
        while True:
            job_id = self._queue.get()
            job = self.get(job_id)
            if job is not None:
                self._run(job)

    def _set(self, job: Job, **fields) -> None:
        with self._lock:
            for k, v in fields.items():
                setattr(job, k, v)
            snapshot = asdict(job)
        self._persist_snapshot(job.id, snapshot)

    def _persist(self, job: Job) -> None:
        self._persist_snapshot(job.id, asdict(job))

    def _persist_snapshot(self, job_id: str, snapshot: dict) -> None:
        path = self.work_root / job_id / "status.json"
        path.write_text(json.dumps(snapshot, ensure_ascii=False), encoding="utf-8")

    def _run(self, job: Job) -> None:
        try:
            job_dir = self.work_root / job.id
            src = next(p for p in job_dir.iterdir() if p.name.startswith("input"))
            self._set(job, stage="asr", progress=0.0)

            def on_progress(p: float) -> None:
                self._set(job, progress=round(p, 4))

            segments, detected = self._transcribe(src, job.language, job.model_size, on_progress)
            self._set(job, stage="translate", progress=0.0)
            texts = [s.text for s in segments]
            translations = self._translate(texts, detected) if texts else []
            self._set(job, stage="write")
            out_dir = self.output_root / job.id
            out_dir.mkdir(parents=True, exist_ok=True)
            write_transcript(out_dir / "transcript.txt", segments)
            write_translation(out_dir / "translation.txt", segments, translations)
            write_docx(out_dir / "result.docx", job.filename, segments, translations)
            self._set(job, stage="done", output_dir=str(out_dir))
        except Exception as e:  # любая ошибка не валит сервис
            self._set(job, stage="error", error=f"{type(e).__name__}: {e}")
