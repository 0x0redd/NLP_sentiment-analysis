from __future__ import annotations

import threading
import traceback
from pathlib import Path

from django.conf import settings
from django.utils import timezone

from .models import ExperimentRun
from .pipeline.config import ExperimentOptions
from .pipeline.runner import run_experiment

_lock = threading.Lock()
_worker: threading.Thread | None = None


def _artifact_dir(run: ExperimentRun) -> Path:
    path = Path(settings.ARTIFACTS_DIR) / str(run.id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _execute(run_id) -> None:
    from django.db import close_old_connections

    close_old_connections()
    try:
        run = ExperimentRun.objects.get(pk=run_id)
        run.status = ExperimentRun.STATUS_RUNNING
        run.started_at = timezone.now()
        run.progress = "Starting…"
        run.save(update_fields=["status", "started_at", "progress"])

        def progress(msg: str) -> None:
            ExperimentRun.objects.filter(pk=run_id).update(progress=msg)

        opts = ExperimentOptions.from_dict(run.options)
        payload = run_experiment(
            run_id=str(run_id),
            artifact_dir=_artifact_dir(run),
            options=opts,
            progress=progress,
        )
        run = ExperimentRun.objects.get(pk=run_id)
        run.status = ExperimentRun.STATUS_SUCCEEDED
        run.summary = {
            "models": payload.get("models", []),
            "sizes": payload.get("sizes", {}),
            "timings": payload.get("timings", {}),
            "n_results": len(payload.get("results", [])),
        }
        run.progress = "Done"
        run.finished_at = timezone.now()
        run.save(update_fields=["status", "summary", "progress", "finished_at"])
    except Exception as exc:
        ExperimentRun.objects.filter(pk=run_id).update(
            status=ExperimentRun.STATUS_FAILED,
            error=f"{exc}\n{traceback.format_exc()}",
            progress="Failed",
            finished_at=timezone.now(),
        )
    finally:
        close_old_connections()
        global _worker
        with _lock:
            _worker = None


def start_run(run: ExperimentRun) -> bool:
    """Start background training if no other run is active. Returns False if busy."""
    global _worker
    with _lock:
        active = ExperimentRun.objects.filter(
            status__in=[ExperimentRun.STATUS_QUEUED, ExperimentRun.STATUS_RUNNING]
        ).exclude(pk=run.pk).exists()
        if active or (_worker is not None and _worker.is_alive()):
            return False
        _worker = threading.Thread(target=_execute, args=(run.id,), daemon=True)
        _worker.start()
        return True


def is_training() -> bool:
    with _lock:
        if _worker is not None and _worker.is_alive():
            return True
    return ExperimentRun.objects.filter(
        status__in=[ExperimentRun.STATUS_QUEUED, ExperimentRun.STATUS_RUNNING]
    ).exists()
