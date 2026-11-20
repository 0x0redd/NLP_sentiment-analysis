from __future__ import annotations

import json
from pathlib import Path

from django.conf import settings
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .jobs import is_training, start_run
from .models import ExperimentRun
from .pipeline.preprocess import preprocess_article, preprocess_negation
from .pipeline.registry import registry
from .pipeline.samples import random_sample
from .pipeline.train_bert import get_device
from .serializers import CreateRunSerializer, ExperimentRunSerializer, PredictSerializer, PreprocessSerializer


def _artifact_dir(run: ExperimentRun) -> Path:
    return Path(settings.ARTIFACTS_DIR) / str(run.id)


def _load_results_json(run: ExperimentRun) -> dict | None:
    path = _artifact_dir(run) / "results.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@api_view(["GET"])
def health(request):
    last = ExperimentRun.objects.filter(status=ExperimentRun.STATUS_SUCCEEDED).first()
    return Response(
        {
            "ok": True,
            "device": get_device(),
            "training": is_training(),
            "loaded_models": registry.loaded_models,
            "active_run_id": registry.active_run_id,
            "last_succeeded_run_id": str(last.id) if last else None,
        }
    )


@api_view(["GET", "POST"])
def runs(request):
    if request.method == "GET":
        qs = ExperimentRun.objects.all()[:50]
        return Response(ExperimentRunSerializer(qs, many=True).data)

    ser = CreateRunSerializer(data=request.data)
    ser.is_valid(raise_exception=True)
    if is_training():
        return Response(
            {"detail": "Another experiment is already queued or running."},
            status=status.HTTP_409_CONFLICT,
        )
    run = ExperimentRun.objects.create(options=ser.validated_data, status=ExperimentRun.STATUS_QUEUED)
    if not start_run(run):
        run.status = ExperimentRun.STATUS_FAILED
        run.error = "Could not start worker (busy)."
        run.save(update_fields=["status", "error"])
        return Response({"detail": "Trainer busy."}, status=status.HTTP_409_CONFLICT)
    return Response(ExperimentRunSerializer(run).data, status=status.HTTP_201_CREATED)


@api_view(["GET"])
def run_detail(request, run_id):
    try:
        run = ExperimentRun.objects.get(pk=run_id)
    except ExperimentRun.DoesNotExist:
        return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)
    return Response(ExperimentRunSerializer(run).data)


def _run_section(run_id, key: str):
    try:
        run = ExperimentRun.objects.get(pk=run_id)
    except ExperimentRun.DoesNotExist:
        return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)
    if run.status != ExperimentRun.STATUS_SUCCEEDED:
        return Response(
            {"detail": f"Run is {run.status}; results unavailable."},
            status=status.HTTP_409_CONFLICT,
        )
    payload = _load_results_json(run)
    if payload is None:
        return Response({"detail": "results.json missing."}, status=status.HTTP_404_NOT_FOUND)
    return Response({"run_id": str(run.id), key: payload.get(key, []) if key != "confusion" else payload.get(key, {})})


@api_view(["GET"])
def run_results(request, run_id):
    return _run_section(run_id, "results")


@api_view(["GET"])
def run_mcnemar(request, run_id):
    return _run_section(run_id, "mcnemar")


@api_view(["GET"])
def run_ablation(request, run_id):
    return _run_section(run_id, "ablation")


@api_view(["GET"])
def run_confusion(request, run_id):
    return _run_section(run_id, "confusion")


@api_view(["GET"])
def run_timings(request, run_id):
    return _run_section(run_id, "timings")


@api_view(["GET"])
def run_controlled(request, run_id):
    return _run_section(run_id, "controlled")


@api_view(["POST"])
def run_activate(request, run_id):
    try:
        run = ExperimentRun.objects.get(pk=run_id)
    except ExperimentRun.DoesNotExist:
        return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)
    if run.status != ExperimentRun.STATUS_SUCCEEDED:
        return Response(
            {"detail": f"Run is {run.status}; cannot activate."},
            status=status.HTTP_409_CONFLICT,
        )
    loaded = registry.load_from_dir(str(run.id), _artifact_dir(run))
    return Response({"active_run_id": str(run.id), "loaded_models": loaded})


@api_view(["POST"])
def predict(request):
    ser = PredictSerializer(data=request.data)
    ser.is_valid(raise_exception=True)
    if not registry.loaded_models:
        return Response(
            {"detail": "No models loaded. Activate a succeeded run or wait for bootstrap."},
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    data = ser.validated_data
    result = registry.predict(
        text=data["text"],
        models=data.get("models") or None,
        preprocess=data.get("preprocess", "article"),
    )
    return Response(result)


@api_view(["POST"])
def preprocess_view(request):
    ser = PreprocessSerializer(data=request.data)
    ser.is_valid(raise_exception=True)
    text = ser.validated_data["text"]
    return Response(
        {
            "text": text,
            "article": preprocess_article(text),
            "negation": preprocess_negation(text),
        }
    )


@api_view(["GET"])
def sample_text(request):
    """Return a random sentence/review from SST-2 or IMDB cache for the inference UI."""
    dataset = request.query_params.get("dataset") or None
    if dataset not in (None, "SST-2", "IMDB", "any"):
        return Response(
            {"detail": "dataset must be SST-2, IMDB, or any."},
            status=status.HTTP_400_BAD_REQUEST,
        )
    if dataset == "any":
        dataset = None
    return Response(random_sample(dataset))
