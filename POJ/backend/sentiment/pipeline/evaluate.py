from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def metrics(y, pred) -> dict[str, float]:
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="weighted", zero_division=0)
    return {
        "Accuracy": float(accuracy_score(y, pred)),
        "Precision": float(p),
        "Recall": float(r),
        "F1-score": float(f),
    }


def results_table(names: list[str], y_by_set: dict, preds: dict) -> list[dict]:
    rows = []
    for name in names:
        for ds, y in y_by_set.items():
            rows.append({"Model": name, "Dataset": ds, **metrics(y, preds[(name, ds)])})
    return rows


def confusion_payload(names: list[str], y_by_set: dict, preds: dict) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for ds, y in y_by_set.items():
        out[ds] = {}
        for name in names:
            cm = confusion_matrix(y, preds[(name, ds)], labels=[0, 1])
            out[ds][name] = cm.tolist()
    return out


def to_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj
