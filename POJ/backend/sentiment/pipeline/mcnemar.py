from __future__ import annotations

import numpy as np
from scipy.stats import binomtest


def holm_correction(p_values) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adjusted = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (n - rank) * p[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted.tolist()


def mcnemar_exact(y, pred_1, pred_2) -> tuple[int, int, float]:
    ok1, ok2 = (pred_1 == y), (pred_2 == y)
    only1 = int((ok1 & ~ok2).sum())
    only2 = int((~ok1 & ok2).sum())
    n = only1 + only2
    p = float(binomtest(only1, n, 0.5).pvalue) if n > 0 else 1.0
    return only1, only2, p


def run_mcnemar(y_by_set: dict, preds: dict, pairs: list[tuple[str, str]]) -> list[dict]:
    rows = []
    for m1, m2 in pairs:
        for ds, y in y_by_set.items():
            if (m1, ds) not in preds or (m2, ds) not in preds:
                continue
            o1, o2, p = mcnemar_exact(y, preds[(m1, ds)], preds[(m2, ds)])
            rows.append(
                {
                    "Model 1": m1,
                    "Model 2": m2,
                    "Dataset": ds,
                    "Only M1 correct": o1,
                    "Only M2 correct": o2,
                    "p-value": p,
                }
            )
    if not rows:
        return []
    p_holm = holm_correction([r["p-value"] for r in rows])
    for r, ph in zip(rows, p_holm):
        r["p Holm"] = ph
        r["Significant (Holm 5%)"] = ph < 0.05
    return rows
