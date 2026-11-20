from __future__ import annotations

import json
import random
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

_CACHE_DIR = Path(__file__).resolve().parents[4] / "cache"
_SAMPLES_JSON = Path(__file__).resolve().parents[2] / "data" / "demo_samples.json"

_FALLBACK = [
    ("SST-2", "a warm , funny , engaging film .", 1),
    ("SST-2", "the movie is not worth the time .", 0),
    ("SST-2", "i am not happy with this film .", 0),
    ("SST-2", "an absolute masterpiece of cinema .", 1),
    ("SST-2", "dull , pointless , and poorly written .", 0),
    ("IMDB", "This film was hardly entertaining from start to finish.", 0),
    ("IMDB", "I really loved every minute of this movie.", 1),
    ("IMDB", "The plot made no sense and the acting was wooden.", 0),
    ("IMDB", "A surprisingly moving story with outstanding performances.", 1),
]


def _clean_imdb(text: str) -> str:
    text = re.sub(r"<br\s*/?>", " ", str(text), flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > 420:
        text = text[:417].rsplit(" ", 1)[0] + "…"
    return text


def _build_from_hf(n_sst: int = 400, n_imdb: int = 200) -> list[dict]:
    from datasets import load_dataset

    rows: list[dict] = []
    try:
        sst = load_dataset("nyu-mll/glue", "sst2", split="validation")
    except Exception:
        sst = load_dataset("glue", "sst2", split="validation")
    for ex in sst:
        rows.append(
            {
                "dataset": "SST-2",
                "text": str(ex["sentence"]).strip(),
                "gold_label": int(ex["label"]),
            }
        )
    # subsample SST train for more variety
    try:
        sst_tr = load_dataset("nyu-mll/glue", "sst2", split="train")
    except Exception:
        sst_tr = load_dataset("glue", "sst2", split="train")
    idx = list(range(len(sst_tr)))
    random.Random(42).shuffle(idx)
    for i in idx[:n_sst]:
        ex = sst_tr[i]
        rows.append(
            {
                "dataset": "SST-2",
                "text": str(ex["sentence"]).strip(),
                "gold_label": int(ex["label"]),
            }
        )

    try:
        imdb = load_dataset("stanfordnlp/imdb", split="test")
    except Exception:
        imdb = load_dataset("imdb", split="test")
    idx = list(range(len(imdb)))
    random.Random(42).shuffle(idx)
    for i in idx[:n_imdb]:
        ex = imdb[i]
        rows.append(
            {
                "dataset": "IMDB",
                "text": _clean_imdb(ex["text"]),
                "gold_label": int(ex["label"]),
            }
        )
    return [r for r in rows if len(r["text"]) >= 8]


def _build_from_parquet_clean() -> list[dict]:
    """Last resort: stemmed 'clean' column in local parquet (less readable)."""
    rows: list[dict] = []
    for dataset, path in [
        ("SST-2", _CACHE_DIR / "SST2_test_5000.parquet"),
        ("IMDB", _CACHE_DIR / "IMDB_test_5000.parquet"),
    ]:
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        col = "clean" if "clean" in df.columns else None
        if col is None:
            continue
        sample = df.sample(n=min(400, len(df)), random_state=42)
        for text in sample[col].astype(str):
            text = text.strip()
            if len(text) < 8:
                continue
            if dataset == "IMDB" and len(text) > 420:
                text = text[:417] + "…"
            rows.append({"dataset": dataset, "text": text})
    return rows


@lru_cache(maxsize=1)
def _load_pool() -> tuple[tuple[str, str, int | None], ...]:
    # 1) Local JSON cache of raw examples
    if _SAMPLES_JSON.exists():
        data = json.loads(_SAMPLES_JSON.read_text(encoding="utf-8"))
        out = []
        for r in data:
            out.append((r["dataset"], r["text"], r.get("gold_label")))
        if out:
            return tuple(out)

    # 2) Build from Hugging Face and persist
    try:
        built = _build_from_hf()
        _SAMPLES_JSON.parent.mkdir(parents=True, exist_ok=True)
        _SAMPLES_JSON.write_text(json.dumps(built, ensure_ascii=False, indent=0), encoding="utf-8")
        return tuple((r["dataset"], r["text"], r.get("gold_label")) for r in built)
    except Exception:
        pass

    # 3) Stemmed parquet / hardcoded fallback
    built = _build_from_parquet_clean()
    if built:
        return tuple((r["dataset"], r["text"], r.get("gold_label")) for r in built)
    return tuple((d, t, g) for d, t, g in _FALLBACK)


def random_sample(dataset: str | None = None) -> dict:
    pool = list(_load_pool())
    if dataset in ("SST-2", "IMDB"):
        filtered = [r for r in pool if r[0] == dataset]
        if filtered:
            pool = filtered
    ds, text, label = random.choice(pool)
    out: dict = {
        "dataset": ds,
        "text": text,
        "source": "demo_samples" if _SAMPLES_JSON.exists() else "live",
    }
    if label is not None:
        out["gold_label"] = int(label)
        out["gold_label_name"] = "positive" if int(label) == 1 else "negative"
    return out
