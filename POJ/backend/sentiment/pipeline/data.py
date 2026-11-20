from __future__ import annotations

from pathlib import Path

import pandas as pd
from datasets import load_dataset

from .config import ExperimentOptions, SEED
from .preprocess import preprocess_article, preprocess_negation

# Local parquet cache from the notebook folder (optional fallback)
_CACHE_DIR = Path(__file__).resolve().parents[4] / "cache"


def _load_sst2_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_pool, validation) with columns text, label."""
    try:
        sst = load_dataset("nyu-mll/glue", "sst2", split="train").to_pandas()
        sst_val = load_dataset("nyu-mll/glue", "sst2", split="validation").to_pandas()
        sst = sst.rename(columns={"sentence": "text"})
        sst_val = sst_val.rename(columns={"sentence": "text"})
        return sst, sst_val
    except Exception:
        pass
    try:
        sst = load_dataset("glue", "sst2", split="train").to_pandas()
        sst_val = load_dataset("glue", "sst2", split="validation").to_pandas()
        sst = sst.rename(columns={"sentence": "text"})
        sst_val = sst_val.rename(columns={"sentence": "text"})
        return sst, sst_val
    except Exception:
        pass

    train_path = _CACHE_DIR / "SST2_train_10000.parquet"
    test_path = _CACHE_DIR / "SST2_test_5000.parquet"
    if train_path.exists() and test_path.exists():
        sst = pd.read_parquet(train_path)
        sst_val = pd.read_parquet(test_path)
        if "sentence" in sst.columns:
            sst = sst.rename(columns={"sentence": "text"})
        if "sentence" in sst_val.columns:
            sst_val = sst_val.rename(columns={"sentence": "text"})
        return sst, sst_val
    raise RuntimeError(
        "Could not load SST-2 from Hugging Face (nyu-mll/glue) or local cache/."
    )


def _load_imdb_test() -> pd.DataFrame:
    try:
        return load_dataset("stanfordnlp/imdb", split="test").to_pandas()
    except Exception:
        pass
    try:
        return load_dataset("imdb", split="test").to_pandas()
    except Exception:
        pass
    path = _CACHE_DIR / "IMDB_test_5000.parquet"
    if path.exists():
        return pd.read_parquet(path)
    raise RuntimeError("Could not load IMDB test from Hugging Face or local cache/.")


def load_splits(options: ExperimentOptions) -> dict:
    sst, sst_val = _load_sst2_frames()
    sst = sst.sample(frac=1, random_state=SEED).reset_index(drop=True)

    n_needed = options.max_train + options.max_sst_test
    if len(sst) < n_needed:
        # Fall back: reuse available rows with replacement-free cap
        if len(sst) < options.max_train:
            raise ValueError(f"SST-2 pool has {len(sst)} rows, need at least {options.max_train}.")
        options_max_test = min(options.max_sst_test, len(sst) - options.max_train)
    else:
        options_max_test = options.max_sst_test

    train_df = sst.iloc[: options.max_train].copy().reset_index(drop=True)
    test_df = sst.iloc[options.max_train : options.max_train + options_max_test].copy().reset_index(drop=True)

    imdb = _load_imdb_test()
    imdb = imdb.sample(n=min(options.max_imdb_test, len(imdb)), random_state=SEED).reset_index(drop=True)

    for df in (train_df, test_df, sst_val, imdb):
        df["clean"] = df["text"].apply(preprocess_article)
        df["clean_neg"] = df["text"].apply(preprocess_negation)

    return {
        "train_df": train_df,
        "test_df": test_df,
        "sst_val": sst_val,
        "imdb": imdb,
        "sizes": {
            "train": len(train_df),
            "sst_test": len(test_df),
            "sst_val": len(sst_val),
            "imdb_test": len(imdb),
            "train_pos_rate": float(train_df["label"].mean()),
            "sst_test_pos_rate": float(test_df["label"].mean()),
            "imdb_pos_rate": float(imdb["label"].mean()),
        },
    }
