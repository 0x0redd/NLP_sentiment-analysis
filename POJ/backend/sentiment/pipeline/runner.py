from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import set_seed

from .config import ExperimentOptions, SEED
from .controlled import build_controlled_table
from .data import load_splits
from .evaluate import confusion_payload, results_table, to_serializable
from .mcnemar import run_mcnemar
from .preprocess import preprocess_article, preprocess_negation
from .train_bert import bert_positive_proba, train_distilbert
from .train_sklearn import (
    run_feature_ablation,
    train_article_models,
    train_mlp,
    train_negation_lr,
)


def _set_seeds() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    try:
        set_seed(SEED)
    except Exception:
        pass


def run_experiment(
    run_id: str,
    artifact_dir: Path,
    options: ExperimentOptions | dict | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Full merged-notebook pipeline. Writes artifacts under artifact_dir."""
    opts = (
        options
        if isinstance(options, ExperimentOptions)
        else ExperimentOptions.from_dict(options)
    )
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    _set_seeds()

    def log(msg: str) -> None:
        if progress:
            progress(msg)

    log("Loading SST-2 / IMDB splits...")
    data = load_splits(opts)
    train_df = data["train_df"]
    test_df = data["test_df"]
    sst_val = data["sst_val"]
    imdb = data["imdb"]

    log("Fitting main TF-IDF...")
    tfidf = TfidfVectorizer()
    X_train = tfidf.fit_transform(train_df["clean"])
    y_train = train_df["label"].astype(int).values
    X_by_set = {
        "SST-2": tfidf.transform(test_df["clean"]),
        "SST-2 val.": tfidf.transform(sst_val["clean"]),
        "IMDB": tfidf.transform(imdb["clean"]),
    }
    Y = {
        "SST-2": test_df["label"].astype(int).values,
        "SST-2 val.": sst_val["label"].astype(int).values,
        "IMDB": imdb["label"].astype(int).values,
    }
    RAW = {
        "SST-2": test_df["text"].tolist(),
        "SST-2 val.": sst_val["text"].tolist(),
        "IMDB": imdb["text"].str.replace(r"<br\s*/?>", " ", regex=True).tolist(),
    }

    log("Training article models (KNN / RF / NB / LR)...")
    fitted, preds, timings = train_article_models(X_train, y_train, X_by_set)
    article_models = list(fitted.keys())

    ablation_rows: list[dict] = []
    if opts.run_ablation:
        log("Running feature-representation ablation...")
        test_bundles = {
            ds: {
                "clean": df["clean"].tolist(),
                "text_lower": (
                    df["text"].str.lower().str.replace(r"<br\s*/?>", " ", regex=True).tolist()
                    if ds == "IMDB"
                    else df["text"].str.lower().tolist()
                ),
            }
            for ds, df in [("SST-2", test_df), ("SST-2 val.", sst_val), ("IMDB", imdb)]
        }
        ablation_rows = run_feature_ablation(train_df, test_bundles, Y, preds)

    lr_neg = None
    tfidf_neg = None
    negation_rows: list[dict] = []
    if opts.run_negation:
        log("Training negation-aware Logistic Regression...")
        test_dfs = {"SST-2": test_df, "SST-2 val.": sst_val, "IMDB": imdb}
        lr_neg, tfidf_neg, neg_preds = train_negation_lr(train_df, test_dfs)
        for ds, y in Y.items():
            from .evaluate import metrics as _metrics

            negation_rows.append(
                {"Preprocess": "article (drop not)", "Dataset": ds, **_metrics(y, preds[("Logistic Regression", ds)])}
            )
            negation_rows.append(
                {"Preprocess": "keep negations", "Dataset": ds, **_metrics(y, neg_preds[ds])}
            )

    log("Training MLP on TF-IDF...")
    mlp, mlp_preds, mlp_time = train_mlp(X_train, y_train, X_by_set)
    fitted["MLP (TF-IDF)"] = mlp
    timings["MLP (TF-IDF)"] = mlp_time
    for ds, pred in mlp_preds.items():
        preds[("MLP (TF-IDF)", ds)] = pred

    bert_model = None
    bert_tokenizer = None
    if opts.run_distilbert:
        log("Fine-tuning DistilBERT...")
        bert_model, bert_tokenizer, bert_preds, bert_time = train_distilbert(
            train_df["text"].tolist(),
            train_df["label"].astype(int).tolist(),
            RAW,
            artifact_dir,
            progress=log,
            fast_mode=opts.fast_mode,
        )
        timings["DistilBERT"] = bert_time
        for ds, pred in bert_preds.items():
            preds[("DistilBERT", ds)] = pred

    all_models = article_models + ["MLP (TF-IDF)"]
    if ("DistilBERT", "SST-2") in preds:
        all_models.append("DistilBERT")

    log("Building evaluation tables...")
    results = results_table(all_models, Y, preds)
    confusion = confusion_payload(all_models, Y, preds)

    pairs = [("MLP (TF-IDF)", "Logistic Regression")]
    if ("DistilBERT", "SST-2") in preds:
        pairs += [
            ("DistilBERT", "Logistic Regression"),
            ("DistilBERT", "MLP (TF-IDF)"),
        ]
    mcnemar_rows = run_mcnemar(Y, preds, pairs)

    bert_proba_fn = None
    if bert_model is not None and bert_tokenizer is not None:
        bert_proba_fn = lambda texts: bert_positive_proba(bert_model, bert_tokenizer, texts)

    controlled = build_controlled_table(
        lr=fitted["Logistic Regression"],
        mlp=fitted["MLP (TF-IDF)"],
        tfidf=tfidf,
        preprocess_article=preprocess_article,
        preprocess_negation=preprocess_negation,
        lr_neg=lr_neg,
        tfidf_neg=tfidf_neg,
        bert_proba_fn=bert_proba_fn,
    )

    log("Saving artifacts...")
    joblib.dump(tfidf, artifact_dir / "tfidf_article.joblib")
    joblib.dump(fitted["KNN"], artifact_dir / "knn.joblib")
    joblib.dump(fitted["Random Forest"], artifact_dir / "random_forest.joblib")
    joblib.dump(fitted["Naive Bayes"], artifact_dir / "naive_bayes.joblib")
    joblib.dump(fitted["Logistic Regression"], artifact_dir / "lr_article.joblib")
    joblib.dump(fitted["MLP (TF-IDF)"], artifact_dir / "mlp_tfidf.joblib")
    if lr_neg is not None and tfidf_neg is not None:
        joblib.dump(lr_neg, artifact_dir / "lr_neg.joblib")
        joblib.dump(tfidf_neg, artifact_dir / "tfidf_neg.joblib")

    payload = {
        "run_id": run_id,
        "options": opts.to_dict(),
        "sizes": data["sizes"],
        "models": all_models,
        "results": results,
        "mcnemar": mcnemar_rows,
        "ablation": ablation_rows,
        "negation": negation_rows,
        "confusion": confusion,
        "timings": {k: float(v) for k, v in timings.items()},
        "controlled": controlled,
    }
    payload = to_serializable(payload)

    (artifact_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(results).to_csv(artifact_dir / "results.csv", index=False)
    if mcnemar_rows:
        pd.DataFrame(mcnemar_rows).to_csv(artifact_dir / "mcnemar.csv", index=False)
    if ablation_rows:
        pd.DataFrame(ablation_rows).to_csv(artifact_dir / "ablation.csv", index=False)

    log("Experiment complete.")
    return payload
