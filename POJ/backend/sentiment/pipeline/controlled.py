from __future__ import annotations

from typing import Callable

import numpy as np

CONTROLLED_EXAMPLES = [
    "I am happy",
    "I am not happy",
    "This movie is good",
    "This movie is not good",
    "I don't like it",
    "The film was bad",
    "The film was not bad",
    "I really love this movie",
    "I really hate this movie",
]


def proba_sklearn(model, texts, preprocess_fn: Callable, vectorizer) -> list[float]:
    cleaned = [preprocess_fn(t) for t in texts]
    probs = model.predict_proba(vectorizer.transform(cleaned))[:, 1]
    return [float(x) for x in probs]


def build_controlled_table(
    *,
    lr,
    mlp,
    tfidf,
    preprocess_article,
    preprocess_negation,
    lr_neg=None,
    tfidf_neg=None,
    bert_proba_fn: Callable | None = None,
) -> list[dict]:
    cols = {
        "Logistic Regression": proba_sklearn(lr, CONTROLLED_EXAMPLES, preprocess_article, tfidf),
        "MLP (TF-IDF)": proba_sklearn(mlp, CONTROLLED_EXAMPLES, preprocess_article, tfidf),
    }
    if lr_neg is not None and tfidf_neg is not None:
        cols["LR (keep negations)"] = proba_sklearn(
            lr_neg, CONTROLLED_EXAMPLES, preprocess_negation, tfidf_neg
        )
    if bert_proba_fn is not None:
        cols["DistilBERT"] = [float(x) for x in bert_proba_fn(CONTROLLED_EXAMPLES)]

    rows = []
    for i, text in enumerate(CONTROLLED_EXAMPLES):
        row = {"Text": text}
        for name, vals in cols.items():
            row[name] = vals[i]
        rows.append(row)
    return rows
