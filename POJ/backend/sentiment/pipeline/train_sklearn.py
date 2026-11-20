from __future__ import annotations

import time
from typing import Callable

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from .config import SEED
from .evaluate import metrics


def article_model_factories() -> dict:
    return {
        "KNN": lambda: KNeighborsClassifier(n_neighbors=5),
        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=8, bootstrap=True, random_state=SEED, n_jobs=-1
        ),
        "Naive Bayes": lambda: MultinomialNB(),
        "Logistic Regression": lambda: LogisticRegression(max_iter=1000, random_state=SEED),
    }


def make_mlp() -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=(256,),
        activation="relu",
        alpha=1e-4,
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=30,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=3,
        random_state=SEED,
    )


def train_article_models(X_train, y_train, X_by_set: dict) -> tuple[dict, dict, dict]:
    fitted = {}
    preds = {}
    timings = {}
    for name, factory in article_model_factories().items():
        model = factory()
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        timings[name] = time.perf_counter() - t0
        fitted[name] = model
        for ds, X in X_by_set.items():
            preds[(name, ds)] = model.predict(X)
    return fitted, preds, timings


def train_mlp(X_train, y_train, X_by_set: dict) -> tuple[MLPClassifier, dict, float]:
    mlp = make_mlp()
    t0 = time.perf_counter()
    mlp.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    preds = {ds: mlp.predict(X) for ds, X in X_by_set.items()}
    return mlp, preds, elapsed


def run_feature_ablation(train_df, test_bundles: dict, y_by_set: dict, article_preds: dict) -> list[dict]:
    """test_bundles: {ds: {"clean": list, "text_lower": list}}"""
    feature_builders = {
        "TF-IDF + bigrams": TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30_000),
        "Bag of Words": CountVectorizer(binary=True, min_df=2, max_features=30_000),
        "Char n-grams (3-5)": CountVectorizer(
            analyzer="char", ngram_range=(3, 5), min_df=2, max_features=30_000
        ),
    }
    y_train = train_df["label"].astype(int).values
    train_text_word = train_df["clean"].tolist()
    train_text_char = train_df["text"].str.lower().tolist()

    rows = []
    for feat_name, vectorizer in feature_builders.items():
        use_char = feat_name.startswith("Char")
        tr = train_text_char if use_char else train_text_word
        Xtr = vectorizer.fit_transform(tr)
        Xte = {}
        for ds, bundle in test_bundles.items():
            texts = bundle["text_lower"] if use_char else bundle["clean"]
            Xte[ds] = vectorizer.transform(texts)
        for model_name, factory in article_model_factories().items():
            clf = factory()
            clf.fit(Xtr, y_train)
            for ds, y in y_by_set.items():
                pred = clf.predict(Xte[ds])
                rows.append(
                    {
                        "Representation": feat_name,
                        "Model": model_name,
                        "Dataset": ds,
                        **metrics(y, pred),
                    }
                )

    for model_name in article_model_factories():
        for ds, y in y_by_set.items():
            rows.append(
                {
                    "Representation": "TF-IDF (main)",
                    "Model": model_name,
                    "Dataset": ds,
                    **metrics(y, article_preds[(model_name, ds)]),
                }
            )
    return rows


def train_negation_lr(train_df, test_dfs: dict):
    tfidf_neg = TfidfVectorizer()
    X_train_neg = tfidf_neg.fit_transform(train_df["clean_neg"])
    y_train = train_df["label"].astype(int).values
    lr_neg = LogisticRegression(max_iter=1000, random_state=SEED)
    lr_neg.fit(X_train_neg, y_train)
    X_neg = {ds: tfidf_neg.transform(df["clean_neg"]) for ds, df in test_dfs.items()}
    preds = {ds: lr_neg.predict(X) for ds, X in X_neg.items()}
    return lr_neg, tfidf_neg, preds
