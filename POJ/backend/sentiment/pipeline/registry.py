from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any

import joblib
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .preprocess import preprocess_article, preprocess_negation
from .train_bert import bert_positive_proba, get_device


class ModelRegistry:
    """Process-wide loaded models for inference."""

    def __init__(self) -> None:
        self._lock = Lock()
        self.active_run_id: str | None = None
        self.tfidf = None
        self.tfidf_neg = None
        self.sklearn_models: dict[str, Any] = {}
        self.lr_neg = None
        self.bert_model = None
        self.bert_tokenizer = None
        self.device = get_device()

    @property
    def loaded_models(self) -> list[str]:
        names = list(self.sklearn_models.keys())
        if self.lr_neg is not None:
            names.append("LR (keep negations)")
        if self.bert_model is not None:
            names.append("DistilBERT")
        return names

    def _reset(self) -> None:
        self.active_run_id = None
        self.tfidf = None
        self.tfidf_neg = None
        self.sklearn_models = {}
        self.lr_neg = None
        self.bert_model = None
        self.bert_tokenizer = None

    def clear(self) -> None:
        with self._lock:
            self._reset()

    def load_from_dir(self, run_id: str, artifact_dir: Path) -> list[str]:
        artifact_dir = Path(artifact_dir)
        with self._lock:
            self._reset()
            self.active_run_id = run_id
            tfidf_path = artifact_dir / "tfidf_article.joblib"
            if tfidf_path.exists():
                self.tfidf = joblib.load(tfidf_path)

            for name, filename in [
                ("KNN", "knn.joblib"),
                ("Random Forest", "random_forest.joblib"),
                ("Naive Bayes", "naive_bayes.joblib"),
                ("Logistic Regression", "lr_article.joblib"),
                ("MLP (TF-IDF)", "mlp_tfidf.joblib"),
            ]:
                path = artifact_dir / filename
                if path.exists() and self.tfidf is not None:
                    self.sklearn_models[name] = joblib.load(path)

            neg_model = artifact_dir / "lr_neg.joblib"
            neg_vec = artifact_dir / "tfidf_neg.joblib"
            if neg_model.exists() and neg_vec.exists():
                self.lr_neg = joblib.load(neg_model)
                self.tfidf_neg = joblib.load(neg_vec)

            bert_dir = artifact_dir / "distilbert"
            if bert_dir.exists() and (bert_dir / "config.json").exists():
                self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_dir)
                self.bert_model = AutoModelForSequenceClassification.from_pretrained(bert_dir)
                self.bert_model.to(self.device)
                self.bert_model.eval()

            return self.loaded_models

    def load_bootstrap(self, bootstrap_dir: Path) -> list[str]:
        """Load all notebook-exported models from sentiment-analysis/artifacts/."""
        bootstrap_dir = Path(bootstrap_dir)
        if not bootstrap_dir.exists():
            return []
        # Reuse the same filenames as experiment runs / notebook §14 export
        return self.load_from_dir("bootstrap", bootstrap_dir)

    def predict(
        self,
        text: str,
        models: list[str] | None = None,
        preprocess: str = "article",
    ) -> dict[str, Any]:
        with self._lock:
            wanted = models or self.loaded_models
            results = {}
            for name in wanted:
                if name == "DistilBERT":
                    if self.bert_model is None or self.bert_tokenizer is None:
                        continue
                    p = float(bert_positive_proba(self.bert_model, self.bert_tokenizer, [text])[0])
                    results[name] = {
                        "label": int(p >= 0.5),
                        "label_name": "positive" if p >= 0.5 else "negative",
                        "p_positive": p,
                    }
                    continue

                if name == "LR (keep negations)":
                    if self.lr_neg is None or self.tfidf_neg is None:
                        continue
                    cleaned = preprocess_negation(text)
                    proba = self.lr_neg.predict_proba(self.tfidf_neg.transform([cleaned]))[0]
                    p = float(proba[1])
                    results[name] = {
                        "label": int(p >= 0.5),
                        "label_name": "positive" if p >= 0.5 else "negative",
                        "p_positive": p,
                    }
                    continue

                model = self.sklearn_models.get(name)
                if model is None or self.tfidf is None:
                    continue
                fn = preprocess_negation if preprocess == "negation" else preprocess_article
                cleaned = fn(text)
                X = self.tfidf.transform([cleaned])
                if hasattr(model, "predict_proba"):
                    p = float(model.predict_proba(X)[0][1])
                    label = int(p >= 0.5)
                else:
                    label = int(model.predict(X)[0])
                    p = float(label)
                results[name] = {
                    "label": label,
                    "label_name": "positive" if label == 1 else "negative",
                    "p_positive": p,
                }
            return {
                "text": text,
                "preprocess": preprocess,
                "predictions": results,
            }


registry = ModelRegistry()
