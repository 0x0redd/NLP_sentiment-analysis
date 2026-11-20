from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .config import SEED


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def train_distilbert(
    train_texts: list[str],
    train_labels: list[int],
    eval_raw: dict[str, list[str]],
    output_dir: Path,
    progress: Callable[[str], None] | None = None,
    fast_mode: bool = True,
) -> tuple[object, object, dict, float]:
    """Fine-tune DistilBERT and return (model, tokenizer, preds_by_dataset, train_seconds)."""
    set_seed(SEED)
    device = get_device()
    if progress:
        progress(f"DistilBERT: loading tokenizer/model on {device}")

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def to_hf(texts, labels, max_len):
        ds = Dataset.from_dict({"text": list(texts), "labels": [int(l) for l in labels]})
        return ds.map(
            lambda b: tokenizer(b["text"], truncation=True, max_length=max_len),
            batched=True,
            remove_columns=["text"],
        )

    train_ds = to_hf(train_texts, train_labels, 128)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "trainer_logs"),
        num_train_epochs=2,
        per_device_train_batch_size=16 if device == "cpu" else 32,
        per_device_eval_batch_size=64 if device == "cpu" else 128,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=100 if fast_mode else 200,
        fp16=(device == "cuda"),
        logging_steps=50 if fast_mode else 200,
        save_strategy="no",
        report_to="none",
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    if progress:
        progress("DistilBERT: training…")
    t0 = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - t0

    max_len = {"SST-2": 128, "SST-2 val.": 128, "IMDB": 256}
    preds = {}
    for ds, texts in eval_raw.items():
        # dummy labels for HF Dataset API
        labels = [0] * len(texts)
        eval_ds = to_hf(texts, labels, max_len[ds])
        logits = trainer.predict(eval_ds).predictions
        preds[ds] = logits.argmax(axis=-1)

    save_path = output_dir / "distilbert"
    trainer.model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    if progress:
        progress(f"DistilBERT: saved to {save_path}")

    return trainer.model, tokenizer, preds, elapsed


def bert_positive_proba(model, tokenizer, texts: list[str], max_length: int = 128) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    enc = tokenizer(
        texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        logits = model(**enc).logits
    return torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
