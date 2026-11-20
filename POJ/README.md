# POJ — Sentiment showcase (Next.js + Django)

Productized runner for [`sentiment_merged.ipynb`](../sentiment_merged.ipynb): full retrain API, live prediction, and research tables (McNemar, ablation, controlled phrases).

## Layout

```
POJ/
  backend/    Django 5 + DRF
  frontend/   Next.js (App Router, TypeScript)
```

## Backend

```powershell
cd POJ\backend
# use the dl conda env (or any env with requirements installed)
conda activate dl
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver 8000
```

### DistilBERT GPU path

- `GET /api/health/` reports `device` (`cuda` or `cpu`).
- On **Train**, enable **Fine-tune DistilBERT**. Prefer a CUDA GPU; CPU works but is slow.
- After a succeeded run, click **Activate**, then use **Predict**.
- Datasets load from `nyu-mll/glue` (SST-2) and `stanfordnlp/imdb`, with fallback to `sentiment-analysis/cache/*.parquet`.

### Bootstrap weights (from the notebook)

Export **all** models from `sentiment_merged copy.ipynb` §14 (and DistilBERT in §9) into `sentiment-analysis/artifacts/`:

| File | Model |
|------|--------|
| `tfidf_article.joblib` | main TF-IDF vectorizer |
| `knn.joblib` | KNN |
| `random_forest.joblib` | Random Forest |
| `naive_bayes.joblib` | Naive Bayes |
| `lr_article.joblib` | Logistic Regression |
| `mlp_tfidf.joblib` | MLP (TF-IDF) |
| `lr_neg.joblib` + `tfidf_neg.joblib` | LR (keep negations) |
| `distilbert/` | DistilBERT checkpoint |

On Django startup these are auto-loaded. Open **Predict**, paste a phrase, and every loaded model returns `label` + `p_positive`.

### Main API

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/health/` | device, loaded models |
| POST | `/api/runs/` | start retrain |
| GET | `/api/runs/{id}/` | poll status |
| POST | `/api/runs/{id}/activate/` | load artifacts into registry |
| GET | `/api/runs/{id}/results/` | metrics table |
| GET | `/api/runs/{id}/mcnemar/` | McNemar–Holm |
| GET | `/api/runs/{id}/ablation/` | feature ablation |
| GET | `/api/runs/{id}/confusion/` | confusion matrices |
| GET | `/api/runs/{id}/timings/` | train times |
| GET | `/api/runs/{id}/controlled/` | controlled phrases |
| POST | `/api/predict/` | live inference |
| POST | `/api/preprocess/` | article vs negation cleaning |

## Frontend

```powershell
cd POJ\frontend
npm install
npm run dev
```

Open http://localhost:3000 — API base URL is `NEXT_PUBLIC_API_URL` (default `http://127.0.0.1:8000`).

## Suggested demo flow

1. Start backend + frontend.
2. **Predict** with bootstrap LR/MLP (if artifacts present), or **Train** with `FAST_MODE` and DistilBERT off.
3. When the run succeeds, **Activate**, then predict again.
4. Open **Results** and **Compare** for the research tables.
