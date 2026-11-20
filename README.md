# NLP Sentiment Analysis

Academic project built around **Xinya Liao (2024), *Sentiment Analysis Based on Machine Learning Models***: reproduce the paper’s binary sentiment protocol, then push a **new contribution** beyond its TF-IDF setup, following the roadmap in [`feature_extraction_methods.pdf`](feature_extraction_methods.pdf) (*Beyond TF-IDF: Feature Extraction Methods for Sentiment Classification*).

## Goal

1. **Reproduce** the article: train classical classifiers on SST-2 with TF-IDF features and evaluate on held-out SST-2 and cross-domain IMDB.
2. **Contribute** by changing what the models see — richer feature extractors and stronger models — and by measuring whether those changes actually help.

The paper’s reference representation is TF-IDF. The PDF argues that this is only one family of features (count-based). The project explores the other two families (static embeddings and contextual transformers) and reports where they beat or fail relative to the paper baseline.

## Paper baseline

| Piece | What we follow |
| --- | --- |
| Task | Binary sentiment (positive / negative) |
| Train / in-domain test | SST-2 (paper-style labelled split) |
| Cross-domain test | IMDB |
| Features | TF-IDF (article preprocessing) |
| Models | KNN, Random Forest, Multinomial Naive Bayes, Logistic Regression |

## Contributions (beyond the article)

- **Feature-side** (aligned with [`feature_extraction_methods.pdf`](feature_extraction_methods.pdf)): Bag-of-Words, TF-IDF + bigrams, character *n*-grams, and planned / partial comparisons with LSA, static embeddings (Word2Vec / GloVe / fastText), Doc2Vec, frozen sentence embeddings, and DistilBERT as an upper bound.
- **Model-side**: MLP on TF-IDF; fine-tuned DistilBERT.
- **Analysis**: McNemar–Holm pairwise tests, feature ablations, negation-aware preprocessing, controlled phrases, length / noise / vocabulary-size probes.

## Repository layout

| Path | Role |
| --- | --- |
| [`feature_extraction_methods.pdf`](feature_extraction_methods.pdf) | Contribution roadmap: why leave TF-IDF, three feature families, experimental protocol |
| [`paper.ipynb`](paper.ipynb) | EDA + feature-representation ablations (BoW, TF-IDF, bigrams, char *n*-grams, …) on SST-2 / IMDB |
| [`proj.ipynb`](proj.ipynb) | TF-IDF + classic ML with hyperparameter search (baseline vs tuned) |
| [`PRJFN2.ipynb`](PRJFN2.ipynb) | Full pipeline with extra analyses (text length, XAI / errors, lexical noise, TF-IDF vocab size) |
| [`sentiment_merged.ipynb`](sentiment_merged.ipynb) | Merged reproduction: paper models + MLP + DistilBERT + McNemar / ablations |
| [`POJ/`](POJ/) | Django + Next.js app: retrain, live prediction, results and comparison tables |

## How to explore

- Start with [`paper.ipynb`](paper.ipynb) for data and feature comparisons.
- Use [`proj.ipynb`](proj.ipynb) when you care about hyperparameter tuning under TF-IDF.
- Use [`PRJFN2.ipynb`](PRJFN2.ipynb) for robustness and explainability experiments.
- Use [`sentiment_merged.ipynb`](sentiment_merged.ipynb) for the end-to-end paper reproduction plus MLP / DistilBERT contributions.
- Run the web demo from [`POJ/README.md`](POJ/README.md).

## Stack

Python (scikit-learn, Hugging Face datasets / Transformers), optional CUDA for DistilBERT, Django REST + Next.js in `POJ/`.
