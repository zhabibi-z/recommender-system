# Architecture — Book Recommender System

## Overview

A hybrid recommender system combining **collaborative filtering** (Bayesian Personalised Ranking) with **semantic search** (sentence-transformers embeddings) to recommend books. Deployed as a Streamlit web app.

## Component Map

```
data/legacy/training.csv          ← user–book interaction matrix
data/legacy/book_metadata.csv     ← title, author, genre, description
        │
        ▼  src/data/download.py
Raw data ingestion (HuggingFace Datasets / local CSV)
        │
        ▼  src/data/clean.py
k-core filtering:
  min_user_ratings = 5   (remove cold-start users)
  min_item_ratings = 10  (remove long-tail items)
Temporal train / val / test split (per-user holdout)
        │
        ├──────────────────────────────────────────┐
        ▼  scripts/pretrain_model.py               ▼  scripts/generate_embeddings.py
BPR Matrix Factorisation                    sentence-transformers
  k=64 latent factors                         all-MiniLM-L6-v2
  n_negatives=4 per positive                  batch_size=64
  L2 reg λ=0.001                              384-dim embeddings
  early stopping (patience=5)                 FAISS index for ANN search
        │                                           │
        ▼                                           ▼
models/bpr_model.pt                   data/embeddings/book_embeddings.npy
                                      data/embeddings/book_embeddings_index.csv
        │                                           │
        └──────────────┬────────────────────────────┘
                       ▼  app.py
             Streamlit dashboard
               ├── Collaborative filter tab  — BPR top-N for known users
               ├── Semantic search tab       — embedding similarity search
               └── Hybrid tab               — weighted combination
```

## Evaluation Metrics

Computed in `src/utils/metrics.py` and tested in `tests/test_metrics.py`:

| Metric | Description |
|---|---|
| `Precision@K` | Fraction of top-K recommendations that are relevant |
| `Recall@K` | Fraction of relevant items recovered in top-K |
| `NDCG@K` | Normalised Discounted Cumulative Gain — rank-aware quality |
| `Hit@K` | Binary: does the top-K list contain at least one relevant item |

## Configuration

All hyperparameters live in `src/config.py` as frozen dataclasses:

```python
from src.config import cfg

cfg.model.k           # latent factor dimension = 64
cfg.data.val_frac     # validation holdout = 0.10
cfg.embeddings.model_name  # all-MiniLM-L6-v2
```

## Design Decisions

**BPR over ALS** — Bayesian Personalised Ranking optimises for ranking (pairwise loss) rather than rating prediction (RMSE). For implicit feedback (click/purchase history), ranking quality matters more than predicted score accuracy.

**Hybrid approach** — Collaborative filtering suffers from cold-start for new items; semantic search handles it via description embeddings. The hybrid tab lets users tune the weighting.

**k-core filtering** — Removes users with fewer than 5 interactions and items with fewer than 10, reducing sparsity and improving embedding quality.

**Temporal holdout** — The most recent interactions per user form the test set, matching real deployment conditions where the model always predicts future behaviour from past data.

## Directory Structure

```
.
├── app.py                      # Streamlit dashboard
├── run_pipeline.py             # End-to-end pipeline runner
├── requirements.txt            # Pinned runtime + dev dependencies
├── data/
│   ├── embeddings/             # Precomputed book embeddings (FAISS-ready)
│   └── legacy/                 # Source CSVs
├── docs/
│   └── ARCHITECTURE.md
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_features.ipynb
├── scripts/
│   ├── generate_embeddings.py  # Compute + save sentence-transformer embeddings
│   └── pretrain_model.py       # Train BPR model
├── src/
│   ├── config.py               # Dataclass configuration singleton
│   ├── data/                   # Ingestion, cleaning, feature engineering
│   └── utils/metrics.py        # Ranking metric implementations
└── tests/
    └── test_metrics.py         # pytest coverage for all ranking metrics
```
