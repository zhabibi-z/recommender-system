# Book Recommender System

> A production-ready hybrid recommender engine combining **Bayesian Personalized Ranking**
> with **semantic search** — built on 500K+ Amazon book reviews with a clean,
> end-to-end ML pipeline.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-app-red.svg)](https://recommender-system-book.streamlit.app)
[![Tests](https://img.shields.io/badge/tests-15%20passed-brightgreen.svg)](#testing)

**[Live Demo] : (https://recommender-system-book.streamlit.app)**

---

## The Problem

Finding your next book is harder than it should be. Bestseller lists show you the same
100 titles. Search requires you to know what you're looking for. Neither approach
understands *your* taste.

This project builds a system that does three things:
1. **Learns your preferences** from your reading history (collaborative filtering)
2. **Understands book descriptions** semantically (dense embeddings)
3. **Blends both signals** into a single ranked list you can tune in real-time

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      DATA PIPELINE                              │
│                                                                 │
│  Amazon Reviews 2023           Open Library / Amazon Metadata   │
│        │                               │                        │
│        ▼                               ▼                        │
│  src/data/download.py          src/data/download.py             │
│  (stream 500K reviews)         (stream book metadata)           │
│        │                               │                        │
│        ▼                               │                        │
│  src/data/clean.py                     │                        │
│  ├─ deduplication                      │                        │
│  ├─ k-core filter (≥20/≥30)           │                        │
│  └─ temporal split (per-user)          │                        │
│        │                               │                        │
│        ▼                               ▼                        │
│  data/processed/               src/data/features.py             │
│  ├─ train.parquet              ├─ user_features.parquet         │
│  ├─ val.parquet                └─ item_features.parquet         │
│  └─ test.parquet                       │                        │
│                                        ▼                        │
│                              generate_embeddings.py             │
│                              (all-MiniLM-L6-v2)                 │
│                              └─ book_embeddings.npy             │
└─────────────────────────────────────────────────────────────────┘
                    │                       │
                    ▼                       ▼
┌───────────────────────────┐  ┌────────────────────────────────┐
│     CF MODEL (BPR)        │  │     SEMANTIC SEARCH            │
│                           │  │                                │
│  Optimises ranking:       │  │  query → encoder → embedding   │
│  P(i ≻ j | u) = σ(s_ui   │  │       → cosine similarity      │
│               − s_uj)    │  │       → ranked book list       │
│                           │  │                                │
│  Trained with Adam,       │  │  all-MiniLM-L6-v2             │
│  early stop on NDCG@10   │  │  384-dim embeddings            │
└───────────┬───────────────┘  └──────────────┬─────────────────┘
            │                                  │
            └──────────────┬───────────────────┘
                           ▼
              ┌────────────────────────┐
              │   HYBRID BLENDING      │
              │                        │
              │  score = α·CF + (1-α)  │
              │          ·semantic     │
              │                        │
              │  α tunable [0, 1]      │
              └────────────┬───────────┘
                           ▼
                    Streamlit App
              ┌──────────────────────┐
              │  📖 For You  (CF)    │
              │  🔍 Search by Vibe   │
              │  🔀 Hybrid           │
              └──────────────────────┘
```

---

## Why BPR Instead of Matrix Factorisation

The original model used MSE-based matrix factorisation, which produced a
**Val RMSE of 1.87** — *worse* than simply predicting the global mean (RMSE 0.78).

**Root cause:** 63% of Amazon book ratings are 5 stars. The dataset is not a
random sample of preferences — it reflects *selection bias* (people review
books they already chose to read). MSE on these ratings is not a useful
training signal.

**BPR fixes this** by optimising a ranking objective: given user *u*, positive
item *i* (interacted), and negative item *j* (not interacted), BPR maximises:

```
L_BPR = -Σ log σ(score(u,i) − score(u,j)) + λ‖Θ‖²
```

This asks "did the user prefer this book over a random book?" — the right
question for a recommender system.

| Model | Val RMSE | NDCG@10 | Recall@10 |
|---|---|---|---|
| Global mean baseline | 0.78 | — | — |
| Matrix factorisation (original) | 1.87 | — | — |
| **BPR (this project)** | — | *run pipeline* | *run pipeline* |

---

## Project Structure

```
├── src/
│   ├── config.py               centralised configuration (dataclasses)
│   ├── data/
│   │   ├── download.py         stream Amazon Reviews 2023 via HuggingFace
│   │   ├── clean.py            dedup, k-core filter, temporal split
│   │   └── features.py         user & item feature engineering
│   ├── models/
│   │   ├── bpr.py              Bayesian Personalized Ranking (PyTorch)
│   │   └── matrix_factorization.py  improved biased MF (baseline)
│   └── utils/
│       └── metrics.py          RMSE, MAE, Precision@K, Recall@K, NDCG@K
│
├── notebooks/
│   ├── 01_eda.ipynb            exploratory data analysis
│   └── 02_features.ipynb       feature engineering analysis
│
├── data/
│   ├── raw/                    downloaded parquet files (gitignored)
│   └── processed/              cleaned splits (gitignored)
│
├── models/                     saved model checkpoints (gitignored)
├── tests/
│   └── test_metrics.py         15 unit tests (all passing)
│
├── app.py                      Streamlit application
├── pretrain_model.py           train BPR model
├── generate_embeddings.py      encode book descriptions
└── run_pipeline.py             orchestrate full pipeline
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline (download → clean → features → embed → train)
python run_pipeline.py

# 3. Launch the app
streamlit run app.py
```

**Or step by step:**
```bash
python -m src.data.download          # download Amazon Reviews 2023
python -m src.data.clean             # clean and split
python -m src.data.features          # feature engineering
python generate_embeddings.py        # sentence embeddings
python pretrain_model.py             # train BPR
streamlit run app.py                 # launch app
```

**Run tests:**
```bash
python -m pytest tests/ -v
```

---

## Data Pipeline

| Stage | Input | Output | Key decision |
|---|---|---|---|
| **Download** | HuggingFace stream | `reviews_raw.parquet`, `metadata_raw.parquet` | Streaming avoids materialising the full dataset |
| **Clean** | Raw reviews | `train/val/test.parquet` | k-core (≥20 user, ≥30 item) + temporal per-user split |
| **Features** | Cleaned ratings + metadata | `user/item_features.parquet` | Mean rating, std, genre affinity, description length |
| **Embed** | Book descriptions + genre | `book_embeddings.npy` | Title + genre + description → richer semantic text |
| **Train** | Train/val splits | `models/bpr_model.pt` | BPR with Adam, early stop on NDCG@10 |

**Why temporal split?**
A random 80/20 split would let the model "see" future ratings during training,
inflating evaluation metrics. Per-user temporal holdout mirrors real deployment:
train on the past, predict the future.

---

## EDA Highlights

See [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb) for full analysis.

| Finding | Implication |
|---|---|
| 63%+ of ratings are 5-star | MSE is dominated by noise; BPR ranking objective is appropriate |
| Power-law user activity | k-core filtering removes noisy tail users |
| Long-tail item popularity | Top 20% of books get ~70% of ratings; uniform negative sampling in BPR counteracts this |
| Temporal structure in reviews | Per-user temporal split for honest evaluation |
| Rich metadata in Amazon 2023 | High description coverage → high-quality semantic embeddings |

---

## Stack

| Layer | Technology |
|---|---|
| Model training | PyTorch |
| Semantic embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Dataset | Amazon Reviews 2023 (McAuley-Lab / HuggingFace) |
| Data processing | pandas, pyarrow |
| App | Streamlit |
| Testing | pytest |

---

## Testing

```
tests/test_metrics.py   15 tests   ✅ all passing
```

Covers: RMSE, MAE, Precision@K, Recall@K, NDCG@K, Hit@K, and the aggregate
`evaluate_ranking` evaluator with edge cases (empty relevant sets, perfect
ranking, worst-case ordering).

---

## Roadmap

- [ ] LightFM — incorporate user/item side features into the BPR objective
- [ ] Two-tower neural network — scale to millions of items with ANN retrieval
- [ ] FAISS index — sub-millisecond semantic search at scale
- [ ] Cold-start onboarding — genre preference survey for new users
- [ ] A/B testing framework — compare model variants on live traffic
