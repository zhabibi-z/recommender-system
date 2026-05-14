# Book Recommender System

A hybrid book recommendation engine combining **collaborative filtering** via matrix factorization with **semantic search** via sentence embeddings. Trained on Amazon book review data and deployed as an interactive web app.

**Live demo:** [recommender-system-book.streamlit.app](https://recommender-system-book.streamlit.app)

---

## How It Works

### Collaborative Filtering
The rating matrix is factorized as:

**P = μ + b_u + b_i + A × F**

| Term | Description |
|------|-------------|
| μ | Global mean rating |
| b_u | User bias vector |
| b_i | Item bias vector |
| A (n × k) | User latent factor matrix |
| F (k × m) | Item latent factor matrix |

Parameters are learned by minimizing masked MSE with L2 regularization via full-batch gradient descent with early stopping.

### Semantic Search
Book descriptions are encoded using `sentence-transformers` (`all-MiniLM-L6-v2`). At query time, cosine similarity between the query embedding and all book embeddings ranks results by semantic relevance.

---

## Dataset

- **1,490** users · **1,186** books · **61,104** ratings
- Rating scale: 1–5
- Sparsity: 96.5%
- Source: Amazon book reviews

---

## Stack

- **PyTorch** — model training
- **sentence-transformers** — semantic embeddings
- **Streamlit** — web interface
- **Open Library API** — book metadata and covers
- **pandas / numpy** — data processing

---

## Results

| Model | Val RMSE |
|-------|----------|
| Global mean baseline | ~1.50 |
| Matrix factorization (no bias) | 3.13 |
| Matrix factorization + bias | **1.87** |

---

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
