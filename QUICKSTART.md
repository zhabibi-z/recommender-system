# Quick Start

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## Deploy to Streamlit Cloud

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Select repo → `app.py` → Deploy

---

## File Structure

```
recommender-system/
├── app.py                    # Streamlit app
├── requirements.txt
├── training.csv              # Amazon book ratings (training set)
├── test_features.csv         # Test set (user/ASIN pairs)
├── book_metadata.csv         # Titles, authors, cover URLs
├── book_descriptions.csv     # Book descriptions for embeddings
├── book_embeddings.npy       # Sentence embeddings (384-dim)
├── book_embeddings_index.csv # ASIN → embedding index mapping
├── book-recommender.ipynb    # Development notebook
└── .streamlit/config.toml
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: streamlit` | `pip install -r requirements.txt` |
| CSV not found | Ensure CSVs are in the same directory as `app.py` |
| Training slow | Reduce `n_steps=300` → `n_steps=150` in `app.py` |
