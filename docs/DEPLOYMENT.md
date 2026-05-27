# Deploying the Book Recommender System

## Local Development

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Streamlit Community Cloud

Free hosting directly from GitHub.

1. Push repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Connect repo, set `app.py` as entry point → Deploy

The app URL will be `https://yourusername-repo-appname.streamlit.app`

---

## Heroku

Create a `Procfile` (already included):

```
web: streamlit run app.py
```

Create `.streamlit/config.toml`:

```toml
[server]
headless = true
port = $PORT
enableCORS = false
```

Deploy:

```bash
heroku create your-app-name
git push heroku main
```

---

## Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
```

```bash
docker build -t recommender-app .
docker run -p 8501:8501 recommender-app
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError` | `pip install -r requirements.txt` |
| CSV not found | Ensure CSVs are in the same directory as `app.py` |
| Training slow | Reduce `n_steps` in `train_model()` or lower `k` |
