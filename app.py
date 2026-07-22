"""
Book Recommender — Streamlit application.

Recommendation modes
--------------------
  For You       Collaborative filtering (BPR ranking model)
  Search        Semantic search via sentence embeddings
  Hybrid        Blend of CF score and semantic similarity

Data loading order (first available wins)
-----------------------------------------
  1. data/processed/*.parquet  — Amazon Reviews 2023 pipeline
  2. training.csv / book_metadata.csv  — legacy Kaggle dataset

Model loading order
-------------------
  1. models/bpr_model.pt  — BPR ranking model (preferred)
  2. model.pt             — legacy matrix factorisation checkpoint
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Book Recommender",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_data() -> dict:
    """Load ratings and metadata, preferring the new pipeline."""
    processed = Path("data/processed")

    if (processed / "ratings.parquet").exists():
        ratings   = pd.read_parquet(processed / "ratings.parquet")
        user_idx  = pd.read_parquet(processed / "user_index.parquet")
        item_idx  = pd.read_parquet(processed / "item_index.parquet")
        user2idx  = dict(zip(user_idx["user_id"], user_idx["user_idx"]))
        item2idx  = dict(zip(item_idx["asin"],    item_idx["item_idx"]))
        idx2item  = {v: k for k, v in item2idx.items()}

        meta_path = Path("data/raw/metadata_raw.parquet")
        if meta_path.exists():
            meta_df = pd.read_parquet(meta_path)
        else:
            meta_df = pd.DataFrame(columns=["asin", "title", "description", "genre", "cover_url"])

        metadata = {}
        for _, row in meta_df.iterrows():
            metadata[row["asin"]] = {
                "title":     row.get("title", ""),
                "author":    "",
                "cover_url": row.get("cover_url", ""),
                "genre":     row.get("genre", ""),
            }

        return {
            "source":   "amazon_2023",
            "ratings":  ratings,
            "user2idx": user2idx,
            "item2idx": item2idx,
            "idx2item": idx2item,
            "metadata": metadata,
            "n_users":  len(user2idx),
            "n_items":  len(item2idx),
        }

    # ── Legacy fallback ────────────────────────────────────────────────────────
    training      = pd.read_csv("data/legacy/training.csv")
    unique_users  = sorted(training["User"].unique().tolist())
    unique_asins  = sorted(training["ASIN"].unique().tolist())
    user2idx      = {u: i for i, u in enumerate(unique_users)}
    item2idx      = {a: i for i, a in enumerate(unique_asins)}
    idx2item      = {i: a for a, i in item2idx.items()}

    metadata: dict = {}
    if Path("data/legacy/book_metadata.csv").exists():
        meta_df = pd.read_csv("data/legacy/book_metadata.csv", dtype=str).fillna("")
        for _, row in meta_df.iterrows():
            metadata[row["asin"]] = {
                "title":     row["title"],
                "author":    row["author"],
                "cover_url": row["cover_url"],
                "genre":     "",
            }

    pivot   = training.pivot_table(index="User", columns="ASIN", values="Rating", fill_value=0)
    pivot   = pivot.reindex(index=unique_users, columns=unique_asins, fill_value=0)
    S       = pivot.values.astype(float)
    ratings = pd.DataFrame({
        "user_id":   training["User"],
        "asin":      training["ASIN"],
        "rating":    training["Rating"],
        "user_idx":  training["User"].map(user2idx),
        "item_idx":  training["ASIN"].map(item2idx),
        "timestamp": 0,
    })

    return {
        "source":   "legacy",
        "ratings":  ratings,
        "user2idx": user2idx,
        "item2idx": item2idx,
        "idx2item": idx2item,
        "metadata": metadata,
        "n_users":  len(unique_users),
        "n_items":  len(unique_asins),
        "S":        S,
    }


@st.cache_resource(show_spinner=False)
def load_model(n_users: int, n_items: int):
    """Load BPR model if available, else fall back to legacy MF."""
    bpr_path = Path("models/bpr_model.pt")
    if bpr_path.exists():
        from src.models.bpr import load as load_bpr
        model, ckpt = load_bpr(bpr_path)
        model.eval()
        return {"type": "bpr", "model": model, "ckpt": ckpt}

    legacy_path = Path("models/model.pt")
    if legacy_path.exists():
        ckpt = torch.load(legacy_path, weights_only=False)
        A, F, b_u, b_i, mu = ckpt["A"], ckpt["F"], ckpt["b_u"], ckpt["b_i"], ckpt["mu"]
        P = np.clip((mu + b_u + b_i + A @ F).detach().numpy(), 1, 5)
        return {"type": "mf_legacy", "P": P, "ckpt": ckpt}

    return None


@st.cache_resource(show_spinner=False)
def load_embeddings():
    emb_file = Path("data/embeddings/book_embeddings.npy")
    idx_file = Path("data/embeddings/book_embeddings_index.csv")
    if not emb_file.exists():
        return None, None
    embeddings = np.load(emb_file)
    index_df   = pd.read_csv(idx_file, dtype=str)
    asin2idx   = {row["asin"]: int(row["idx"]) for _, row in index_df.iterrows()}
    return embeddings, asin2idx


@st.cache_resource(show_spinner=False)
def load_encoder():
    import io
    import sys
    from sentence_transformers import SentenceTransformer
    old_err, sys.stderr = sys.stderr, io.StringIO()
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    finally:
        sys.stderr = old_err


# ── Scoring helpers ───────────────────────────────────────────────────────────

def cf_scores_for_user(user_idx: int, model_state: dict, n_items: int) -> np.ndarray:
    if model_state["type"] == "bpr":
        return model_state["model"].score_all_items(user_idx)
    else:
        return model_state["P"][user_idx]


def encode_query(query: str) -> np.ndarray:
    enc = load_encoder()
    v   = enc.encode([query])[0]
    n   = np.linalg.norm(v)
    return v / (n if n > 0 else 1)


def semantic_results(query: str, embeddings: np.ndarray, asin2idx: dict, top_k: int):
    q    = encode_query(query)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / np.where(norms == 0, 1, norms)
    idxs  = list(asin2idx.values())
    asins = list(asin2idx.keys())
    sims  = normed[idxs] @ q
    top   = np.argsort(sims)[::-1][:top_k]
    return [(asins[i], float(sims[i])) for i in top]


def hybrid_results(
    query: str, user_idx: int,
    model_state: dict, n_items: int,
    embeddings: np.ndarray, asin2idx: dict,
    item2idx: dict, alpha: float, top_k: int,
):
    q      = encode_query(query)
    norms  = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / np.where(norms == 0, 1, norms)
    cf     = cf_scores_for_user(user_idx, model_state, n_items)

    results = []
    for asin, emb_idx in asin2idx.items():
        if asin not in item2idx:
            continue
        i_idx      = item2idx[asin]
        cf_raw     = float(cf[i_idx])
        cf_min, cf_max = cf.min(), cf.max()
        cf_norm    = (cf_raw - cf_min) / (cf_max - cf_min + 1e-8)
        sem_norm   = max(0.0, float(normed[emb_idx] @ q))
        hybrid     = alpha * cf_norm + (1 - alpha) * sem_norm
        results.append((asin, hybrid, cf_raw, sem_norm))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


# ── Rendering ─────────────────────────────────────────────────────────────────

def render_book(asin: str, caption: str, rank: int | None, metadata: dict) -> None:
    info      = metadata.get(asin, {})
    title     = info.get("title") or asin
    author    = info.get("author", "")
    cover_url = info.get("cover_url", "")
    col_img, col_txt = st.columns([1, 6])
    with col_img:
        if cover_url:
            st.image(cover_url, width=60)
        else:
            st.markdown("📖")
    with col_txt:
        label = f"**{rank+1}. {title}**" if rank is not None else f"**{title}**"
        st.markdown(label)
        prefix = f"{author} · " if author else ""
        st.caption(f"{prefix}{caption}")


# ── Load everything ───────────────────────────────────────────────────────────

data        = load_data()
model_state = load_model(data["n_users"], data["n_items"])
embeddings, asin2idx = load_embeddings()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    if model_state is None:
        st.error("No model found. Run: `python scripts/pretrain_model.py`")
    elif model_state["type"] == "bpr":
        st.success("BPR model loaded", icon="⚡")
        ckpt = model_state["ckpt"]
        col1, col2 = st.columns(2)
        with col1:
            st.metric("NDCG@10",   f"{ckpt.get('ndcg_10', 0):.3f}")
        with col2:
            st.metric("Recall@10", f"{ckpt.get('recall_10', 0):.3f}")
        st.metric("Latent Factors (k)", ckpt.get("k", "—"))
    else:
        st.info("Legacy MF model loaded")

    st.markdown("---")
    source_label = "Amazon 2023" if data["source"] == "amazon_2023" else "Legacy (Kaggle)"
    st.caption(f"Data source: {source_label}")
    st.metric("Users",  f"{data['n_users']:,}")
    st.metric("Items",  f"{data['n_items']:,}")
    st.metric("Ratings", f"{len(data['ratings']):,}")

    st.markdown("---")
    top_k = st.slider("Recommendations to show", 3, 20, 10)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📚 Book Recommender System")
st.markdown(
    "Hybrid recommendations powered by **BPR ranking** and **semantic search** "
    "on Amazon Reviews 2023."
)

if model_state is None:
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📖 For You", "🔍 Search by Vibe", "🔀 Hybrid"])

# ── Tab 1: Collaborative Filtering ───────────────────────────────────────────
with tab1:
    users       = sorted(data["user2idx"].keys())
    sel_user    = st.selectbox("Select a user:", options=users, index=0)
    user_idx    = data["user2idx"][sel_user]

    rated_items = set(
        data["ratings"][data["ratings"]["user_idx"] == user_idx]["item_idx"].tolist()
    )
    scores = cf_scores_for_user(user_idx, model_state, data["n_items"])

    unrated = sorted(
        [(i, scores[i]) for i in range(data["n_items"]) if i not in rated_items],
        key=lambda x: x[1], reverse=True,
    )

    st.markdown("### Top Recommendations")
    for rank, (i_idx, score) in enumerate(unrated[:top_k]):
        asin    = data["idx2item"][i_idx]
        caption = f"Score: {score:.3f}"
        render_book(asin, caption, rank, data["metadata"])

    st.markdown("### Previously Rated")
    rated_df = data["ratings"][data["ratings"]["user_idx"] == user_idx]
    if len(rated_df):
        for _, row in rated_df.head(8).iterrows():
            asin   = data["idx2item"].get(int(row["item_idx"]), "")
            rating = int(row.get("rating", 0))
            render_book(asin, "⭐" * rating, None, data["metadata"])
    else:
        st.info("No rated items found for this user.")

# ── Tab 2: Semantic Search ────────────────────────────────────────────────────
with tab2:
    st.subheader("Search by Vibe")
    st.caption("Describe what you're looking for — the model finds semantically similar books.")

    if embeddings is None:
        st.error("Embeddings not found. Run: `python scripts/generate_embeddings.py`")
    else:
        query = st.text_input("", placeholder="e.g. dark romance with a strong female lead", key="sem_q")
        if query:
            with st.spinner("Searching …"):
                results = semantic_results(query, embeddings, asin2idx, top_k)
            st.markdown(f"### Results for *\"{query}\"*")
            for rank, (asin, score) in enumerate(results):
                render_book(asin, f"Similarity: {score:.3f}", rank, data["metadata"])
        else:
            st.info("Enter a description above to search.")

# ── Tab 3: Hybrid ─────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Hybrid Recommendations")
    st.caption("Blends your taste profile (CF) with description matching (semantic).")

    if embeddings is None:
        st.error("Embeddings not found. Run: `python scripts/generate_embeddings.py`")
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            h_user  = st.selectbox("Select a user:", options=sorted(data["user2idx"].keys()),
                                   index=0, key="h_user")
        with col_b:
            h_query = st.text_input("Describe what you're looking for:",
                                    placeholder="e.g. spy thriller set in Cold War Berlin",
                                    key="h_query")

        alpha = st.slider(
            "Blend: Personal taste ← → Description match",
            min_value=0.0, max_value=1.0, value=0.5, step=0.05,
            help="1.0 = pure CF (ignores query)  |  0.0 = pure semantic (ignores user taste)",
        )

        if h_query:
            h_user_idx = data["user2idx"][h_user]
            with st.spinner("Finding hybrid recommendations …"):
                h_results = hybrid_results(
                    h_query, h_user_idx, model_state, data["n_items"],
                    embeddings, asin2idx, data["item2idx"], alpha, top_k,
                )
            st.markdown(f"### Results for *\"{h_query}\"* · User {h_user}")
            for rank, (asin, hy, cf_s, sem_s) in enumerate(h_results):
                render_book(asin, f"Hybrid: {hy:.3f}  CF: {cf_s:.2f}  Sem: {sem_s:.3f}",
                            rank, data["metadata"])
        else:
            st.info("Enter a description above to get hybrid recommendations.")

st.markdown("---")
st.caption("BPR · Semantic Embeddings · PyTorch · Streamlit · Amazon Reviews 2023")
