import streamlit as st
import pandas as pd
import numpy as np
import torch
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Book Recommender",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📚 Book Recommender System")
st.markdown("Hybrid recommendations powered by **matrix factorization** and **semantic search**.")


@st.cache_data
def load_data():
    training = pd.read_csv("training.csv")
    test_features = pd.read_csv("test_features.csv")
    unique_users = sorted(training["User"].unique().tolist())
    unique_asins = sorted(training["ASIN"].unique().tolist())

    metadata = {}
    if Path("book_metadata.csv").exists():
        meta_df = pd.read_csv("book_metadata.csv", dtype=str).fillna("")
        for _, row in meta_df.iterrows():
            asin = row["asin"]
            cover_url = row["cover_url"]
            if not cover_url and not asin.startswith("B"):
                cover_url = f"https://covers.openlibrary.org/b/isbn/{asin}-M.jpg"
            metadata[asin] = {
                "title": row["title"],
                "author": row["author"],
                "cover_url": cover_url,
            }

    return training, test_features, unique_users, unique_asins, metadata


@st.cache_data
def build_matrices(df, unique_users, unique_asins):
    pivot = df.pivot_table(index="User", columns="ASIN", values="Rating", fill_value=0)
    pivot = pivot.reindex(index=unique_users, columns=unique_asins, fill_value=0)
    S = pivot.values.astype(float)
    R = (S != 0).astype(float)
    return S, R


@st.cache_resource
def train_model(S_train, R_train, S_val, R_val, k=15, lr=0.1, n_steps=300, lambda_reg=0.001):
    n_users, n_items = S_train.shape

    S_tr = torch.tensor(S_train, dtype=torch.float32)
    R_tr = torch.tensor(R_train, dtype=torch.float32)
    S_v  = torch.tensor(S_val,   dtype=torch.float32)
    R_v  = torch.tensor(R_val,   dtype=torch.float32)

    N_tr = R_tr.sum().item()
    N_v  = R_v.sum().item()
    mu   = (S_tr * R_tr).sum() / N_tr

    torch.manual_seed(42)
    A   = torch.randn(n_users, k, requires_grad=True)
    F   = torch.randn(k, n_items, requires_grad=True)
    b_u = torch.zeros(n_users, 1, requires_grad=True)
    b_i = torch.zeros(1, n_items, requires_grad=True)

    best_val, best_A, best_F, best_bu, best_bi = float("inf"), None, None, None, None
    patience_counter, patience = 0, 15

    bar  = st.progress(0)
    info = st.empty()

    for step in range(n_steps):
        P = mu + b_u + b_i + A @ F
        loss = torch.sum(R_tr * (S_tr - P) ** 2) / N_tr
        reg  = lambda_reg * (torch.sum(A**2) + torch.sum(F**2) + torch.sum(b_u**2) + torch.sum(b_i**2))
        (loss + reg).backward()

        with torch.no_grad():
            A   -= lr * A.grad;   A.grad.zero_()
            F   -= lr * F.grad;   F.grad.zero_()
            b_u -= lr * b_u.grad; b_u.grad.zero_()
            b_i -= lr * b_i.grad; b_i.grad.zero_()

        with torch.no_grad():
            val_loss = (torch.sum(R_v * (S_v - (mu + b_u + b_i + A @ F)) ** 2) / N_v).item()

        if val_loss < best_val:
            best_val = val_loss
            best_A, best_F, best_bu, best_bi = A.detach().clone(), F.detach().clone(), b_u.detach().clone(), b_i.detach().clone()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                info.text(f"Early stopping at step {step + 1}")
                break

        bar.progress(min((step + 1) / n_steps, 1.0))
        if (step + 1) % 50 == 0:
            info.text(f"Step {step+1}/{n_steps} · Val MSE: {best_val:.4f}")

    bar.empty()
    info.empty()
    return best_A, best_F, best_bu, best_bi, mu.item()


@st.cache_resource
def load_embeddings():
    if not Path("book_embeddings.npy").exists():
        return None, None
    embeddings = np.load("book_embeddings.npy")
    index_df = pd.read_csv("book_embeddings_index.csv", dtype=str)
    asin_to_idx = {row["asin"]: int(row["idx"]) for _, row in index_df.iterrows()}
    return embeddings, asin_to_idx


@st.cache_resource
def load_encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def semantic_search(query, embeddings, asin_to_idx, unique_asins, top_k=10):
    encoder = load_encoder()
    q = encoder.encode([query])[0]
    norms  = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / np.where(norms == 0, 1, norms)
    q_norm = q / (np.linalg.norm(q) or 1)
    scores = normed @ q_norm

    catalog_idx   = [asin_to_idx[a] for a in unique_asins if a in asin_to_idx]
    catalog_asins = [a for a in unique_asins if a in asin_to_idx]
    top = np.argsort(scores[catalog_idx])[::-1][:top_k]
    return [(catalog_asins[i], float(scores[catalog_idx[i]])) for i in top]


def render_book(asin, caption, idx=None):
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
        st.markdown(f"**{idx+1}. {title}**" if idx is not None else f"**{title}**")
        st.caption(f"{author + ' · ' if author else ''}{caption}")


# ── Data & model ──────────────────────────────────────────────────────────────
training, test_features, unique_users, unique_asins, metadata = load_data()

train_df, val_df = train_test_split(training, random_state=42, stratify=training["ASIN"])
S_train, R_train = build_matrices(train_df, unique_users, unique_asins)
S_val,   R_val   = build_matrices(val_df,   unique_users, unique_asins)

with st.sidebar:
    st.header("Settings")
    k      = st.slider("Latent Factors (k)", 2, 20, 15)
    top_k  = st.slider("Recommendations", 3, 20, 10)
    st.markdown("---")
    st.subheader("Dataset")
    st.metric("Users",    len(unique_users))
    st.metric("Books",    len(unique_asins))
    st.metric("Ratings",  int(R_train.sum()))
    st.metric("Sparsity", f"{(1 - R_train.mean()) * 100:.1f}%")

with st.spinner("Training model..."):
    A, F, b_u, b_i, mu = train_model(S_train, R_train, S_val, R_val, k=k, lr=0.1, n_steps=300, lambda_reg=0.001)

P = np.clip((mu + b_u + b_i + A @ F).detach().numpy(), 1, 5)
embeddings, asin_to_idx = load_embeddings()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📖 For You", "🔍 Search by Vibe"])

with tab1:
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_user = st.selectbox("Select a user:", options=unique_users, index=0)
    with col2:
        st.metric("User", selected_user)

    user_idx         = unique_users.index(selected_user)
    rated_indices    = set(np.where(R_train[user_idx] > 0)[0])
    rated_asins      = [unique_asins[i] for i in rated_indices]
    unrated_preds    = sorted(
        [(i, P[user_idx][i]) for i in range(len(unique_asins)) if i not in rated_indices],
        key=lambda x: x[1], reverse=True
    )

    st.markdown("### Top Recommendations")
    for i, (idx, pred) in enumerate(unrated_preds[:top_k]):
        render_book(unique_asins[idx], f"Predicted rating: {pred:.2f} ⭐", idx=i)

    st.markdown("### Previously Rated")
    if rated_asins:
        for asin in rated_asins[:10]:
            rating = int(S_train[user_idx, unique_asins.index(asin)])
            render_book(asin, f"{'⭐' * rating}")
        if len(rated_asins) > 10:
            st.caption(f"...and {len(rated_asins) - 10} more")
    else:
        st.info("No ratings found for this user.")

    st.markdown("---")
    st.subheader("Model Performance")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Train RMSE", f"{np.sqrt(np.mean((P[R_train > 0] - S_train[R_train > 0])**2)):.3f}")
    with c2:
        st.metric("Val RMSE", f"{np.sqrt(np.mean((P[R_val > 0] - S_val[R_val > 0])**2)):.3f}")
    with c3:
        st.metric("Parameters", f"{k * (len(unique_users) + len(unique_asins)):,}")
    with c4:
        st.metric("k", k)

with tab2:
    st.subheader("Search by Vibe")
    st.caption("Describe what you're looking for and the model finds semantically similar books.")

    if embeddings is None:
        st.error("Embeddings not found.")
    else:
        query = st.text_input("", placeholder="e.g. dark romance with a strong female lead")
        if query:
            with st.spinner("Searching..."):
                results = semantic_search(query, embeddings, asin_to_idx, unique_asins, top_k=top_k)
            st.markdown(f"### Results for *\"{query}\"*")
            for i, (asin, score) in enumerate(results):
                render_book(asin, f"Similarity: {score:.2f}", idx=i)
        else:
            st.info("Enter a description above to search.")

st.markdown("---")
st.caption("Matrix Factorization · Semantic Embeddings · PyTorch · Streamlit")
