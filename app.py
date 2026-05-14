import streamlit as st
import pandas as pd
import numpy as np
import torch
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="📚 Book Recommender System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📚 Book Recommender System")
st.markdown("""
Collaborative filtering-based recommendations using **matrix factorization**.
Built with PyTorch and trained on Amazon book reviews.
""")

# ============================================================================
# LOAD DATA & TRAIN MODEL (CACHED)
# ============================================================================
@st.cache_data
def load_and_prepare_data():
    """Load training and test data"""
    training = pd.read_csv("MATH_Final_Project_Data_training.csv")
    test_features = pd.read_csv("MATH_Final_Project_Data_test_features.csv")

    unique_users = sorted(training["User"].unique().tolist())
    unique_asins = sorted(training["ASIN"].unique().tolist())

    metadata = {}
    if Path("book_metadata.csv").exists():
        meta_df = pd.read_csv("book_metadata.csv", dtype=str).fillna("")
        for _, row in meta_df.iterrows():
            metadata[row["asin"]] = {
                "title": row["title"],
                "author": row["author"],
                "cover_url": row["cover_url"],
            }

    return training, test_features, unique_users, unique_asins, metadata


def book_display_name(asin, metadata):
    info = metadata.get(asin, {})
    title = info.get("title", "")
    return title if title else asin

@st.cache_data
def create_matrices(df, unique_users, unique_asins):
    """Create score and mask matrices"""
    pivot = df.pivot_table(index="User", columns="ASIN", values="Rating", fill_value=0)
    pivot = pivot.reindex(index=unique_users, columns=unique_asins, fill_value=0)
    S = pivot.values.astype(float)
    R = (S != 0).astype(float)
    return S, R

@st.cache_resource
def train_model(S_train, R_train, S_val, R_val, k=15, lr=0.1, n_steps=300, lambda_reg=0.001):
    """Train the matrix factorization model with bias terms and early stopping"""
    n_users, n_asins = S_train.shape

    S_train_t = torch.tensor(S_train, dtype=torch.float32)
    R_train_t = torch.tensor(R_train, dtype=torch.float32)
    S_val_t   = torch.tensor(S_val,   dtype=torch.float32)
    R_val_t   = torch.tensor(R_val,   dtype=torch.float32)

    N_train_count = R_train_t.sum().item()
    N_val_count   = R_val_t.sum().item()

    # Global mean over observed training ratings
    mu = (S_train_t * R_train_t).sum() / N_train_count

    torch.manual_seed(42)
    A      = torch.randn(n_users, k, requires_grad=True)
    F      = torch.randn(k, n_asins, requires_grad=True)
    b_u    = torch.zeros(n_users, 1, requires_grad=True)   # user bias
    b_i    = torch.zeros(1, n_asins, requires_grad=True)   # item bias

    best_val_loss = float("inf")
    best_A, best_F, best_bu, best_bi = None, None, None, None
    patience_counter = 0
    patience = 15

    progress_bar = st.progress(0)
    status_text = st.empty()

    for step in range(n_steps):
        P = mu + b_u + b_i + A @ F

        loss = torch.sum(R_train_t * (S_train_t - P) ** 2) / N_train_count
        reg_loss = lambda_reg * (torch.sum(A ** 2) + torch.sum(F ** 2) +
                                 torch.sum(b_u ** 2) + torch.sum(b_i ** 2))
        total_loss = loss + reg_loss

        total_loss.backward()

        with torch.no_grad():
            A   -= lr * A.grad;   A.grad.zero_()
            F   -= lr * F.grad;   F.grad.zero_()
            b_u -= lr * b_u.grad; b_u.grad.zero_()
            b_i -= lr * b_i.grad; b_i.grad.zero_()

        with torch.no_grad():
            P_val = mu + b_u + b_i + A @ F
            val_loss = torch.sum(R_val_t * (S_val_t - P_val) ** 2) / N_val_count

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_A  = A.detach().clone()
            best_F  = F.detach().clone()
            best_bu = b_u.detach().clone()
            best_bi = b_i.detach().clone()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                status_text.text(f"✓ Training complete (early stopped at step {step+1})")
                break

        progress_bar.progress(min((step + 1) / n_steps, 1.0))
        if (step + 1) % 50 == 0:
            status_text.text(f"Step {step+1}/{n_steps} | Val MSE: {best_val_loss:.4f}")

    progress_bar.empty()
    status_text.empty()

    return best_A, best_F, best_bu, best_bi, mu.item()

# ============================================================================
# SEMANTIC SEARCH HELPERS
# ============================================================================

@st.cache_resource
def load_embeddings():
    if not Path("book_embeddings.npy").exists():
        return None, None
    embeddings = np.load("book_embeddings.npy")
    index_df = pd.read_csv("book_embeddings_index.csv", dtype=str)
    asin_to_idx = {row["asin"]: int(row["idx"]) for _, row in index_df.iterrows()}
    return embeddings, asin_to_idx


def semantic_search(query, embeddings, asin_to_idx, unique_asins, metadata, top_k=10):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_vec = model.encode([query])[0]
    # cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / np.where(norms == 0, 1, norms)
    query_norm = query_vec / (np.linalg.norm(query_vec) or 1)
    scores = normed @ query_norm
    # only score books that are in our catalog
    catalog_indices = [asin_to_idx[a] for a in unique_asins if a in asin_to_idx]
    catalog_asins   = [a for a in unique_asins if a in asin_to_idx]
    catalog_scores  = scores[catalog_indices]
    top_order = np.argsort(catalog_scores)[::-1][:top_k]
    results = [(catalog_asins[i], float(catalog_scores[i])) for i in top_order]
    return results


# ============================================================================
# MAIN APP
# ============================================================================

# Load data
training, test_features, unique_users, unique_asins, metadata = load_and_prepare_data()

# Create train/val split
from sklearn.model_selection import train_test_split
train, val = train_test_split(training, random_state=42, stratify=training["ASIN"])

S_train, R_train = create_matrices(train, unique_users, unique_asins)
S_val, R_val = create_matrices(val, unique_users, unique_asins)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    k = st.slider("Number of Latent Factors (k)", 2, 20, 15)
    top_k = st.slider("Number of Recommendations", 3, 20, 10)
    st.markdown("---")
    st.subheader("Model Info")
    st.metric("Users", len(unique_users))
    st.metric("Books", len(unique_asins))
    st.metric("Training Ratings", int(R_train.sum()))
    st.metric("Sparsity", f"{(1 - R_train.mean()) * 100:.1f}%")

# Train model
st.subheader("🤖 Training Model...")
A, F, b_u, b_i, mu = train_model(S_train, R_train, S_val, R_val, k=k, lr=0.1, n_steps=300, lambda_reg=0.001)

# Generate predictions
P = (mu + b_u + b_i + A @ F).detach().numpy()
P_clamped = np.clip(P, 1, 5)

# Load embeddings for semantic search
embeddings, asin_to_idx = load_embeddings()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📖 For You  (Collaborative)", "🔍 Search by Vibe  (Semantic)"])


def render_book(asin, caption, idx=None):
    info = metadata.get(asin, {})
    title = info.get("title") or asin
    author = info.get("author", "")
    cover_url = info.get("cover_url", "")
    col_cover, col_text = st.columns([1, 6])
    with col_cover:
        if cover_url:
            st.image(cover_url, width=60)
        else:
            st.markdown("📖")
    with col_text:
        label = f"**{idx+1}. {title}**" if idx is not None else f"**{title}**"
        st.markdown(label)
        st.caption(f"{author + ' · ' if author else ''}{caption}")


# ── Tab 1: Collaborative Filtering ───────────────────────────────────────────
with tab1:
    st.subheader("📖 Get Recommendations")
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_user = st.selectbox("Select a User ID:", options=unique_users, index=0)
    with col2:
        st.metric("User", selected_user)

    user_idx = unique_users.index(selected_user)
    user_predictions = P_clamped[user_idx]
    user_rated_indices = set(np.where(R_train[user_idx] > 0)[0])
    user_rated_asins = [unique_asins[i] for i in user_rated_indices]

    unrated_indices = [i for i in range(len(unique_asins)) if i not in user_rated_indices]
    unrated_predictions = sorted([(i, user_predictions[i]) for i in unrated_indices],
                                 key=lambda x: x[1], reverse=True)
    top_recommendations = unrated_predictions[:top_k]

    st.markdown("### ⭐ Top Recommendations (Unrated Books)")
    for i, (idx, pred) in enumerate(top_recommendations):
        render_book(unique_asins[idx], f"Predicted: {pred:.2f} ⭐", idx=i)

    st.markdown("### 📚 User's Rated Books")
    if user_rated_asins:
        for asin in user_rated_asins[:10]:
            rating = int(S_train[user_idx, unique_asins.index(asin)])
            render_book(asin, f"Rated: {'⭐' * rating}")
        if len(user_rated_asins) > 10:
            st.caption(f"...and {len(user_rated_asins) - 10} more books")
    else:
        st.info("No rated books found for this user.")

    st.markdown("---")
    st.subheader("📊 Model Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        train_rmse = np.sqrt(np.mean((P_clamped[R_train > 0] - S_train[R_train > 0]) ** 2))
        st.metric("Train RMSE", f"{train_rmse:.3f}")
    with col2:
        val_rmse = np.sqrt(np.mean((P_clamped[R_val > 0] - S_val[R_val > 0]) ** 2))
        st.metric("Validation RMSE", f"{val_rmse:.3f}")
    with col3:
        st.metric("Parameters", f"{k * (len(unique_users) + len(unique_asins)):,}")
    with col4:
        st.metric("Model Size", f"k={k}")


# ── Tab 2: Semantic Search ────────────────────────────────────────────────────
with tab2:
    st.subheader("🔍 Search by Vibe")
    st.markdown("Describe what you're looking for and we'll find the closest matches.")

    if embeddings is None:
        st.error("Embeddings file not found. Run `generate_embeddings.py` first.")
    else:
        query = st.text_input(
            "What kind of book are you looking for?",
            placeholder="e.g. dark romance with strong female lead",
        )
        if query:
            with st.spinner("Searching..."):
                results = semantic_search(query, embeddings, asin_to_idx,
                                          unique_asins, metadata, top_k=top_k)
            st.markdown(f"### Top {top_k} matches for: *\"{query}\"*")
            for i, (asin, score) in enumerate(results):
                render_book(asin, f"Similarity: {score:.2f}", idx=i)
        else:
            st.info("Type a description above to find matching books.")


# Footer
st.markdown("---")
st.markdown("""
**Built with:** PyTorch • Streamlit • Pandas • sentence-transformers
**Algorithm:** Matrix Factorization (collaborative) + Semantic Embeddings (content-based)
**Loss:** Mean Squared Error with L2 Regularization
""")
