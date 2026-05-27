"""
Encode book descriptions as sentence embeddings for semantic search.

Text is constructed as "{title}. {genre}. {description}" to give the encoder
richer context than the description alone. Prefers Amazon Reviews 2023 metadata
(data/raw/metadata_raw.parquet) and falls back to legacy book_descriptions.csv.
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.config import cfg, ROOT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_EMB_DIR        = ROOT / "data" / "embeddings"
EMBEDDINGS_FILE = _EMB_DIR / "book_embeddings.npy"
INDEX_FILE      = _EMB_DIR / "book_embeddings_index.csv"


def load_metadata() -> pd.DataFrame:
    new_source = cfg.data.raw_dir / "metadata_raw.parquet"
    if new_source.exists():
        log.info("Loading Amazon Reviews 2023 metadata from %s", new_source)
        df = pd.read_parquet(new_source)
        df["text"] = (
            df["title"].fillna("") + ". "
            + df["genre"].fillna("") + ". "
            + df["description"].fillna("")
        ).str.strip()
        return df[["asin", "title", "text"]]

    legacy = ROOT / "data" / "legacy" / "book_descriptions.csv"
    if legacy.exists():
        log.info("Falling back to legacy book_descriptions.csv")
        return pd.read_csv(legacy, dtype=str).fillna("")[["asin", "title", "text"]]

    log.error("No metadata source found — run run_pipeline.py first.")
    sys.exit(1)


def main() -> None:
    df         = load_metadata()
    embeddable = df[df["text"].str.strip().str.len() > 20].reset_index(drop=True)
    log.info(
        "Total books: %d  |  Embeddable: %d  |  Skipped: %d",
        len(df), len(embeddable), len(df) - len(embeddable),
    )

    encoder    = SentenceTransformer(cfg.embeddings.model_name)
    embeddings = encoder.encode(
        embeddable["text"].tolist(),
        batch_size=cfg.embeddings.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    _EMB_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_FILE, embeddings.astype(np.float32))
    pd.DataFrame({"asin": embeddable["asin"].values, "idx": np.arange(len(embeddable))}).to_csv(
        INDEX_FILE, index=False
    )

    log.info("Saved embeddings → %s  shape: %s", EMBEDDINGS_FILE, embeddings.shape)
    log.info("Saved index      → %s", INDEX_FILE)


if __name__ == "__main__":
    main()
