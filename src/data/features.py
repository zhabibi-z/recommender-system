"""
Feature engineering for users and items.

User features: mean_rating, rating_count, rating_std, days_active, top_genre
Item features: mean_rating, rating_count, rating_std, description_len,
               has_description, genre, price_tier
"""
import logging

import pandas as pd

from src.config import cfg

log = logging.getLogger(__name__)


def build_user_features(
    train:    pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    asin_genre = metadata.set_index("asin")["genre"].fillna("Unknown").to_dict()
    enriched   = train.copy()
    enriched["genre"] = enriched["asin"].map(asin_genre).fillna("Unknown")

    feats = (
        enriched.groupby("user_idx")
        .agg(
            mean_rating  = ("rating", "mean"),
            rating_count = ("rating", "count"),
            rating_std   = ("rating", "std"),
        )
        .reset_index()
    )
    feats["rating_std"] = feats["rating_std"].fillna(0.0)

    span = enriched.groupby("user_idx")["timestamp"].agg(["min", "max"])
    feats["days_active"] = ((span["max"] - span["min"]) / 86_400).clip(lower=0).values

    top_genre = (
        enriched.groupby(["user_idx", "genre"])
        .size()
        .reset_index(name="cnt")
        .sort_values("cnt", ascending=False)
        .drop_duplicates("user_idx")
        .set_index("user_idx")["genre"]
    )
    feats["top_genre"] = feats["user_idx"].map(top_genre).fillna("Unknown")

    out = cfg.data.processed_dir / "user_features.parquet"
    feats.to_parquet(out, index=False)
    log.info("User features: %s → %s", feats.shape, out)
    return feats


def build_item_features(
    ratings:  pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    feats = (
        ratings.groupby("asin")
        .agg(
            mean_rating  = ("rating", "mean"),
            rating_count = ("rating", "count"),
            rating_std   = ("rating", "std"),
        )
        .reset_index()
    )
    feats["rating_std"] = feats["rating_std"].fillna(0.0)

    meta  = metadata[["asin", "description", "genre", "price", "cover_url"]].copy()
    feats = feats.merge(meta, on="asin", how="left")

    feats["description_len"] = feats["description"].fillna("").str.len()
    feats["has_description"] = feats["description_len"] > 50
    feats["genre"]            = feats["genre"].fillna("Unknown")
    feats["cover_url"]        = feats["cover_url"].fillna("")
    feats["price_tier"]       = feats["price"].apply(_price_tier)
    feats = feats.drop(columns=["description", "price"])

    out = cfg.data.processed_dir / "item_features.parquet"
    feats.to_parquet(out, index=False)
    log.info("Item features: %s → %s", feats.shape, out)
    return feats


def _price_tier(price) -> str:
    try:
        p = float(price)
        if p < 10:
            return "budget"
        if p < 25:
            return "mid"
        return "premium"
    except (TypeError, ValueError):
        return "unknown"


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    processed = cfg.data.processed_dir
    train    = pd.read_parquet(processed / "train.parquet")
    ratings  = pd.read_parquet(processed / "ratings.parquet")
    metadata = pd.read_parquet(cfg.data.raw_dir / "metadata_raw.parquet")
    return build_user_features(train, metadata), build_item_features(ratings, metadata)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    uf, itf = run()
    print(f"User features : {uf.shape}")
    print(f"Item features : {itf.shape}")
