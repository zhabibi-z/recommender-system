"""
Download and cache the Amazon Reviews 2023 — Books dataset.

Streams directly from the HuggingFace hub filesystem (plain JSONL) rather than
using the dataset loading script, which is incompatible with datasets >= 4.0.
Only the sampled rows are written to disk; re-running is a no-op if cache exists.
"""
import json
import logging

import pandas as pd
from huggingface_hub import HfFileSystem

from src.config import cfg

log = logging.getLogger(__name__)

_HF_REPO      = "McAuley-Lab/Amazon-Reviews-2023"
_REVIEW_PATH  = f"datasets/{_HF_REPO}/raw/review_categories/Books.jsonl"
_META_PATH    = f"datasets/{_HF_REPO}/raw/meta_categories/meta_Books.jsonl"


def _iter_jsonl(fs: HfFileSystem, path: str):
    with fs.open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def download_reviews(max_rows: int = cfg.data.sample_size) -> pd.DataFrame:
    out = cfg.data.raw_dir / "reviews_raw.parquet"
    if out.exists():
        log.info("Reviews cache found — loading %s", out)
        return pd.read_parquet(out)

    log.info("Streaming reviews from %s (max %d rows) …", _REVIEW_PATH, max_rows)
    fs   = HfFileSystem()
    rows: list[dict] = []
    for i, rec in enumerate(_iter_jsonl(fs, _REVIEW_PATH)):
        if i >= max_rows:
            break
        rows.append({
            "user_id":   rec["user_id"],
            "asin":      rec["parent_asin"],
            "rating":    float(rec["rating"]),
            "timestamp": int(rec["timestamp"]),
        })
        if (i + 1) % 50_000 == 0:
            log.info("  streamed %d / %d", i + 1, max_rows)

    df = pd.DataFrame(rows)
    df.to_parquet(out, index=False)
    log.info("Saved %d reviews → %s", len(df), out)
    return df


def download_metadata(asins: set[str]) -> pd.DataFrame:
    out = cfg.data.raw_dir / "metadata_raw.parquet"
    if out.exists():
        log.info("Metadata cache found — loading %s", out)
        return pd.read_parquet(out)

    log.info("Streaming metadata for %d unique ASINs …", len(asins))
    fs   = HfFileSystem()
    rows: list[dict] = []
    for i, rec in enumerate(_iter_jsonl(fs, _META_PATH)):
        asin = rec.get("parent_asin", "")
        if asin not in asins:
            continue

        # `description` is either a list of strings or a plain string
        raw_desc    = rec.get("description") or []
        description = " ".join(raw_desc).strip() if isinstance(raw_desc, list) else str(raw_desc).strip()

        cats      = rec.get("categories") or []
        images    = rec.get("images") or []
        cover_url = ""
        if images and isinstance(images[0], dict):
            cover_url = images[0].get("large") or images[0].get("thumb") or ""

        raw_price = rec.get("price")
        try:
            price = float(raw_price) if raw_price not in (None, "", "—") else None
        except (ValueError, TypeError):
            price = None

        rows.append({
            "asin":          asin,
            "title":         (rec.get("title") or "").strip(),
            "description":   description,
            "price":         price,
            "genre":         cats[0] if cats else "",
            "cover_url":     cover_url,
            "avg_rating":    rec.get("average_rating"),
            "rating_number": rec.get("rating_number"),
        })

        if (i + 1) % 100_000 == 0:
            log.info("  scanned %d metadata rows, collected %d matches", i + 1, len(rows))

    df = pd.DataFrame(rows).drop_duplicates("asin").reset_index(drop=True)
    df.to_parquet(out, index=False)
    log.info("Saved metadata for %d books → %s", len(df), out)
    return df


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    reviews  = download_reviews()
    metadata = download_metadata(set(reviews["asin"].unique()))
    return reviews, metadata


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    reviews, metadata = run()
    print(f"\nReviews  : {reviews.shape}")
    print(f"Metadata : {metadata.shape}")
    print(f"Desc coverage : {(metadata['description'].str.len() > 50).mean():.1%}")
