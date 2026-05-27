"""
Data cleaning pipeline: deduplication → k-core filtering → ID encoding → temporal split.

Temporal splitting is done per-user (most-recent interactions held out) rather than
by a global time cut, so every user appears in every split and evaluation is stable.
"""
import logging

import numpy as np
import pandas as pd

from src.config import cfg

log = logging.getLogger(__name__)


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = (
        df.sort_values("timestamp")
          .drop_duplicates(subset=["user_id", "asin"], keep="last")
          .reset_index(drop=True)
    )
    removed = before - len(df)
    if removed:
        log.info("Removed %d duplicate (user, item) pairs", removed)
    return df


def kcore_filter(
    df: pd.DataFrame,
    min_user: int = cfg.data.min_user_ratings,
    min_item: int = cfg.data.min_item_ratings,
) -> pd.DataFrame:
    """
    Iteratively prune until every user has ≥ min_user interactions and every
    item has ≥ min_item interactions.

    A single pass is insufficient: removing low-activity items can drop some
    users below the threshold, and vice versa.
    """
    iteration = 0
    while True:
        user_counts = df["user_id"].value_counts()
        item_counts = df["asin"].value_counts()
        valid_users = user_counts[user_counts >= min_user].index
        valid_items = item_counts[item_counts >= min_item].index
        filtered = df[df["user_id"].isin(valid_users) & df["asin"].isin(valid_items)]
        iteration += 1
        if len(filtered) == len(df):
            break
        log.info(
            "k-core iter %d: %d → %d interactions (%d users, %d items)",
            iteration, len(df), len(filtered),
            filtered["user_id"].nunique(), filtered["asin"].nunique(),
        )
        df = filtered

    log.info(
        "k-core done: %d interactions | %d users | %d items",
        len(df), df["user_id"].nunique(), df["asin"].nunique(),
    )
    return df.reset_index(drop=True)


def encode_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    users = sorted(df["user_id"].unique())
    items = sorted(df["asin"].unique())
    user2idx: dict[str, int] = {u: i for i, u in enumerate(users)}
    item2idx: dict[str, int] = {a: i for i, a in enumerate(items)}
    df = df.copy()
    df["user_idx"] = df["user_id"].map(user2idx).astype(np.int32)
    df["item_idx"] = df["asin"].map(item2idx).astype(np.int32)
    log.info("Encoded %d users × %d items", len(users), len(items))
    return df, user2idx, item2idx


def temporal_split(
    df:        pd.DataFrame,
    val_frac:  float = cfg.data.val_frac,
    test_frac: float = cfg.data.test_frac,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    For each user, hold out the most recent `test_frac` interactions for test
    and the next `val_frac` for validation.

    Per-user splitting ensures every user is represented in every split,
    giving a stable and unbiased evaluation.
    """
    df = df.sort_values(["user_idx", "timestamp"])
    train_parts, val_parts, test_parts = [], [], []

    for _, grp in df.groupby("user_idx", sort=False):
        n       = len(grp)
        n_test  = max(1, int(round(n * test_frac)))
        n_val   = max(1, int(round(n * val_frac)))
        n_train = n - n_test - n_val
        if n_train < 1:
            train_parts.append(grp)
            continue
        train_parts.append(grp.iloc[:n_train])
        val_parts.append(grp.iloc[n_train : n_train + n_val])
        test_parts.append(grp.iloc[n_train + n_val :])

    train = pd.concat(train_parts).reset_index(drop=True)
    val   = pd.concat(val_parts).reset_index(drop=True)
    test  = pd.concat(test_parts).reset_index(drop=True)

    log.info("Temporal split — train: %d  val: %d  test: %d", len(train), len(val), len(test))
    return train, val, test


def run(reviews: pd.DataFrame) -> dict[str, pd.DataFrame]:
    df = remove_duplicates(reviews)
    df = kcore_filter(df)
    df, user2idx, item2idx = encode_ids(df)
    train, val, test = temporal_split(df)

    out = cfg.data.processed_dir
    df.to_parquet(out / "ratings.parquet",   index=False)
    train.to_parquet(out / "train.parquet",  index=False)
    val.to_parquet(out / "val.parquet",      index=False)
    test.to_parquet(out / "test.parquet",    index=False)

    pd.DataFrame(
        [{"user_id": u, "user_idx": i} for u, i in user2idx.items()]
    ).to_parquet(out / "user_index.parquet", index=False)

    pd.DataFrame(
        [{"asin": a, "item_idx": i} for a, i in item2idx.items()]
    ).to_parquet(out / "item_index.parquet", index=False)

    log.info("Processed files saved to %s", out)
    return {"ratings": df, "train": train, "val": val, "test": test}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    from src.data.download import run as download
    reviews, _ = download()
    splits = run(reviews)
    for name, split in splits.items():
        print(f"{name:8s}: {len(split):>8,} rows")
