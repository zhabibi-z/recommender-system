"""Train and save the BPR model. Requires data/processed/ — run run_pipeline.py first."""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import cfg
from src.models.bpr import train_bpr, save as save_bpr
from src.utils.metrics import evaluate_ranking

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main() -> None:
    processed = cfg.data.processed_dir
    required  = ["train.parquet", "val.parquet", "test.parquet", "ratings.parquet"]
    missing   = [f for f in required if not (processed / f).exists()]
    if missing:
        log.error("Missing processed files: %s — run: python run_pipeline.py", missing)
        sys.exit(1)

    train   = pd.read_parquet(processed / "train.parquet")
    val     = pd.read_parquet(processed / "val.parquet")
    test    = pd.read_parquet(processed / "test.parquet")
    ratings = pd.read_parquet(processed / "ratings.parquet")

    n_users = int(ratings["user_idx"].max()) + 1
    n_items = int(ratings["item_idx"].max()) + 1
    log.info("Dataset — %d users, %d items, %d train interactions", n_users, n_items, len(train))

    model = train_bpr(train, val, n_users, n_items)

    log.info("Evaluating on held-out test set …")
    metrics = evaluate_ranking(
        score_fn = model.score_all_items,
        test_df  = test,
        train_df = train,
        k_values = [5, 10, 20],
        n_sample = 500,
    )
    print("\nTest Metrics\n" + metrics.to_string() + "\n")

    out = cfg.models_dir / "bpr_model.pt"
    save_bpr(model, out, {
        "n_users":   n_users,
        "n_items":   n_items,
        "k":         cfg.model.k,
        "ndcg_10":   float(metrics.loc[10, "NDCG@K"]),
        "recall_10": float(metrics.loc[10, "Recall@K"]),
        "hit_10":    float(metrics.loc[10, "Hit@K"]),
    })


if __name__ == "__main__":
    main()
