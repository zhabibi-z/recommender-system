"""
Full data + training pipeline — run this once from scratch.

Steps
-----
  1. Download Amazon Reviews 2023 Books dataset (streaming, resumable)
  2. Clean: deduplicate, k-core filter, temporal split
  3. Feature engineering: user/item features
  4. Generate sentence embeddings for semantic search
  5. Train BPR model and evaluate

Usage
-----
    python run_pipeline.py              # full pipeline
    python run_pipeline.py --skip-data  # skip download if data already cached
"""
import argparse
import logging
import subprocess
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def banner(title: str) -> None:
    log.info("─" * 60)
    log.info("  %s", title)
    log.info("─" * 60)


def main(skip_data: bool = False) -> None:
    t0 = time.time()

    # ── Step 1 & 2: Download + Clean ─────────────────────────────────────────
    if not skip_data:
        banner("Step 1/5 — Download Amazon Reviews 2023")
        from src.data.download import run as download
        reviews, metadata = download()
        log.info("Downloaded %d reviews, %d books", len(reviews), len(metadata))

        banner("Step 2/5 — Clean & Split")
        from src.data.clean import run as clean
        splits = clean(reviews)
        for name, df in splits.items():
            log.info("  %-10s %d rows", name, len(df))
    else:
        banner("Step 1–2 — Skipped (--skip-data)")
        import pandas as pd
        from src.config import cfg
        reviews  = pd.read_parquet(cfg.data.raw_dir  / "reviews_raw.parquet")
        metadata = pd.read_parquet(cfg.data.raw_dir  / "metadata_raw.parquet")

    # ── Step 3: Features ──────────────────────────────────────────────────────
    banner("Step 3/5 — Feature Engineering")
    from src.data.features import run as build_features
    user_feats, item_feats = build_features()
    log.info("User features: %s", user_feats.shape)
    log.info("Item features: %s", item_feats.shape)

    # ── Step 4: Embeddings ────────────────────────────────────────────────────
    banner("Step 4/5 — Generate Sentence Embeddings")
    result = subprocess.run(
        [sys.executable, "scripts/generate_embeddings.py"],
        capture_output=False,
    )
    if result.returncode != 0:
        log.error("Embedding generation failed")
        sys.exit(1)

    # ── Step 5: Train BPR ─────────────────────────────────────────────────────
    banner("Step 5/5 — Train BPR Model")
    result = subprocess.run(
        [sys.executable, "scripts/pretrain_model.py"],
        capture_output=False,
    )
    if result.returncode != 0:
        log.error("Model training failed")
        sys.exit(1)

    elapsed = time.time() - t0
    banner(f"Pipeline complete in {elapsed/60:.1f} min")
    log.info("Launch the app:  streamlit run app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-data", action="store_true",
                        help="Skip download/clean if data is already cached")
    args = parser.parse_args()
    main(skip_data=args.skip_data)
