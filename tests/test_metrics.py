"""
Unit tests for src/utils/metrics.py

Run:
    python -m pytest tests/ -v
"""
import numpy as np
import pandas as pd
import pytest

from src.utils.metrics import (
    rmse, mae,
    precision_at_k, recall_at_k, ndcg_at_k, hit_rate_at_k,
    evaluate_ranking,
)


# ── Pointwise ────────────────────────────────────────────────────────────────

def test_rmse_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == pytest.approx(0.0)


def test_rmse_known():
    y_true = np.array([3.0, 3.0, 3.0])
    y_pred = np.array([4.0, 4.0, 4.0])
    assert rmse(y_true, y_pred) == pytest.approx(1.0)


def test_mae_perfect():
    y = np.array([1.0, 5.0, 3.5])
    assert mae(y, y) == pytest.approx(0.0)


def test_mae_known():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 3.0, 4.0])
    assert mae(y_true, y_pred) == pytest.approx(1.0)


# ── Ranking metrics ───────────────────────────────────────────────────────────

def test_precision_perfect():
    recommended = [0, 1, 2, 3, 4]
    relevant    = {0, 1, 2, 3, 4}
    assert precision_at_k(recommended, relevant, 5) == pytest.approx(1.0)


def test_precision_none_relevant():
    recommended = [0, 1, 2]
    relevant    = {5, 6, 7}
    assert precision_at_k(recommended, relevant, 3) == pytest.approx(0.0)


def test_precision_partial():
    recommended = [0, 1, 2, 3, 4]
    relevant    = {0, 2, 4}
    assert precision_at_k(recommended, relevant, 5) == pytest.approx(3 / 5)


def test_recall_all_found():
    recommended = [0, 1, 2]
    relevant    = {0, 1}
    assert recall_at_k(recommended, relevant, 3) == pytest.approx(1.0)


def test_recall_empty_relevant():
    assert recall_at_k([0, 1, 2], set(), 3) == pytest.approx(0.0)


def test_ndcg_perfect():
    recommended = [0, 1, 2]
    relevant    = {0, 1, 2}
    assert ndcg_at_k(recommended, relevant, 3) == pytest.approx(1.0)


def test_ndcg_worst_order():
    # All relevant but in reverse-ideal order: still NDCG < 1
    recommended = [2, 1, 0]
    relevant    = {0}
    score = ndcg_at_k(recommended, relevant, 3)
    assert 0 < score < 1


def test_ndcg_no_relevant():
    recommended = [0, 1, 2]
    relevant    = {5, 6}
    assert ndcg_at_k(recommended, relevant, 3) == pytest.approx(0.0)


def test_hit_rate_hit():
    assert hit_rate_at_k([0, 1, 2], {2}, 3) == pytest.approx(1.0)


def test_hit_rate_miss():
    assert hit_rate_at_k([0, 1, 2], {5}, 3) == pytest.approx(0.0)


# ── Aggregate evaluator ───────────────────────────────────────────────────────

def test_evaluate_ranking_shape():
    n_users, n_items = 10, 50
    rng = np.random.default_rng(0)

    interactions = []
    for u in range(n_users):
        for i in rng.choice(n_items, size=10, replace=False):
            interactions.append({"user_idx": u, "item_idx": int(i)})
    df       = pd.DataFrame(interactions)
    train_df = df.groupby("user_idx").head(7).reset_index(drop=True)
    test_df  = df.groupby("user_idx").tail(3).reset_index(drop=True)

    def score_fn(u):
        return rng.random(n_items)

    result = evaluate_ranking(score_fn, test_df, train_df, k_values=[5, 10])
    assert list(result.index) == [5, 10]
    assert set(result.columns) == {"Precision@K", "Recall@K", "NDCG@K", "Hit@K"}
    assert (result.values >= 0).all()
    assert (result.values <= 1).all()
