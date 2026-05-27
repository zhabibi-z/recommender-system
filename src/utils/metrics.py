"""Evaluation metrics for recommender systems: RMSE, MAE, Precision@K, Recall@K, NDCG@K, Hit@K."""
from typing import Callable

import numpy as np
import pandas as pd


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if k == 0:
        return 0.0
    return sum(1 for item in recommended[:k] if item in relevant) / k


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for item in recommended[:k] if item in relevant) / len(relevant)


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Normalised Discounted Cumulative Gain at K."""
    dcg  = sum(1.0 / np.log2(r + 2) for r, item in enumerate(recommended[:k]) if item in relevant)
    idcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    return float(any(item in relevant for item in recommended[:k]))


def evaluate_ranking(
    score_fn:  Callable[[int], np.ndarray],
    test_df:   pd.DataFrame,
    train_df:  pd.DataFrame,
    k_values:  list[int] = [5, 10, 20],
    n_sample:  int       = 500,
    seed:      int       = 42,
) -> pd.DataFrame:
    """
    Evaluate a ranking model over a random sample of test users.

    For each user: obtain scores from `score_fn`, mask train-seen items,
    rank all items, then compute Precision@K, Recall@K, NDCG@K, Hit@K.

    Returns a DataFrame indexed by k.
    """
    test_pos  = test_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    train_pos = train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()

    users = list(test_pos.keys())
    if len(users) > n_sample:
        rng   = np.random.default_rng(seed)
        users = rng.choice(users, n_sample, replace=False).tolist()

    accum: dict[int, dict[str, list]] = {
        k: {"precision": [], "recall": [], "ndcg": [], "hit": []}
        for k in k_values
    }

    for u in users:
        relevant = test_pos.get(u, set())
        if not relevant:
            continue

        scores = score_fn(int(u)).copy()
        for i in train_pos.get(u, set()):
            if i < len(scores):
                scores[i] = -1e9

        ranked = np.argsort(scores)[::-1].tolist()

        for k in k_values:
            accum[k]["precision"].append(precision_at_k(ranked, relevant, k))
            accum[k]["recall"].append(recall_at_k(ranked, relevant, k))
            accum[k]["ndcg"].append(ndcg_at_k(ranked, relevant, k))
            accum[k]["hit"].append(hit_rate_at_k(ranked, relevant, k))

    rows = [
        {
            "k":           k,
            "Precision@K": float(np.mean(accum[k]["precision"])),
            "Recall@K":    float(np.mean(accum[k]["recall"])),
            "NDCG@K":      float(np.mean(accum[k]["ndcg"])),
            "Hit@K":       float(np.mean(accum[k]["hit"])),
        }
        for k in k_values
    ]
    return pd.DataFrame(rows).set_index("k").round(4)
