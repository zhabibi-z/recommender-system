"""Bayesian Personalised Ranking (BPR) matrix factorisation model.

Reference: Rendle et al. "BPR: Bayesian Personalized Ranking from Implicit
Feedback." UAI 2009.  https://arxiv.org/abs/1205.2618
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

log = logging.getLogger(__name__)


class BPRModel(nn.Module):
    """Matrix factorisation model optimised with pairwise BPR loss."""

    def __init__(self, n_users: int, n_items: int, k: int = 64) -> None:
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.k = k
        self.user_emb = nn.Embedding(n_users, k)
        self.item_emb = nn.Embedding(n_items, k)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(
        self,
        user: torch.Tensor,
        pos_item: torch.Tensor,
        neg_item: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        u = self.user_emb(user)
        i = self.item_emb(pos_item)
        j = self.item_emb(neg_item)
        return (u * i).sum(dim=1), (u * j).sum(dim=1)

    def score_all_items(self, user_idx: int) -> np.ndarray:
        """Rank scores for all items for one user (higher = more recommended)."""
        with torch.no_grad():
            u = self.user_emb.weight[user_idx]          # (k,)
            scores = self.item_emb.weight @ u            # (n_items,)
        return scores.cpu().numpy()


def _sample_negative(
    positives: set[int], n_items: int, rng: np.random.Generator
) -> int:
    while True:
        j = int(rng.integers(0, n_items))
        if j not in positives:
            return j


def train_bpr(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    n_users: int,
    n_items: int,
    k: int = 64,
    n_negatives: int = 4,
    lr: float = 0.01,
    reg: float = 0.001,
    n_epochs: int = 40,
    patience: int = 5,
    batch_size: int = 2048,
) -> BPRModel:
    """Train a BPR model; returns the best checkpoint by validation BPR loss."""
    user_col = "user_idx" if "user_idx" in train_df.columns else train_df.columns[0]
    item_col = "item_idx" if "item_idx" in train_df.columns else train_df.columns[1]

    # Build per-user positive set for negative sampling
    user_positives: dict[int, set[int]] = {}
    for u, i in zip(train_df[user_col].values, train_df[item_col].values):
        user_positives.setdefault(int(u), set()).add(int(i))

    users_arr = train_df[user_col].values.astype(np.int64)
    items_arr = train_df[item_col].values.astype(np.int64)
    rng = np.random.default_rng(42)

    model = BPRModel(n_users, n_items, k)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    no_improve = 0
    best_state: dict = {}

    for epoch in range(n_epochs):
        model.train()
        perm = rng.permutation(len(users_arr))
        epoch_loss, n_batches = 0.0, 0

        for start in range(0, len(perm), batch_size):
            idx = perm[start : start + batch_size]
            u_np = users_arr[idx]
            i_np = items_arr[idx]
            j_np = np.array([
                _sample_negative(user_positives.get(int(u), set()), n_items, rng)
                for u in u_np
            ])

            u_t = torch.from_numpy(u_np)
            i_t = torch.from_numpy(i_np)
            j_t = torch.from_numpy(j_np)

            pos_s, neg_s = model(u_t, i_t, j_t)
            bpr_loss = -torch.log(torch.sigmoid(pos_s - neg_s) + 1e-8).mean()
            l2 = reg * (
                model.user_emb(u_t).norm(2).pow(2)
                + model.item_emb(i_t).norm(2).pow(2)
                + model.item_emb(j_t).norm(2).pow(2)
            )
            loss = bpr_loss + l2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        # Validation BPR loss on a sample
        model.eval()
        with torch.no_grad():
            n_val = min(2000, len(val_df))
            val_u = torch.from_numpy(val_df[user_col].values[:n_val].astype(np.int64))
            val_i = torch.from_numpy(val_df[item_col].values[:n_val].astype(np.int64))
            val_j = torch.from_numpy(np.array([
                _sample_negative(user_positives.get(int(u), set()), n_items, rng)
                for u in val_u.numpy()
            ]))
            vps, vns = model(val_u, val_i, val_j)
            val_loss = -torch.log(torch.sigmoid(vps - vns) + 1e-8).mean().item()

        log.info(
            "Epoch %2d/%d  train=%.4f  val=%.4f",
            epoch + 1, n_epochs, epoch_loss / max(n_batches, 1), val_loss,
        )

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            no_improve = 0
            best_state = {k_: v.clone() for k_, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info("Early stopping at epoch %d (patience=%d)", epoch + 1, patience)
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model


def save(model: BPRModel, path: Path, metadata: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_users":    model.n_users,
            "n_items":    model.n_items,
            "k":          model.k,
            "metadata":   metadata or {},
        },
        path,
    )
    log.info("Model saved → %s", path)


def load(path: Path) -> tuple[BPRModel, dict]:
    """Return (model, ckpt_dict) so callers can inspect metadata."""
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
    model = BPRModel(ckpt["n_users"], ckpt["n_items"], ckpt["k"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
