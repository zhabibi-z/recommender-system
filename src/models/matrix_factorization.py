"""
Biased Matrix Factorisation — rating-prediction baseline.

Decomposes the user-item rating matrix as:

    P(u, i) = μ + b_u + b_i + A[u] · F[i]

where μ is the global mean, b_u and b_i are user/item biases, and A, F are
latent factor matrices. Parameters are learned by minimising masked MSE with
L2 regularisation. Optional per-user mean-centring reduces the impact of
rating skew before training.

For ranking-based recommendations, prefer src/models/bpr.py.
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.config import cfg

log = logging.getLogger(__name__)


@dataclass
class MFConfig:
    k:          int   = 32
    lr:         float = 0.01
    momentum:   float = 0.9
    n_steps:    int   = 500
    lambda_reg: float = 0.001
    patience:   int   = 20
    normalize:  bool  = True   # subtract per-user mean before training


class MFModel:
    """Learned MF weights with an optional per-user mean offset."""

    def __init__(
        self,
        A:          torch.Tensor,
        F:          torch.Tensor,
        b_u:        torch.Tensor,
        b_i:        torch.Tensor,
        mu:         float,
        user_means: Optional[np.ndarray] = None,
    ) -> None:
        self.A, self.F, self.b_u, self.b_i = A, F, b_u, b_i
        self.mu         = mu
        self.user_means = user_means

    def predict_matrix(self) -> np.ndarray:
        P = (self.mu + self.b_u + self.b_i + self.A @ self.F).detach().numpy()
        if self.user_means is not None:
            P = P + self.user_means[:, None]
        return np.clip(P, 1.0, 5.0)


def train_mf(
    S_train: np.ndarray,
    R_train: np.ndarray,
    S_val:   np.ndarray,
    R_val:   np.ndarray,
    mfcfg:   MFConfig = MFConfig(),
) -> MFModel:
    """
    Train biased MF with optional per-user mean-centring.

    S_train, S_val: rating matrices (users × items), 0 where unrated.
    R_train, R_val: binary mask matrices (1 where rated).
    """
    n_users, n_items = S_train.shape

    user_means = None
    S_tr = S_train.copy()
    S_v  = S_val.copy()

    if mfcfg.normalize:
        row_sums   = (S_tr * R_train).sum(axis=1)
        row_counts = R_train.sum(axis=1).clip(min=1)
        user_means = row_sums / row_counts
        S_tr = S_tr - (user_means[:, None] * R_train)
        S_v  = S_v  - (user_means[:, None] * R_val)

    S_tr_t = torch.tensor(S_tr,    dtype=torch.float32)
    R_tr_t = torch.tensor(R_train, dtype=torch.float32)
    S_v_t  = torch.tensor(S_v,     dtype=torch.float32)
    R_v_t  = torch.tensor(R_val,   dtype=torch.float32)

    N_tr = R_tr_t.sum().item()
    N_v  = R_v_t.sum().item()
    mu   = float((S_tr_t * R_tr_t).sum() / N_tr)

    torch.manual_seed(cfg.data.random_seed)
    A   = torch.randn(n_users, mfcfg.k, requires_grad=True)
    F   = torch.randn(mfcfg.k, n_items, requires_grad=True)
    b_u = torch.zeros(n_users, 1,       requires_grad=True)
    b_i = torch.zeros(1,       n_items, requires_grad=True)

    v_A, v_F, v_bu, v_bi = (torch.zeros_like(t) for t in (A, F, b_u, b_i))

    best_val, best_state, patience_ctr = float("inf"), None, 0

    for step in range(1, mfcfg.n_steps + 1):
        P    = mu + b_u + b_i + A @ F
        loss = torch.sum(R_tr_t * (S_tr_t - P) ** 2) / N_tr
        reg  = mfcfg.lambda_reg * (
            torch.sum(A**2) + torch.sum(F**2) +
            torch.sum(b_u**2) + torch.sum(b_i**2)
        )
        (loss + reg).backward()

        with torch.no_grad():
            for param, vel in zip([A, F, b_u, b_i], [v_A, v_F, v_bu, v_bi]):
                vel.mul_(mfcfg.momentum).add_(param.grad, alpha=mfcfg.lr)
                param.sub_(vel)
                param.grad.zero_()

            val_loss = (
                torch.sum(R_v_t * (S_v_t - (mu + b_u + b_i + A @ F)) ** 2) / N_v
            ).item()

        if val_loss < best_val:
            best_val     = val_loss
            best_state   = {
                "A": A.detach().clone(), "F": F.detach().clone(),
                "b_u": b_u.detach().clone(), "b_i": b_i.detach().clone(),
            }
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= mfcfg.patience:
                log.info("Early stopping at step %d (val RMSE %.4f)", step, best_val**0.5)
                break

        if step % 100 == 0:
            log.info("Step %d/%d | val RMSE %.4f", step, mfcfg.n_steps, best_val**0.5)

    log.info("MF training done — val RMSE: %.4f", best_val**0.5)
    return MFModel(
        A=best_state["A"], F=best_state["F"],
        b_u=best_state["b_u"], b_i=best_state["b_i"],
        mu=mu, user_means=user_means,
    )


def save(model: MFModel, path: Path, meta: dict) -> None:
    torch.save(
        {"A": model.A, "F": model.F, "b_u": model.b_u, "b_i": model.b_i,
         "mu": model.mu, "user_means": model.user_means, **meta},
        path,
    )
    log.info("Saved MF model → %s", path)


def load(path: Path) -> tuple[MFModel, dict]:
    ckpt = torch.load(path, weights_only=False, map_location="cpu")
    return MFModel(
        A=ckpt["A"], F=ckpt["F"], b_u=ckpt["b_u"], b_i=ckpt["b_i"],
        mu=ckpt["mu"], user_means=ckpt.get("user_means"),
    ), ckpt
