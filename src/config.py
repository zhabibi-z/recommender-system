"""Centralised configuration — import the singleton `cfg` anywhere in the codebase."""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class DataConfig:
    raw_dir:          Path  = field(default_factory=lambda: ROOT / "data" / "raw")
    processed_dir:    Path  = field(default_factory=lambda: ROOT / "data" / "processed")
    min_user_ratings: int   = 5      # k-core threshold per user
    min_item_ratings: int   = 10     # k-core threshold per item
    sample_size:      int   = 500_000
    val_frac:         float = 0.10   # per-user temporal holdout fractions
    test_frac:        float = 0.15
    random_seed:      int   = 42


@dataclass
class ModelConfig:
    k:           int   = 64      # latent factor dimension
    lr:          float = 0.005
    n_epochs:    int   = 30
    batch_size:  int   = 4096
    lambda_reg:  float = 0.001   # L2 regularisation weight
    n_negatives: int   = 4       # negative samples per positive (BPR)
    patience:    int   = 5       # early-stopping patience in epochs


@dataclass
class EmbeddingConfig:
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 64


@dataclass
class Config:
    data:       DataConfig      = field(default_factory=DataConfig)
    model:      ModelConfig     = field(default_factory=ModelConfig)
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    models_dir: Path            = field(default_factory=lambda: ROOT / "models")

    def __post_init__(self) -> None:
        for d in (self.data.raw_dir, self.data.processed_dir, self.models_dir):
            d.mkdir(parents=True, exist_ok=True)


cfg = Config()
