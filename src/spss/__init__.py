"""SPSS — Similar Player Search System.

Pipeline de pré-processamento reproduzível (notebooks 1–2) e recuperação de
jogadores similares por aspecto × métrica (notebook 3).
"""

from __future__ import annotations

from .distances import (
    METRIC_LABELS,
    METRICS,
    normalize_rows,
    prepare_metric_space,
    score_block,
    topk_from_scores,
)
from .features import ALL_FEATURES, FEATURE_SETS, zscore_by_role
from .preprocessing import clean, load_raw, split_outfield_gk
from .retrieval import Retriever
from .roles import ROLE_MAP, primary_role

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # roles / preprocessing
    "ROLE_MAP",
    "primary_role",
    "load_raw",
    "clean",
    "split_outfield_gk",
    # features
    "FEATURE_SETS",
    "ALL_FEATURES",
    "zscore_by_role",
    # distances
    "METRICS",
    "METRIC_LABELS",
    "normalize_rows",
    "prepare_metric_space",
    "score_block",
    "topk_from_scores",
    # retrieval
    "Retriever",
]
