import numpy as np
import pandas as pd
import pytest

from spss.features import ALL_FEATURES, FEATURE_SETS
from spss.retrieval import NEIGHBORS_COLUMNS, Retriever


def _universe(n=8, seed=1):
    rng = np.random.default_rng(seed)
    uids = np.arange(1000, 1000 + n)
    z = {"UID": uids, "primary_role": np.array(["AM"] * n)}
    for feat in ALL_FEATURES:
        z[feat] = rng.normal(size=n).astype(float)
    z_df = pd.DataFrame(z)
    meta = pd.DataFrame({
        "UID": uids,
        "Name": [f"P{i}" for i in range(n)],
        "Club": ["C"] * n,
        "Age": [20] * n,
        "Position": ["AM (C)"] * n,
        "Nat": ["BRA"] * n,
    })
    return z_df, meta


def test_similar_returns_k_excludes_self():
    z_df, meta = _universe()
    ret = Retriever.from_frames(z_df, FEATURE_SETS, meta)
    out = ret.similar(1000, aspect="T", metric="cosine", k=3)
    assert len(out) == 3
    assert 1000 not in out["UID"].tolist()
    # cosseno -> coluna 'similarity', maior = mais perto (rank 1 primeiro)
    assert out["similarity"].is_monotonic_decreasing


def test_similar_euclidean_distance_ascending():
    z_df, meta = _universe()
    ret = Retriever.from_frames(z_df, FEATURE_SETS, meta)
    out = ret.similar(1000, aspect="F", metric="euclidean", k=3)
    assert "distance" in out.columns
    assert out["distance"].is_monotonic_increasing


def test_aspect_alias_and_full_name_equivalent():
    z_df, meta = _universe()
    ret = Retriever.from_frames(z_df, FEATURE_SETS, meta)
    by_alias = ret.similar(1000, aspect="M", metric="pearson", k=3)
    by_name = ret.similar(1000, aspect="mental", metric="pearson", k=3)
    assert by_alias["UID"].tolist() == by_name["UID"].tolist()


def test_name_collision_raises_with_candidates():
    z_df, meta = _universe()
    meta.loc[1, "Name"] = "P0"  # colide com o índice 0
    ret = Retriever.from_frames(z_df, FEATURE_SETS, meta)
    with pytest.raises(ValueError):
        ret.resolve_query("P0")


def test_unknown_uid_raises():
    z_df, meta = _universe()
    ret = Retriever.from_frames(z_df, FEATURE_SETS, meta)
    with pytest.raises(KeyError):
        ret.resolve_query(999999)


def test_export_neighbors_schema_and_rowcount(tmp_path):
    n = 8
    z_df, meta = _universe(n=n)
    ret = Retriever.from_frames(z_df, FEATURE_SETS, meta)
    out = tmp_path / "neighbors.csv"
    ret.export_neighbors(out, k=3, block_size=4, verbose=False)

    df = pd.read_csv(out)
    assert list(df.columns) == NEIGHBORS_COLUMNS
    assert len(df) == n * 4 * 3 * 3  # jogadores × aspectos × métricas × k
    assert set(df["aspect"]) == {"T", "M", "F", "SP"}
    assert set(df["score_kind"]) == {"distance", "similarity"}
    # nenhum jogador é vizinho de si mesmo
    assert (df["query_uid"] != df["neighbor_uid"]).all()
