import numpy as np
import pandas as pd
import pytest

from spss.features import ALL_FEATURES, FEATURE_SETS, zscore_by_role


def _synthetic_outfield(n=200, roles=("AM", "DEF"), seed=0):
    rng = np.random.default_rng(seed)
    role_col = np.array([roles[i % len(roles)] for i in range(n)])
    data = {"UID": np.arange(n), "primary_role": role_col}
    for feat in ALL_FEATURES:
        data[feat] = rng.integers(1, 21, size=n).astype(float)
    return pd.DataFrame(data)


def test_feature_set_counts_and_disjoint():
    assert len(FEATURE_SETS["technical"]) == 10
    assert len(FEATURE_SETS["mental"]) == 14
    assert len(FEATURE_SETS["physical"]) == 8
    assert len(FEATURE_SETS["set_pieces"]) == 4
    assert len(ALL_FEATURES) == 36
    assert len(set(ALL_FEATURES)) == 36  # disjuntos, sem repetição


def test_zscore_by_role_mean0_std1_within_role():
    df = _synthetic_outfield()
    z, stats = zscore_by_role(df, ALL_FEATURES)

    for role in df["primary_role"].unique():
        sub = z[z["primary_role"] == role][ALL_FEATURES]
        assert np.allclose(sub.mean().to_numpy(), 0.0, atol=1e-6)
        assert np.allclose(sub.std(ddof=0).to_numpy(), 1.0, atol=1e-6)

    assert list(stats.columns) == ["role", "attribute", "mu", "sigma"]
    assert len(stats) == df["primary_role"].nunique() * 36
    # ordem dos atributos preservada dentro de cada role
    first_role = stats["role"].iloc[0]
    assert stats[stats["role"] == first_role]["attribute"].tolist() == ALL_FEATURES


def test_zscore_zero_variance_raises():
    df = _synthetic_outfield(n=50, roles=("AM",))
    df["Cro"] = 1.0  # variância zero dentro do único role
    with pytest.raises(ValueError):
        zscore_by_role(df, ALL_FEATURES)
