"""Conjuntos de features e normalização z-score por posição.

Reproduz o Notebook 2: define os quatro conjuntos disjuntos T/M/F/SP e
padroniza cada atributo para média 0 / desvio 1 *dentro* de cada
``primary_role`` (z-score por posição, ``ddof=0``).
"""

from __future__ import annotations

import pandas as pd

# Quatro conjuntos disjuntos de atributos de jogador de linha (36 no total).
FEATURE_SETS: dict[str, list[str]] = {
    "technical": [
        "Cro", "Dri", "Fin", "Fir", "Hea", "Lon", "Mar", "Pas", "Tck", "Tec",
    ],
    "mental": [
        "Agg", "Ant", "Bra", "Cmp", "Cnt", "Dec", "Det", "Fla",
        "Ldr", "OtB", "Pos", "Tea", "Vis", "Wor",
    ],
    "physical": [
        "Acc", "Agi", "Bal", "Jum", "Nat.1", "Pac", "Sta", "Str",
    ],
    "set_pieces": [
        "Cor", "Fre", "Pen", "L Th",
    ],
}

# Ordem canônica dos 36 atributos (T + M + F + SP), igual à de ``outfield_z.csv``.
ALL_FEATURES: list[str] = [col for cols in FEATURE_SETS.values() for col in cols]

# Atributos exclusivos de goleiro — ficam fora dos conjuntos de linha.
GK_ONLY = {"Aer", "Cmd", "Com", "Ecc", "Han", "Kic", "1v1", "Ref", "TRO", "Pun", "Thr"}

assert len(ALL_FEATURES) == 36, "O experimento espera 36 atributos no total."
assert len(set(ALL_FEATURES)) == len(ALL_FEATURES), "Há atributos repetidos entre conjuntos."


def zscore_by_role(
    df_outfield: pd.DataFrame,
    features: list[str] = ALL_FEATURES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Padroniza cada atributo para média 0 / desvio 1 dentro de cada role.

    Replica o Notebook 2 (desvio populacional, ``ddof=0``).

    Parameters
    ----------
    df_outfield:
        DataFrame de jogadores de linha contendo ``UID``, ``primary_role`` e os
        atributos em ``features`` na escala 1–20.
    features:
        Lista de atributos a padronizar (default: os 36 de :data:`ALL_FEATURES`).

    Returns
    -------
    (z, stats_long):
        ``z`` — DataFrame ``[UID, primary_role, *features]`` com os valores
        padronizados. ``stats_long`` — DataFrame ``[role, attribute, mu, sigma]``
        em formato long (uma linha por par role×atributo).

    Raises
    ------
    ValueError
        Se algum atributo tiver variância zero dentro de algum role (divisão por
        zero). Não ocorre nos dados reais, mas falha de forma explícita.
    """
    grouped = df_outfield.groupby("primary_role")[features]
    mu_by_role = grouped.mean()
    sigma_by_role = grouped.std(ddof=0)

    zero_var = sigma_by_role.eq(0)
    if zero_var.to_numpy().any():
        bad = [
            (role, attr)
            for role in zero_var.index
            for attr in features
            if zero_var.loc[role, attr]
        ]
        raise ValueError(f"Atributo(s) com variância zero dentro de um role: {bad}")

    mu_per_player = grouped.transform("mean")
    sigma_per_player = grouped.transform(lambda s: s.std(ddof=0))
    z_values = (df_outfield[features] - mu_per_player) / sigma_per_player

    z = z_values.copy()
    z.insert(0, "primary_role", df_outfield["primary_role"].to_numpy())
    z.insert(0, "UID", df_outfield["UID"].to_numpy())

    stats_long = pd.DataFrame(
        [
            {
                "role": role,
                "attribute": attr,
                "mu": mu_by_role.loc[role, attr],
                "sigma": sigma_by_role.loc[role, attr],
            }
            for role in mu_by_role.index
            for attr in features
        ]
    )
    return z, stats_long
