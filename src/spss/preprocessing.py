"""Carga e higiene do dataset bruto FM2023 (reproduz o Notebook 1).

Remove colunas residuais, deduplica por ``UID`` (chave única — ``Name`` colide)
e separa goleiros de jogadores de linha pelo ``primary_role`` derivado.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .roles import primary_role

# Colunas removidas na higiene: índice residual do CSV e coluna constante.
DROP_COLUMNS = ["Unnamed: 0", "Rec"]


def load_raw(path: str | Path) -> pd.DataFrame:
    """Lê o CSV bruto ``merged_players (1).csv`` (91.672 × 88)."""
    return pd.read_csv(path)


def clean(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Remove colunas residuais, deduplica por ``UID`` e adiciona ``primary_role``.

    As duplicatas (~4,9%) são cópias exatas de um merge, diferindo apenas no
    índice; ``drop_duplicates(subset="UID")`` mantém a primeira ocorrência.
    """
    df = (
        df_raw
        .drop(columns=[c for c in DROP_COLUMNS if c in df_raw.columns])
        .drop_duplicates(subset="UID")
        .reset_index(drop=True)
    )
    df["primary_role"] = df["Position"].map(primary_role)
    return df


def split_outfield_gk(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa jogadores de linha (``primary_role != "GK"``) de goleiros.

    Returns
    -------
    (df_outfield, df_gk)
    """
    df_gk = df[df["primary_role"] == "GK"].reset_index(drop=True)
    df_outfield = df[df["primary_role"] != "GK"].reset_index(drop=True)
    return df_outfield, df_gk
