"""Mapeamento de ``Position`` (FM2023) para o ``primary_role`` canônico.

Os papéis canônicos são seis posições de linha (AM, DEF, DM, FW, MID, WB) mais
GK para goleiros. A lógica é extraída verbatim do Notebook 1 (EDA).
"""

from __future__ import annotations

import re

ROLE_MAP = {
    "GK": "GK",
    "D": "DEF",
    "WB": "WB",
    "DM": "DM",
    "M": "MID",
    "AM": "AM",
    "ST": "FW",
}


def primary_role(pos_str: str) -> str:
    """Deriva o papel tático primário a partir da string de posição do FM.

    Pega a primeira posição listada, extrai o token base (antes de ``/``, espaço
    ou ``(``) e mapeia via :data:`ROLE_MAP`. Tokens fora do mapa retornam
    ``"OTHER"``.

    Exemplos
    --------
    >>> primary_role("AM (R), ST (C)")
    'AM'
    >>> primary_role("D (C)")
    'DEF'
    >>> primary_role("GK")
    'GK'
    """
    first_token = pos_str.split(",")[0].strip()
    base = re.split(r"[/ (]", first_token)[0]
    return ROLE_MAP.get(base, "OTHER")
