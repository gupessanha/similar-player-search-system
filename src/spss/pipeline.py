"""Pipeline de pré-processamento reproduzível: CSV bruto -> ``data/processed``.

Reproduz, ponta a ponta e fora dos notebooks 1 e 2, a geração dos cinco
artefatos consumidos pela modelagem. Uso:

    uv run python -m spss.pipeline
"""

from __future__ import annotations

import json
from pathlib import Path

from .features import ALL_FEATURES, FEATURE_SETS, zscore_by_role
from .preprocessing import clean, load_raw, split_outfield_gk


def project_root() -> Path:
    """Raiz do repositório (dois níveis acima de ``src/spss``)."""
    return Path(__file__).resolve().parents[2]


def main(
    raw_csv: str | Path | None = None,
    out_dir: str | Path | None = None,
    verbose: bool = True,
) -> dict[str, int]:
    """Regenera os cinco artefatos em ``data/processed`` a partir do CSV bruto.

    Returns
    -------
    dict
        Contagens ``{"outfield", "gk", "stats"}`` para checagem (esperado:
        78.283 / 8.880 / 216).
    """
    root = project_root()
    raw_csv = Path(raw_csv) if raw_csv else root / "data" / "merged_players (1).csv"
    out_dir = Path(out_dir) if out_dir else root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = clean(load_raw(raw_csv))
    df_outfield, df_gk = split_outfield_gk(df)
    z, stats = zscore_by_role(df_outfield, ALL_FEATURES)

    z.to_csv(out_dir / "outfield_z.csv", index=False)
    df_outfield[["UID", "primary_role", *ALL_FEATURES]].to_csv(
        out_dir / "outfield_raw.csv", index=False
    )
    df_gk[["UID", "primary_role", *ALL_FEATURES]].to_csv(
        out_dir / "gk_raw.csv", index=False
    )
    with open(out_dir / "feature_sets.json", "w", encoding="utf-8") as handle:
        json.dump(FEATURE_SETS, handle, indent=2)
    stats.to_csv(out_dir / "zscore_stats.csv", index=False)

    counts = {"outfield": len(df_outfield), "gk": len(df_gk), "stats": len(stats)}
    if verbose:
        print(
            f"Artefatos gravados em {out_dir}\n"
            f"  outfield: {counts['outfield']:,} | gk: {counts['gk']:,} | "
            f"stats: {counts['stats']} linhas"
        )
    return counts


if __name__ == "__main__":
    main()
