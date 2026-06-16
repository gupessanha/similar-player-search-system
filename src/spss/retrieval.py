"""Recuperação de jogadores similares por aspecto × métrica, com nomes legíveis.

A classe :class:`Retriever` transforma o método validado no Notebook 3 numa
busca usável: dado um jogador-consulta (por ``UID`` ou ``Name``), devolve os
vizinhos mais próximos em cada aspecto (T/M/F/SP) sob cada métrica
(Euclidiana/Cosseno/Pearson), já com nomes — e exporta a seleção completa de
todos os jogadores para CSV.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .distances import (
    METRIC_LABELS,
    METRICS,
    prepare_metric_space,
    score_block,
    topk_from_scores,
)
from .features import FEATURE_SETS

# Aliases curtos para os aspectos expostos ao usuário.
ASPECT_ALIASES = {"technical": "T", "mental": "M", "physical": "F", "set_pieces": "SP"}
ALIAS_TO_ASPECT = {alias: name for name, alias in ASPECT_ALIASES.items()}

# Colunas de metadados lidas do CSV bruto para tornar a saída legível.
META_COLUMNS = ["UID", "Name", "Club", "Age", "Position", "Nat"]

DEFAULT_ASPECTS = ("technical", "mental", "physical", "set_pieces")

# Schema do CSV exportado por :meth:`Retriever.export_neighbors`.
NEIGHBORS_COLUMNS = [
    "query_uid", "query_name", "query_role",
    "aspect", "metric", "rank",
    "neighbor_uid", "neighbor_name", "neighbor_role",
    "score", "score_kind",
]


class Retriever:
    """Busca de jogadores similares sobre o universo z-scored de jogadores de linha."""

    def __init__(
        self,
        z_df: pd.DataFrame,
        feature_sets: dict[str, list[str]],
        meta_df: pd.DataFrame,
        aspects: tuple[str, ...] = DEFAULT_ASPECTS,
    ) -> None:
        self.feature_sets = feature_sets
        self.aspects = list(aspects)

        self.uids = z_df["UID"].to_numpy()
        self.roles = z_df["primary_role"].to_numpy()
        self.uid_to_row = {int(uid): i for i, uid in enumerate(self.uids)}

        # Uma matriz float32 por aspecto + os espaços preparados (aspecto × métrica).
        self.matrices = {
            aspect: z_df[feature_sets[aspect]].to_numpy(dtype=np.float32, copy=True)
            for aspect in self.aspects
        }
        self.spaces = {
            (aspect, metric): prepare_metric_space(self.matrices[aspect], metric)
            for aspect in self.aspects
            for metric in METRICS
        }

        # Metadados alinhados à ordem EXATA de z_df (linha i <-> jogador i).
        meta_indexed = meta_df.drop_duplicates(subset="UID").set_index("UID")
        meta_aligned = meta_indexed.reindex(self.uids)
        if meta_aligned["Name"].isna().any():
            missing = int(meta_aligned["Name"].isna().sum())
            raise ValueError(f"{missing} jogador(es) de linha sem metadados em meta_df.")
        self.meta = meta_aligned.reset_index(drop=True)
        self.meta.insert(0, "primary_role", self.roles)
        self._names = self.meta["Name"].to_numpy()

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_artifacts(
        cls,
        processed_dir: str | Path,
        merged_csv: str | Path,
        aspects: tuple[str, ...] = DEFAULT_ASPECTS,
    ) -> "Retriever":
        """Carrega o recuperador dos artefatos em ``data/processed`` + CSV bruto."""
        processed_dir = Path(processed_dir)
        z_df = pd.read_csv(processed_dir / "outfield_z.csv")
        with open(processed_dir / "feature_sets.json", encoding="utf-8") as handle:
            feature_sets = json.load(handle)
        meta_df = pd.read_csv(merged_csv, usecols=META_COLUMNS)
        return cls(z_df, feature_sets, meta_df, aspects=aspects)

    @classmethod
    def from_frames(
        cls,
        z_df: pd.DataFrame,
        feature_sets: dict[str, list[str]],
        meta_df: pd.DataFrame,
        aspects: tuple[str, ...] = DEFAULT_ASPECTS,
    ) -> "Retriever":
        """Constrói diretamente de DataFrames (útil para testes sintéticos)."""
        return cls(z_df, feature_sets, meta_df, aspects=aspects)

    # ------------------------------------------------------------------ helpers
    def _resolve_aspect(self, aspect: str) -> str:
        if aspect in self.feature_sets:
            return aspect
        if aspect in ALIAS_TO_ASPECT:
            return ALIAS_TO_ASPECT[aspect]
        raise ValueError(
            f"aspecto inválido: {aspect!r}; use {self.aspects} ou aliases {list(ALIAS_TO_ASPECT)}."
        )

    def resolve_query(self, query: int | np.integer | str) -> int:
        """Mapeia ``UID`` (int) ou ``Name`` (str) para o índice de linha no universo.

        Nomes que colidem (ex.: 34 "Paulinho") levantam ``ValueError`` com a tabela
        de candidatos e seus ``UID`` para o usuário desambiguar.
        """
        if isinstance(query, (int, np.integer)):
            if int(query) not in self.uid_to_row:
                raise KeyError(f"UID {query} não está no universo de jogadores de linha.")
            return self.uid_to_row[int(query)]

        hits = self.meta.index[self.meta["Name"] == query].to_numpy()
        if len(hits) == 0:
            raise KeyError(f"Nenhum jogador chamado {query!r}.")
        if len(hits) > 1:
            opts = self.meta.loc[hits, ["Name", "Club", "Age", "primary_role"]].copy()
            opts.insert(0, "UID", self.uids[hits])
            raise ValueError(
                f"{len(hits)} jogadores chamados {query!r}. Especifique por UID:\n"
                f"{opts.to_string(index=False)}"
            )
        return int(hits[0])

    # ------------------------------------------------------------------ retrieval
    def similar(
        self,
        query: int | np.integer | str,
        aspect: str = "technical",
        metric: str = "cosine",
        k: int = 10,
    ) -> pd.DataFrame:
        """Top-``k`` vizinhos nomeados de ``query`` em (``aspect``, ``metric``).

        O score exibido é honesto por métrica: ``similarity`` (maior = mais
        similar) para cosseno/pearson; ``distance`` (menor = mais similar, em
        distância euclidiana real) para euclidiana.
        """
        aspect = self._resolve_aspect(aspect)
        if metric not in METRICS:
            raise ValueError(f"métrica inválida: {metric!r}; use {list(METRICS)}.")

        row_idx = self.resolve_query(query)
        prepared = self.spaces[(aspect, metric)]
        query_block = self.matrices[aspect][[row_idx]]
        scores = score_block(query_block, prepared)
        neighbors = topk_from_scores(scores, np.array([row_idx]), k)[0]
        raw = scores[0, neighbors]

        if metric == "euclidean":
            score_col, score_kind = np.sqrt(raw), "distance"
        else:
            score_col, score_kind = 1.0 - raw, "similarity"

        out = self.meta.iloc[neighbors][["Name", "Club", "Age", "primary_role"]].reset_index(drop=True)
        out.insert(0, "rank", np.arange(1, len(neighbors) + 1))
        out.insert(1, "UID", self.uids[neighbors])
        out = out.rename(columns={"primary_role": "role"})
        out[score_kind] = score_col
        return out

    def grid(
        self,
        query: int | np.integer | str,
        k: int = 5,
        aspects: tuple[str, ...] | None = None,
        metrics: tuple[str, ...] = METRICS,
    ) -> pd.DataFrame:
        """DataFrame longo: uma linha por (aspecto, métrica, rank) para ``query``."""
        aspect_list = [self._resolve_aspect(a) for a in (aspects or self.aspects)]
        frames = []
        for aspect in aspect_list:
            for metric in metrics:
                block = self.similar(query, aspect=aspect, metric=metric, k=k)
                score_kind = "distance" if metric == "euclidean" else "similarity"
                frames.append(
                    block.assign(
                        aspect=ASPECT_ALIASES[aspect],
                        metric=METRIC_LABELS[metric],
                        score=block[score_kind].round(3),
                    )[["aspect", "metric", "rank", "Name", "Club", "role", "score"]]
                )
        return pd.concat(frames, ignore_index=True)

    def show_grid(
        self,
        query: int | np.integer | str,
        k: int = 5,
        aspects: tuple[str, ...] | None = None,
    ) -> pd.DataFrame:
        """Renderiza a grade aspecto × métrica (nome do vizinho por célula).

        Mostra um pivot com MultiIndex ``(aspecto, rank)`` nas linhas e as três
        métricas nas colunas. Retorna o DataFrame longo subjacente.
        """
        from IPython.display import display

        row_idx = self.resolve_query(query)
        q = self.meta.iloc[row_idx]
        print(
            f"Consulta: {q['Name']} ({q['Club']}, {q['primary_role']}, "
            f"UID {self.uids[row_idx]})"
        )
        long = self.grid(query, k=k, aspects=aspects)
        wide = long.pivot_table(
            index=["aspect", "rank"], columns="metric", values="Name", aggfunc="first"
        )
        order = [ASPECT_ALIASES[self._resolve_aspect(a)] for a in (aspects or self.aspects)]
        wide = wide.reindex(order, level="aspect")
        wide = wide[[METRIC_LABELS[m] for m in METRICS if METRIC_LABELS[m] in wide.columns]]
        display(wide)
        return long

    # ------------------------------------------------------------------ export
    def export_neighbors(
        self,
        out_path: str | Path,
        k: int = 10,
        block_size: int = 256,
        aspects: tuple[str, ...] | None = None,
        metrics: tuple[str, ...] = METRICS,
        verbose: bool = True,
    ) -> Path:
        """Exporta os top-``k`` vizinhos de TODOS os jogadores em cada aspecto×métrica.

        Escreve um CSV longo (schema :data:`NEIGHBORS_COLUMNS`), uma condição por
        vez em modo append, para limitar o pico de memória.
        """
        out_path = Path(out_path)
        aspect_list = [self._resolve_aspect(a) for a in (aspects or self.aspects)]
        n = len(self.uids)
        if k > n - 1:
            raise ValueError(f"k={k} é grande demais para um universo de {n} jogadores.")

        if out_path.exists():
            out_path.unlink()

        header_written = False
        rank_template = np.tile(np.arange(1, k + 1), n)
        query_uid_col = np.repeat(self.uids, k)
        query_name_col = np.repeat(self._names, k)
        query_role_col = np.repeat(self.roles, k)

        for aspect in aspect_list:
            matrix = self.matrices[aspect]
            aspect_alias = ASPECT_ALIASES[aspect]
            for metric in metrics:
                prepared = self.spaces[(aspect, metric)]
                score_kind = "distance" if metric == "euclidean" else "similarity"

                neighbors_all = np.empty((n, k), dtype=np.int64)
                scores_all = np.empty((n, k), dtype=np.float32)
                for start in range(0, n, block_size):
                    block_idx = np.arange(start, min(start + block_size, n))
                    block_scores = score_block(matrix[block_idx], prepared)
                    nb = topk_from_scores(block_scores, block_idx, k)
                    neighbors_all[block_idx] = nb
                    gathered = np.take_along_axis(block_scores, nb, axis=1)
                    scores_all[block_idx] = (
                        np.sqrt(gathered) if metric == "euclidean" else 1.0 - gathered
                    )

                neigh_flat = neighbors_all.reshape(-1)
                chunk = pd.DataFrame({
                    "query_uid": query_uid_col,
                    "query_name": query_name_col,
                    "query_role": query_role_col,
                    "aspect": aspect_alias,
                    "metric": METRIC_LABELS[metric],
                    "rank": rank_template,
                    "neighbor_uid": self.uids[neigh_flat],
                    "neighbor_name": self._names[neigh_flat],
                    "neighbor_role": self.roles[neigh_flat],
                    "score": scores_all.reshape(-1),
                    "score_kind": score_kind,
                })[NEIGHBORS_COLUMNS]
                chunk.to_csv(out_path, mode="a", header=not header_written, index=False)
                header_written = True
                if verbose:
                    print(f"  {aspect_alias} | {METRIC_LABELS[metric]}: {len(chunk):,} linhas")

        if verbose:
            print(f"Export concluído: {out_path}")
        return out_path
