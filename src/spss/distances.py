"""Primitivas de distância para recuperação de similaridade (fonte canônica).

Extraídas do Notebook 3. ``score_block`` devolve *distâncias transformadas*
(menor = mais perto):

- ``euclidean`` → distância ao quadrado (``>= 0``);
- ``cosine`` / ``pearson`` → ``1 - similaridade`` (similaridade recortada em
  ``[-1, 1]``).

As distâncias são calculadas em blocos para evitar materializar a matriz
``N × N`` completa.
"""

from __future__ import annotations

import numpy as np

METRICS = ("euclidean", "cosine", "pearson")
METRIC_LABELS = {
    "euclidean": "Euclidiana",
    "cosine": "Cosseno",
    "pearson": "Pearson",
}


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Normaliza cada linha para norma L2 unitária (linhas nulas ficam nulas)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


def prepare_metric_space(matrix: np.ndarray, metric: str) -> dict[str, np.ndarray | str]:
    """Pré-computa a representação do alvo para uma métrica.

    Para ``euclidean`` guarda também as normas ao quadrado das linhas; para
    ``cosine``/``pearson`` guarda as linhas já normalizadas (e centradas, no caso
    de Pearson).
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    if metric == "euclidean":
        return {
            "metric": metric,
            "X": matrix,
            "x_sq": np.einsum("ij,ij->i", matrix, matrix),
        }
    if metric == "cosine":
        return {
            "metric": metric,
            "X": normalize_rows(matrix),
        }
    if metric == "pearson":
        centered = matrix - matrix.mean(axis=1, keepdims=True)
        return {
            "metric": metric,
            "X": normalize_rows(centered),
        }
    raise ValueError(f"Métrica desconhecida: {metric}")


def score_block(query_block: np.ndarray, prepared: dict[str, np.ndarray | str]) -> np.ndarray:
    """Distâncias transformadas de um bloco de consultas contra todo o alvo.

    Retorna matriz ``(n_queries, n_target)`` onde menor = mais similar.
    """
    metric = prepared["metric"]
    target = prepared["X"]

    if metric == "euclidean":
        query_block = np.asarray(query_block, dtype=np.float32)
        q_sq = np.einsum("ij,ij->i", query_block, query_block)[:, None]
        scores = q_sq + prepared["x_sq"][None, :] - 2.0 * query_block @ target.T
        return np.maximum(scores, 0.0)

    if metric == "cosine":
        query_ready = normalize_rows(np.asarray(query_block, dtype=np.float32))
    else:
        centered = np.asarray(query_block, dtype=np.float32)
        centered = centered - centered.mean(axis=1, keepdims=True)
        query_ready = normalize_rows(centered)

    similarity = np.clip(query_ready @ target.T, -1.0, 1.0)
    return 1.0 - similarity


def topk_from_scores(scores: np.ndarray, query_rows: np.ndarray, top_k: int) -> np.ndarray:
    """Top-``k`` vizinhos (índices) por linha, excluindo a própria consulta.

    ``query_rows[i]`` é o índice da consulta ``i`` na matriz-alvo; essa coluna é
    marcada como infinita para nunca aparecer entre os vizinhos. Os vizinhos
    saem ordenados do mais próximo ao mais distante.
    """
    working = scores.copy()
    working[np.arange(len(query_rows)), query_rows] = np.inf
    partition = np.argpartition(working, kth=top_k - 1, axis=1)[:, :top_k]
    partition_scores = np.take_along_axis(working, partition, axis=1)
    order = np.argsort(partition_scores, axis=1)
    return np.take_along_axis(partition, order, axis=1)
