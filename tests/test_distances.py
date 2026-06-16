import numpy as np

from spss.distances import prepare_metric_space, score_block, topk_from_scores


def test_euclidean_self_zero_and_symmetric():
    X = np.array([[1.0, 2, 3], [4, 5, 6], [1, 2, 3.001]], dtype=np.float32)
    prep = prepare_metric_space(X, "euclidean")
    scores = score_block(X, prep)
    assert np.allclose(np.diag(scores), 0.0, atol=1e-3)
    assert np.allclose(scores, scores.T, atol=1e-2)


def test_cosine_collinear_zero_orthogonal_one():
    X = np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    prep = prepare_metric_space(X, "cosine")
    scores = score_block(X, prep)
    assert scores[0, 1] < 1e-5     # colineares -> distância cosseno ~0
    assert scores[0, 2] > 0.9      # ortogonais -> ~1


def test_topk_excludes_self_and_sorted():
    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    prep = prepare_metric_space(X, "euclidean")
    scores = score_block(X, prep)
    nb = topk_from_scores(scores, np.array([0, 1, 2, 3]), top_k=2)

    assert 0 not in nb[0].tolist()   # exclui a própria consulta
    assert nb[0][0] == 1             # vizinho mais próximo de 0 é 1
    gathered = np.take_along_axis(scores, nb, axis=1)
    assert np.all(np.diff(gathered, axis=1) >= 0)  # ordenado do mais perto ao mais longe
