import numpy as np

from marketvantage.retrieval import expand_queries, mmr_select


def test_expand_queries_basic():
    qs = expand_queries("ai lip sync", n=4)
    assert len(qs) == 4
    assert qs[0] == "ai lip sync"
    # Ensure no duplicates
    assert len(set(qs)) == len(qs)


def test_mmr_select_diversity():
    # Create 6 doc vectors in 3 clusters; query is close to cluster A
    rng = np.random.default_rng(0)
    cluster_a = np.tile(np.array([1.0, 0.0, 0.0]), (2, 1)) + 0.01 * rng.standard_normal((2, 3))
    cluster_b = np.tile(np.array([0.0, 1.0, 0.0]), (2, 1)) + 0.01 * rng.standard_normal((2, 3))
    cluster_c = np.tile(np.array([0.0, 0.0, 1.0]), (2, 1)) + 0.01 * rng.standard_normal((2, 3))
    docs = np.vstack([cluster_a, cluster_b, cluster_c])
    # Normalize
    docs = docs / np.linalg.norm(docs, axis=1, keepdims=True)

    query_vec = np.array([1.0, 0.1, 0.0])
    query_vec = query_vec / np.linalg.norm(query_vec)

    cand_ids = list(range(docs.shape[0]))
    selected = mmr_select(query_vec, docs, cand_ids, top_k=3, lambda_=0.7)

    # Should start with a doc from cluster A, and include others for diversity
    assert len(selected) == 3
    assert selected[0] in [0, 1]
    # Ensure distinct ids and within range
    assert len(set(selected)) == 3
    assert all(0 <= i < docs.shape[0] for i in selected)


