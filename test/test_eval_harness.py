import math


def recall_at_k(ground_truth: set[int], ranked: list[int], k: int) -> float:
    if not ground_truth:
        return 0.0
    topk = set(ranked[:k])
    return len(ground_truth & topk) / float(len(ground_truth))


def dcg_at_k(rels: list[int], k: int) -> float:
    s = 0.0
    for i, rel in enumerate(rels[:k], start=1):
        s += (2 ** rel - 1) / math.log2(i + 1)
    return s


def ndcg_at_k(ground_truth: set[int], ranked: list[int], k: int) -> float:
    if not ground_truth:
        return 0.0
    rels = [1 if i in ground_truth else 0 for i in ranked[:k]]
    ideal = sorted(rels, reverse=True)
    dcg = dcg_at_k(rels, k)
    idcg = dcg_at_k(ideal, k)
    return 0.0 if idcg == 0 else dcg / idcg


def test_eval_metrics_simple():
    gt = {2, 5}
    ranked = [2, 1, 3, 5, 7]
    r5 = recall_at_k(gt, ranked, 5)
    n5 = ndcg_at_k(gt, ranked, 5)
    assert 0.0 < r5 <= 1.0
    assert 0.0 < n5 <= 1.0
    # Better ranking improves ndcg
    ranked_better = [2, 5, 1, 3, 7]
    assert ndcg_at_k(gt, ranked_better, 5) >= n5


