from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np

try:
    # CrossEncoder is optional; we will gracefully degrade if unavailable
    from sentence_transformers import CrossEncoder  # type: ignore
except Exception:  # pragma: no cover
    CrossEncoder = None  # type: ignore


_RERANKER: Optional["CrossEncoder"] = None


def get_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    global _RERANKER
    if CrossEncoder is None:
        return None
    if _RERANKER is None:
        try:
            _RERANKER = CrossEncoder(model_name)
        except Exception:
            # Model download or CUDA/torch issues – degrade gracefully
            _RERANKER = None
    return _RERANKER


def rerank(query: str, passages: List[Tuple[int, str]], top_k: int) -> List[int]:
    """Rerank candidate (id, text) passages using a cross-encoder.

    If the reranker is unavailable, returns the first top_k ids in original order.
    """
    if not passages:
        return []
    model = get_reranker()
    if model is None:
        return [cid for cid, _ in passages[:top_k]]
    try:
        pairs = [(query, p) for _, p in passages]
        scores = model.predict(pairs)
        ranked = sorted(zip(passages, scores), key=lambda x: float(x[1]), reverse=True)
        return [cid for (cid, _), _ in ranked[:top_k]]
    except Exception:
        # Any runtime error – degrade gracefully
        return [cid for cid, _ in passages[:top_k]]


def mmr_select(
    query_vec: np.ndarray,
    doc_vecs: np.ndarray,
    candidate_ids: List[int],
    top_k: int,
    lambda_: float = 0.7,
) -> List[int]:
    """Select a diverse subset of candidates using Maximal Marginal Relevance.

    Assumes query_vec and doc_vecs are L2-normalized for cosine.
    """
    if not candidate_ids:
        return []
    if top_k <= 0:
        return []
    # Map candidate id -> row index
    id_to_idx = {cid: cid for cid in range(doc_vecs.shape[0])}

    selected: List[int] = []
    candidates = candidate_ids.copy()

    # Precompute relevance scores q·d
    rel_scores = {}
    for cid in candidates:
        idx = id_to_idx.get(cid)
        if idx is None or idx >= doc_vecs.shape[0]:
            continue
        rel_scores[cid] = float(np.dot(doc_vecs[idx], query_vec.reshape(-1)))

    while candidates and len(selected) < top_k:
        if not selected:
            # pick highest relevance first
            best = max(candidates, key=lambda c: rel_scores.get(c, -1.0))
            selected.append(best)
            candidates.remove(best)
            continue

        best_cid = None
        best_score = -1e9
        for c in list(candidates):
            rel = rel_scores.get(c, -1.0)
            # diversity = max similarity to any already selected
            idx_c = id_to_idx.get(c)
            if idx_c is None or idx_c >= doc_vecs.shape[0]:
                continue
            sel_vecs = doc_vecs[selected]
            div = float(np.max(np.matmul(sel_vecs, doc_vecs[idx_c].reshape(-1, 1))))
            score = lambda_ * rel - (1.0 - lambda_) * div
            if score > best_score:
                best_score = score
                best_cid = c
        if best_cid is None:
            break
        selected.append(best_cid)
        candidates.remove(best_cid)

    return selected


def expand_queries(question: str, n: int = 4) -> List[str]:
    """Lightweight query expansion without external calls."""
    base = (question or "").strip()
    if not base:
        return []
    variants = [
        base,
        f"{base} overview",
        f"{base} key points",
        f"Explain {base}",
        f"What are the main aspects of {base}?",
    ]
    # Deduplicate while preserving order
    seen = set()
    out: List[str] = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            out.append(v)
        if len(out) >= n:
            break
    return out


# -------------------- BM25 Hybrid Retrieval --------------------
try:
    from rank_bm25 import BM25Okapi  # type: ignore
except Exception:  # pragma: no cover
    BM25Okapi = None  # type: ignore


def build_bm25(corpus_texts: List[str]):
    if BM25Okapi is None:
        return None
    if not corpus_texts:
        return None
    tokenized = [t.split() for t in corpus_texts]
    try:
        return BM25Okapi(tokenized)
    except Exception:
        return None


def bm25_search(bm25, query: str, top_n: int) -> List[Tuple[int, float]]:
    if bm25 is None or not query:
        return []
    try:
        scores = bm25.get_scores(query.split())
        idxs = np.argsort(-np.array(scores))[: top_n]
        return [(int(i), float(scores[int(i)])) for i in idxs]
    except Exception:
        return []


def reciprocal_rank_fusion(cand_lists: List[List[Tuple[int, float]]], k: int = 60) -> List[int]:
    """Fuse candidate lists of (doc_id, score) using RRF."""
    from collections import defaultdict

    if not cand_lists:
        return []
    fused = defaultdict(float)
    for cand in cand_lists:
        # sort descending by score, generate ranks
        ranked = sorted(cand, key=lambda x: x[1], reverse=True)
        for r, (doc_id, _) in enumerate(ranked, start=1):
            fused[doc_id] += 1.0 / (k + r)
    # Return doc_ids ordered by fused score
    return [doc_id for doc_id, _ in sorted(fused.items(), key=lambda x: x[1], reverse=True)]


# -------------------- Windowed Retrieval --------------------
def expand_with_neighbors(ids: List[int], total: int, window: int = 1) -> List[int]:
    if not ids:
        return []
    out: List[int] = []
    seen = set()
    for i in ids:
        for delta in range(-window, window + 1):
            j = i + delta
            if 0 <= j < total and j not in seen:
                seen.add(j)
                out.append(j)
    return out


