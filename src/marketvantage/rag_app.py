import json
import pathlib
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np

try:
    import faiss  # type: ignore
except Exception as e:  # pragma: no cover
    faiss = None  # type: ignore

# Ensure package imports work when run directly by Streamlit
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Eager-load .env at app startup so env vars (e.g., GROQ_API_KEY) are present
try:
    from dotenv import load_dotenv
    _dotenv_path = PROJECT_ROOT / ".env"
    if _dotenv_path.exists():
        load_dotenv(dotenv_path=str(_dotenv_path), override=False)
except Exception:
    pass

from marketvantage.you_search import configure_logging, search_you  # noqa: E402
from marketvantage.you_news import search_live_news  # noqa: E402
from marketvantage.you_contents import fetch_contents  # noqa: E402
from marketvantage.llm_groq import generate_answer  # noqa: E402
from marketvantage.retrieval import rerank, mmr_select, expand_queries, build_bm25, bm25_search, reciprocal_rank_fusion, expand_with_neighbors  # noqa: E402


DATA_DIR = pathlib.Path("data/faiss")
DATA_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Chunk:
    text: str
    url: str
    title: str


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return value or "topic"


def _domain_score(url: str) -> float:
    try:
        from urllib.parse import urlparse
        host = urlparse(url).netloc.lower()
    except Exception:
        host = ""
    reputable = {
        "arxiv.org": 2.0,
        "nature.com": 1.8,
        "nytimes.com": 1.7,
        "bbc.com": 1.6,
        "theverge.com": 1.5,
        "wired.com": 1.5,
        "mit.edu": 1.9,
        "stanford.edu": 1.9,
        "github.com": 1.4,
        "docs.google.com": 1.2,
    }
    score = 1.0
    for k, v in reputable.items():
        if host.endswith(k):
            score = max(score, v)
    # Downweight spammy signals
    if any(s in url.lower() for s in ["utm_", "ref=", "?share="]):
        score *= 0.9
    if url.count("/") > 12:
        score *= 0.9
    return score


def extract_urls(search_data: Dict[str, Any], news_data: Dict[str, Any], max_urls: int) -> List[Tuple[str, str]]:
    seen = set()
    urls: List[Tuple[str, str]] = []  # (url, title)

    for item in (search_data.get("results", {}).get("web", []) or [])[: max_urls]:
        url = item.get("url")
        title = item.get("title", "")
        if isinstance(url, str) and url and url not in seen:
            seen.add(url)
            urls.append((url, title))

    for item in (news_data.get("news", {}).get("results", []) or [])[: max_urls]:
        url = item.get("url")
        title = item.get("title", "")
        if isinstance(url, str) and url and url not in seen:
            seen.add(url)
            urls.append((url, title))

    # Apply simple source-quality scoring
    scored = sorted(urls, key=lambda ut: _domain_score(ut[0]), reverse=True)
    return scored[:max_urls]


def _clean_markdown(md: str) -> str:
    # Simple heuristics: remove cookie banners, nav/footer boilerplate, excessive blank lines
    if not md:
        return ""
    lines = [l for l in md.splitlines()]
    drop_phrases = [
        "accept cookies",
        "cookie policy",
        "subscribe",
        "newsletter",
        "privacy policy",
        "terms of service",
        "all rights reserved",
        "sign in",
        "sign up",
        "related articles",
        "advertisement",
    ]
    out: List[str] = []
    seen_footer = False
    for l in lines:
        lo = l.strip().lower()
        if any(p in lo for p in drop_phrases):
            continue
        # Skip very long nav-like lines (many pipes or separators)
        if lo.count("|") >= 4 or lo.count(" · ") >= 2:
            continue
        out.append(l)
    # Collapse multiple blank lines
    cleaned: List[str] = []
    blank = 0
    for l in out:
        if not l.strip():
            blank += 1
            if blank <= 2:
                cleaned.append("")
        else:
            blank = 0
            cleaned.append(l)
    return "\n".join(cleaned).strip()


def _split_markdown_blocks(md: str) -> List[str]:
    # Preserve code fences as atomic blocks, and split by headings
    if not md:
        return []
    lines = md.splitlines()
    blocks: List[str] = []
    buf: List[str] = []
    in_code = False
    fence = ""
    for l in lines:
        if l.strip().startswith("```"):
            if not in_code:
                # start fence – flush current buffer as a block
                if buf:
                    blocks.append("\n".join(buf).strip())
                    buf = []
                in_code = True
                fence = l.strip()
                buf.append(l)
                continue
            else:
                buf.append(l)
                blocks.append("\n".join(buf).strip())
                buf = []
                in_code = False
                fence = ""
                continue
        if not in_code and l.startswith(("# ", "## ", "### ")):
            # heading boundary
            if buf:
                blocks.append("\n".join(buf).strip())
                buf = []
            buf.append(l)
        else:
            buf.append(l)
    if buf:
        blocks.append("\n".join(buf).strip())
    # Remove empties
    return [b for b in blocks if b]


def _estimate_tokens(text: str) -> int:
    # Rough heuristic: ~1 token per 0.75 words in English
    words = [w for w in text.split() if w]
    return max(1, int(len(words) / 0.75))


def chunk_markdown(
    md: str,
    url: str,
    title: str,
    token_target: int = 450,
    overlap_ratio: float = 0.15,
) -> List[Chunk]:
    if not md:
        return []
    cleaned = _clean_markdown(md)
    blocks = _split_markdown_blocks(cleaned)
    chunks: List[Chunk] = []
    cur: List[str] = []
    cur_tokens = 0
    token_overlap = max(1, int(token_target * overlap_ratio))

    def flush_if_needed(force: bool = False) -> None:
        nonlocal cur, cur_tokens
        if not cur:
            return
        if force or cur_tokens >= token_target:
            text = "\n".join(cur).strip()
            if text:
                chunks.append(Chunk(text=text, url=url, title=title))
            cur = []
            cur_tokens = 0

    for b in blocks:
        b_tokens = _estimate_tokens(b)
        # Avoid splitting inside tables/lists if possible; keep small items together
        if cur_tokens + b_tokens <= token_target or not cur:
            cur.append(b)
            cur_tokens += b_tokens
        else:
            # emit current, then start a new one; apply overlap by carrying tail
            flush_if_needed(force=True)
            cur.append(b)
            cur_tokens = b_tokens

    flush_if_needed(force=True)

    # Merge very short chunks into neighbors
    merged: List[Chunk] = []
    i = 0
    while i < len(chunks):
        ch = chunks[i]
        if _estimate_tokens(ch.text) < max(50, int(0.15 * token_target)) and i + 1 < len(chunks):
            nxt = chunks[i + 1]
            merged.append(Chunk(text=f"{ch.text}\n\n{nxt.text}", url=url, title=title))
            i += 2
        else:
            merged.append(ch)
            i += 1
    return merged


def build_corpus(urls_and_titles: List[Tuple[str, str]], return_format: str = "markdown") -> List[Chunk]:
    urls = [u for (u, _) in urls_and_titles]
    title_map = {u: t for (u, t) in urls_and_titles}
    
    try:
        # You.com API has a limit of 10 URLs per request, so batch them
        all_contents = []
        batch_size = 10
        for i in range(0, len(urls), batch_size):
            batch_urls = urls[i:i + batch_size]
            try:
                batch_contents = fetch_contents(batch_urls, format=return_format)
                if isinstance(batch_contents, list):
                    all_contents.extend(batch_contents)
            except Exception as e:
                LOGGER.warning(f"Error fetching contents for batch {i//batch_size + 1}: {e}")
                continue
        contents = all_contents
    except Exception as e:
        LOGGER.error(f"Error fetching contents: {e}", exc_info=True)
        contents = []
    
    all_chunks: List[Chunk] = []
    fetched_urls = set()
    
    for item in contents:
        if not isinstance(item, dict):
            # Skip unexpected item types
            continue
        url = item.get("url") or ""
        if not isinstance(url, str):
            url = str(url)
        fetched_urls.add(url)
        title_val = item.get("title", title_map.get(url, ""))
        title = title_val if isinstance(title_val, str) else str(title_val)
        md_val = item.get("markdown") if return_format == "markdown" else item.get("html")
        md = md_val if isinstance(md_val, str) else (str(md_val) if md_val is not None else "")
        for ch in chunk_markdown(md or "", url, title):
            all_chunks.append(ch)
    
    # Create minimal chunks for URLs where content fetch failed
    for url, title in urls_and_titles:
        if url not in fetched_urls:
            # Create a minimal chunk with just the URL and title
            chunk_text = f"Source: {title}\nURL: {url}\n\n(Content not available)"
            all_chunks.append(Chunk(text=chunk_text, url=url, title=title))
    
    return all_chunks


def embed_chunks(model: SentenceTransformer, chunks: List[Chunk]) -> np.ndarray:
    texts = [c.text for c in chunks]
    if not texts:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32)
    emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return emb.astype(np.float32)


def save_index(topic_slug: str, index: Any, chunks: List[Chunk], *, model_name: str) -> None:
    model_slug = slugify(model_name.replace("/", "-"))
    idx_path = DATA_DIR / f"{topic_slug}__{model_slug}.index"
    meta_path = DATA_DIR / f"{topic_slug}__{model_slug}_meta.json"
    faiss.write_index(index, str(idx_path))  # type: ignore
    meta = [{"url": c.url, "title": c.title} for c in chunks]
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")


def load_index(topic_slug: str, *, model_name: str) -> Tuple[Any, List[Dict[str, str]]]:
    model_slug = slugify(model_name.replace("/", "-"))
    idx_path = DATA_DIR / f"{topic_slug}__{model_slug}.index"
    meta_path = DATA_DIR / f"{topic_slug}__{model_slug}_meta.json"
    if not idx_path.exists() or not meta_path.exists():
        raise FileNotFoundError
    index = faiss.read_index(str(idx_path))  # type: ignore
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return index, meta


def ensure_index(topic: str, max_urls: int, freshness: str) -> Tuple[Any, List[Chunk]]:
    if faiss is None:
        raise RuntimeError("faiss is not available. Ensure faiss-cpu is installed.")

    topic_slug = slugify(topic)

    model_name = st.session_state.get("embedding_model") or "sentence-transformers/all-MiniLM-L6-v2"
    try:
        index, meta = load_index(topic_slug, model_name=model_name)
        st.session_state["faiss_index"] = index
        st.session_state["faiss_meta"] = meta
        # Build chunks for BM25 and display
        urls_and_titles = [(m.get("url", ""), m.get("title", "")) for m in meta]
        chunks = build_corpus(urls_and_titles, return_format="markdown")
        st.session_state["faiss_chunks"] = chunks
        # Build BM25 index
        bm25 = build_bm25([c.text for c in chunks])
        st.session_state["bm25_index"] = bm25
        st.session_state["topic_slug"] = topic_slug
        return index, chunks
    except FileNotFoundError:
        pass

    search_data = search_you(topic, count=max_urls, freshness=freshness)
    news_data = search_live_news(topic, count=max_urls)
    urls_and_titles = extract_urls(search_data, news_data, max_urls)

    chunks = build_corpus(urls_and_titles, return_format="markdown")

    model = get_model(model_name)
    embeddings = embed_chunks(model, chunks)
    d = embeddings.shape[1] if embeddings.size else 384
    n = len(chunks)
    
    # Choose index based on corpus size
    if n >= 50000:
        # IVF+PQ for very large corpora with compression
        # nlist: number of clusters (Voronoi cells) - typically sqrt(n) or n^(1/3)
        nlist = min(int(np.sqrt(n)), 4096)
        # m: number of sub-vectors for PQ (dimension must be divisible by m)
        # nbits: codebook size (2^nbits centroids per sub-vector)
        m = 8  # split into 8 sub-vectors
        nbits = 8  # 256 centroids per sub-vector
        
        # Adjust m to ensure d is divisible by m
        while d % m != 0 and m > 1:
            m -= 1
        
        quantizer = faiss.IndexFlatIP(d)  # type: ignore
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)  # type: ignore
        index.nprobe = min(nlist // 4, 64)  # search in top clusters
        
        # Train on a sample: FAISS requires at least 39 * nlist training vectors
        min_train_size = max(39 * nlist, 1000)
        train_sample_size = min(n, max(min_train_size, 256000))
        train_ids = np.random.choice(n, size=train_sample_size, replace=False)
        train_vectors = embeddings[train_ids]
        index.train(train_vectors)  # type: ignore
    elif n > 20000:
        # HNSW for medium-large corpora
        hnsw = faiss.IndexHNSWFlat(d, 32)  # type: ignore
        hnsw.hnsw.efConstruction = 200  # type: ignore
        index = hnsw
    else:
        # Flat index for small corpora (exact search)
        index = faiss.IndexFlatIP(d)  # type: ignore  # cosine if embeddings normalized
    
    if embeddings.size:
        index.add(embeddings)  # type: ignore

    save_index(topic_slug, index, chunks, model_name=model_name)

    st.session_state["faiss_index"] = index
    st.session_state["faiss_chunks"] = chunks
    st.session_state["faiss_embeddings"] = embeddings
    st.session_state["bm25_index"] = build_bm25([c.text for c in chunks])
    st.session_state["topic_slug"] = topic_slug

    return index, chunks


@st.cache_resource(show_spinner=False)
def get_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    return SentenceTransformer(model_name)


def _ensure_embeddings(chunks: List[Chunk]) -> np.ndarray:
    emb = st.session_state.get("faiss_embeddings")
    if emb is not None and isinstance(emb, np.ndarray) and emb.size:
        return emb
    # Compute on-demand and cache
    model_name = st.session_state.get("embedding_model") or "sentence-transformers/all-MiniLM-L6-v2"
    model = get_model(model_name)
    emb = embed_chunks(model, chunks)
    st.session_state["faiss_embeddings"] = emb
    return emb


def search_top_k(query: str, k: int) -> List[int]:
    index = st.session_state.get("faiss_index")
    chunks: List[Chunk] = st.session_state.get("faiss_chunks", [])
    if index is None or not chunks:
        return []

    model_name = st.session_state.get("embedding_model") or "sentence-transformers/all-MiniLM-L6-v2"
    model = get_model(model_name)
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    topN = max(k * 6, 30)

    # Multi-query expansion
    all_dense_ids: List[int] = []
    for q in expand_queries(query, n=4):
        q_vec = model.encode([q], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        D, I = index.search(q_vec, topN)  # type: ignore
        if I.size:
            all_dense_ids.extend([i for i in I[0] if 0 <= i < len(chunks)])

    # BM25 candidates
    bm25 = st.session_state.get("bm25_index")
    bm25_cands = bm25_search(bm25, query, topN)
    bm25_ids = [i for i, _ in bm25_cands]

    # Fuse
    dense_scored = [(i, float(topN - r)) for r, i in enumerate(all_dense_ids[:topN], start=1)]
    fused_ids = reciprocal_rank_fusion([dense_scored, bm25_cands])
    # keep only ids in range
    cand_ids = [i for i in fused_ids if 0 <= i < len(chunks)]
    if not cand_ids:
        # fallback to dense
        cand_ids = list(dict.fromkeys(all_dense_ids))

    if not cand_ids:
        return []

    # Rerank
    cand_texts = [(i, chunks[i].text) for i in cand_ids]
    reranked_ids = rerank(query, cand_texts, top_k=min(len(cand_ids), max(k, 10)))

    # MMR diversity
    doc_vecs = _ensure_embeddings(chunks)
    mmr_ids = mmr_select(q_emb.reshape(-1), doc_vecs, reranked_ids, top_k=k, lambda_=0.7)
    if mmr_ids:
        return mmr_ids[:k]
    return reranked_ids[:k]


def render():
    configure_logging(verbosity=1)
    st.set_page_config(page_title="RAG on Web + News", layout="wide")
    st.title("RAG: Web + Live News")

    with st.sidebar:
        st.markdown("Configure and Build Index")
        topic = st.text_input("Topic", value="ai lip sync")
        max_urls = st.slider("Max URLs", 1, 20, 8)
        freshness = st.selectbox("Freshness", ["day", "week", "month", "year"], index=3)
        embed_model = st.selectbox(
            "Embedding model",
            [
                "sentence-transformers/all-MiniLM-L6-v2",
                # "intfloat/e5-base-v2",
                # "BAAI/bge-base-en-v1.5",
                # "thenlper/gte-large",
            ],
            index=0,
        )
        if st.button("Build / Load Index", type="primary"):
            with st.spinner("Building index (or loading cached)..."):
                st.session_state["embedding_model"] = embed_model
                index, chunks = ensure_index(topic, max_urls, freshness)
                st.success("Index ready.")

        st.divider()
        st.markdown("Ask a question about the topic")
        question = st.text_input("Question", value="What tools are used for AI lip sync?")
        top_k = st.slider("Top-K passages", 1, 10, 5)
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Search"):
                with st.spinner("Retrieving..."):
                    ids = search_top_k(question, top_k)
                    st.session_state["last_results"] = ids
        with col_b:
            if st.button("Generate Answer with Groq"):
                import os as _os
                import logging as _logging
                _logging.getLogger("marketvantage.rag_app").info(
                    "Groq button pressed", extra={"GROQ_API_KEY_present": bool(_os.getenv("GROQ_API_KEY"))}
                )
                ids = st.session_state.get("last_results") or search_top_k(question, top_k)
                chunks: List[Chunk] = st.session_state.get("faiss_chunks", [])
                # Windowed retrieval: include neighbors for local context
                win_ids = expand_with_neighbors(ids, total=len(chunks), window=1)
                selected = [
                    {"text": chunks[i].text, "title": chunks[i].title, "url": chunks[i].url}
                    for i in win_ids
                    if 0 <= i < len(chunks)
                ]
                with st.spinner("Generating answer..."):
                    try:
                        answer = generate_answer(question, selected)
                        st.session_state["last_answer"] = answer
                    except Exception as e:
                        st.error(f"Groq error: {e}")

    st.markdown("### Retrieved Passages")
    ids: List[int] = st.session_state.get("last_results", [])
    chunks: List[Chunk] = st.session_state.get("faiss_chunks", [])
    meta = st.session_state.get("faiss_meta")

    if meta and not chunks:
        st.info("Loading contents for display...")
        urls_and_titles = [(m.get("url", ""), m.get("title", "")) for m in meta]
        chunks = build_corpus(urls_and_titles, return_format="markdown")
        st.session_state["faiss_chunks"] = chunks

    if ids and chunks:
        for idx in ids:
            if 0 <= idx < len(chunks):
                ch = chunks[idx]
                st.markdown(f"**Title:** {ch.title}")
                st.markdown(f"**URL:** {ch.url}")
                st.code(ch.text, language="markdown")
                st.markdown("---")
    else:
        st.write("No results yet. Build the index and ask a question.")

    answer = st.session_state.get("last_answer")
    if answer:
        st.markdown("### Groq Answer")
        st.write(answer)


if __name__ == "__main__":
    render()
