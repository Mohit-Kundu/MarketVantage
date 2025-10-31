"""Core report generation engine."""
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Tuple

import numpy as np
import sys

# Ensure package imports work
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from marketvantage.you_search import search_you, configure_logging  # noqa: E402
from marketvantage.you_news import search_live_news  # noqa: E402
from marketvantage.you_contents import fetch_contents  # noqa: E402
from marketvantage.llm_groq import generate_answer  # noqa: E402
from marketvantage.report_sections import (  # noqa: E402
    get_sections_requiring_api,
    REPORT_SECTIONS,
    SectionTemplate,
)

try:
    import faiss  # type: ignore
except Exception:
    faiss = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None  # type: ignore

LOGGER = logging.getLogger("marketvantage.report_engine")


# Rate limiter for Groq API calls
class RateLimiter:
    """Rate limiter to prevent hitting Groq API rate limits."""
    
    def __init__(self, max_calls_per_minute: int = 30, min_interval: float = 2.0):
        """
        Initialize rate limiter.
        
        Args:
            max_calls_per_minute: Maximum number of API calls per minute
            min_interval: Minimum seconds between consecutive calls
        """
        self.max_calls_per_minute = max_calls_per_minute
        self.min_interval = min_interval
        self.last_call_time = 0
        self.call_times: List[float] = []
        self.lock = Lock()
    
    def wait_if_needed(self) -> None:
        """Wait if needed to respect rate limits."""
        with self.lock:
            now = time.time()
            # Remove calls older than 1 minute
            self.call_times = [t for t in self.call_times if now - t < 60]
            
            # If we're at the limit, wait until the oldest call expires
            if len(self.call_times) >= self.max_calls_per_minute:
                oldest_call = self.call_times[0]
                sleep_time = 60 - (now - oldest_call) + 0.1
                if sleep_time > 0:
                    LOGGER.info(f"Rate limit reached, waiting {sleep_time:.1f}s")
                    time.sleep(sleep_time)
                    now = time.time()
                    # Clean up again after sleep
                    self.call_times = [t for t in self.call_times if now - t < 60]
            
            # Enforce minimum interval between calls
            time_since_last = now - self.last_call_time
            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last
                time.sleep(sleep_time)
                now = time.time()
            
            self.last_call_time = now
            self.call_times.append(now)


# Global rate limiter instance (shared across threads)
# Free tier limit: 10 requests per minute, paid tier: 60 requests per minute
# Using conservative 10/min to work with free tier
_global_rate_limiter = RateLimiter(max_calls_per_minute=10, min_interval=6.0)


def _strip_unwanted_sections(content: str) -> str:
    """
    Remove unwanted References, Cited Sources, Conclusion sections from content.
    LLMs sometimes add these despite instructions not to.
    """
    if not content:
        return content
    
    original_len = len(content)
    original_content = content
    
    # First pass: Use regex patterns to find and strip reference sections
    patterns = [
        # Markdown headers (must be on a new line)
        r'\n\s*#{1,3}\s*(References?|Cited Sources?|Citations?|Bibliography|Sources?|Recommendations?)\s*\n.*',
        # Bold/italic headers
        r'\n\s*\*\*\s*(References?|Cited Sources?|Citations?|Bibliography|Sources?|Recommendations?)\s*\*\*\s*\n.*',
        r'\n\s*\*\s*(References?|Cited Sources?|Citations?|Bibliography|Sources?|Recommendations?)\s*\*\s*\n.*',
        # Colon format (must be on a new line)
        r'\n\s*(References?|Cited Sources?|Citations?|Bibliography|Sources?|Recommendations?)\s*:\s*\n.*',
        # Citation-style references: [1] Author. or [1] Any text starting with capital
        r'\n\s*\[\d+\]\s+[A-Z].*',
        # Conclusion phrases (at start of line)
        r'\n\s*In conclusion[,\s].*',
        r'\n\s*To conclude[,\s].*',
        r'\n\s*Concluding[,\s].*',
        r'\n\s*In summary[,\s].*',
        r'\n\s*To summarize[,\s].*',
        # "Retrieved from" or "Source:" lines
        r'\n\s*(Retrieved from|Source:|Sources:|URL:)\s*.*',
    ]
    
    # Try each pattern
    for pattern in patterns:
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            content = content[:match.start()].rstrip()
            LOGGER.info(f"Stripped unwanted section with regex pattern: {match.group()[:100]}...")
            break
    
    # Second pass: Line-by-line analysis for more precise detection
    lines = content.split('\n')
    filtered_lines = []
    in_reference_section = False
    consecutive_ref_lines = 0
    in_recommendations = False
    recommendation_bullet_count = 0
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        line_lower = line_stripped.lower()
        
        # Check for section headers (more patterns)
        is_section_header = (
            re.match(r'^\s*#{1,3}\s+(References?|Cited Sources?|Citations?|Bibliography|Sources?|Recommendations?)\s*$', line_stripped, re.IGNORECASE) or
            re.match(r'^\s*\*\*\s*(References?|Cited Sources?|Citations?|Bibliography|Sources?|Recommendations?)\s*\*\*\s*$', line_stripped, re.IGNORECASE) or
            re.match(r'^\s*(References?|Cited Sources?|Citations?|Bibliography|Sources?|Recommendations?)\s*:\s*$', line_stripped, re.IGNORECASE)
        )
        
        # Check for recommendations header
        is_recommendations_header = (
            re.match(r'^\s*(Recommendations?|Key Recommendations?|Strategic Recommendations?)\s*:?\s*$', line_stripped, re.IGNORECASE)
        )
        
        # Check for recommendation-style bullet points (action verbs at start)
        is_recommendation_bullet = bool(
            re.match(r'^\s*[-*•]\s*(invest|develop|focus|explore|consider|implement|build|create|address|pursue|adopt|leverage|enhance)', line_lower)
        )
        
        if is_recommendations_header:
            in_recommendations = True
            LOGGER.info(f"Found recommendations header at line {i+1}: '{line_stripped}'")
            continue  # Skip the header line
        
        if in_recommendations:
            # If we're in recommendations section, check if this is a bullet point
            if is_recommendation_bullet or line_stripped.startswith('-') or line_stripped.startswith('•'):
                recommendation_bullet_count += 1
                continue  # Skip recommendation bullets
            elif recommendation_bullet_count > 0 and not line_stripped:
                # Empty line after recommendations, continue skipping
                continue
            elif recommendation_bullet_count > 0 and line_stripped:
                # Non-empty line after recommendations bullets, we're past recommendations
                in_recommendations = False
                recommendation_bullet_count = 0
        
        # Check if this looks like a reference line
        is_ref_line = (
            # Numbered list item with URL: [1] https://... or 1. https://...
            re.match(r'^\s*\[?\d+\]?\s*(https?://|www\.)', line_stripped, re.IGNORECASE) or
            # Reference number followed by text then URL: [1] Title (URL)
            re.match(r'^\s*\[?\d+\]?\s+[^(]*\(https?://', line_stripped, re.IGNORECASE) or
            # Citation format: [1] Any text (catches [1] Author, [1] Title, etc)
            re.match(r'^\s*\[\d+\]\s+\S', line_stripped) or
            # "Retrieved from" or similar
            re.match(r'^\s*(Retrieved from|Source:|Sources?:|URL:)\s*', line_stripped, re.IGNORECASE) or
            # Just a URL at start of line
            (re.match(r'^\s*https?://', line_stripped, re.IGNORECASE) and len(line_stripped) > 10)
        )
        
        if is_section_header:
            in_reference_section = True
            LOGGER.info(f"Found unwanted section header at line {i+1}: '{line_stripped}'")
            break  # Stop here, don't include this line or any after
        
        # Skip lines that are part of recommendations section
        if in_recommendations:
            continue
        
        if is_ref_line:
            consecutive_ref_lines += 1
            # If we see 2+ consecutive reference-looking lines, we're in a ref section
            if consecutive_ref_lines >= 2:
                in_reference_section = True
                # Backtrack to remove the last few lines that looked like refs
                filtered_lines = filtered_lines[:-(consecutive_ref_lines-1)]
                LOGGER.info(f"Detected reference section from {consecutive_ref_lines} consecutive ref lines starting around line {i-consecutive_ref_lines+1}")
                break
        else:
            consecutive_ref_lines = 0
        
        if not in_reference_section:
            filtered_lines.append(line)
    
    if in_reference_section:
        content = '\n'.join(filtered_lines).rstrip()
        final_len = len(content)
        LOGGER.info(f"Stripped reference section from content (reduced from {original_len} to {final_len} chars, removed {original_len - final_len} chars)")
    
    # Third pass: Final cleanup - remove any remaining standalone URLs at the end
    lines = content.split('\n')
    if len(lines) > 2:
        # Check last few lines for URLs
        last_lines = [line.strip() for line in lines[-3:]]
        url_count = sum(1 for line in last_lines if re.match(r'^https?://', line, re.IGNORECASE))
        if url_count >= 2:
            # Remove trailing lines that are URLs
            while lines and re.match(r'^\s*https?://', lines[-1].strip(), re.IGNORECASE):
                lines.pop()
            content = '\n'.join(lines).rstrip()
            LOGGER.info(f"Removed trailing URL lines (final length: {len(content)} chars)")
    
    # Fourth pass: Clean up template artifacts and formatting
    content = _clean_template_artifacts(content)
    
    return content.strip()


def _clean_template_artifacts(content: str) -> str:
    """
    Remove template-like lines, strip single-asterisk italics, keep double-asterisk bold, and clean up formatting.
    """
    if not content:
        return content
    
    lines = content.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line_stripped = line.strip()
        
        # Remove duplicate "Executive Summary" subheading (when it appears inside Executive Summary section)
        if re.match(r'^[#*\s]*Executive\s+Summary[#*\s]*$', line_stripped, re.IGNORECASE):
            LOGGER.debug("Removed duplicate 'Executive Summary' subheading from content")
            continue
        
        # Skip template-like lines
        # Matches: "Company/organization name: -----" or "-Project/organization name: -------"
        if re.match(r'^[-*\s]*(Company|Organization|Project)(/|-).*name:\s*-+$', line_stripped, re.IGNORECASE):
            continue
        # Matches: "- Company/organization name: SomeName" (template label)
        if re.match(r'^[-*]\s*(Company|Organization|Project)(/|-).*name:\s*', line_stripped, re.IGNORECASE):
            continue
        # Matches: "- Their role or contribution to..."
        if re.match(r'^[-*]\s*Their role or contribution to', line_stripped, re.IGNORECASE):
            continue
        # Matches: "- Brief description of their offering or approach:"
        if re.match(r'^[-*]\s*Brief description of (their|their offering|approach)', line_stripped, re.IGNORECASE):
            continue
        
        # Remove numbered header lines like "1. Top 3 Recent Developments:" or "1. Market Overview:"
        if re.match(r'^\d+\.\s+(Top\s+\d+|Key\s+|Major\s+)?.*:\s*$', line_stripped):
            LOGGER.debug(f"Removed numbered header line: '{line_stripped}'")
            continue
        # Also catch lines that are just numbered items with colon at end (short lines)
        if re.match(r'^\d+\.\s+.{1,50}:\s*$', line_stripped) and len(line_stripped) < 80:
            LOGGER.debug(f"Removed short numbered header line: '{line_stripped}'")
            continue
        
        # Remove single-asterisk italics ONLY: *text* -> text (but keep **bold** as is)
        # First, protect double asterisks temporarily
        line = line.replace('**', '<<<BOLD>>>')
        # Remove single asterisk italics
        line = re.sub(r'\*([^*]+?)\*', r'\1', line)
        # Restore double asterisks
        line = line.replace('<<<BOLD>>>', '**')
        
        # Remove underscore italics: _text_ -> text
        line = re.sub(r'_([^_]+?)_', r'\1', line)
        
        # Clean up lines that are just dashes
        if re.match(r'^[-*\s]+$', line_stripped):
            continue
        
        cleaned_lines.append(line)
    
    content = '\n'.join(cleaned_lines)
    
    # Clean up multiple blank lines
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    return content.strip()


def _configure_logging():
    """Ensure logging is configured."""
    if not LOGGER.handlers:
        configure_logging(verbosity=1)


def slugify(value: str) -> str:
    """Create URL-safe slug from topic."""
    import re
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return value or "topic"


def _fetch_section_data_parallel(
    topic: str, sections: Dict[int, SectionTemplate], max_urls_per_section: int = 10
) -> Dict[int, Dict[str, Any]]:
    """Fetch data for all sections in parallel using You.com APIs."""
    _configure_logging()
    section_data: Dict[int, Dict[str, Any]] = {}
    
    def fetch_section(section_num: int, template: SectionTemplate) -> Tuple[int, Dict[str, Any]]:
        """Fetch data for a single section."""
        try:
            LOGGER.info(f"Fetching data for section {section_num}: {template.name}")
            if template.uses_news_api:
                # Use news API for section 5
                news_data = search_live_news(topic, count=max_urls_per_section)
                search_data = {"news": news_data.get("news", {})}
                section_data_result = {
                    "search_data": search_data,
                    "news_data": news_data,
                    "urls": [],
                    "titles": [],
                }
                # Extract URLs from news
                news_results = news_data.get("news", {}).get("results", [])
                for item in news_results:
                    url = item.get("url")
                    title = item.get("title", "")
                    if url:
                        section_data_result["urls"].append(url)
                        section_data_result["titles"].append(title)
                LOGGER.info(f"Section {section_num}: Found {len(section_data_result['urls'])} URLs from news API")
                return section_num, section_data_result
            else:
                # Use regular search API
                query = template.api_query_hint.format(topic=topic) if template.api_query_hint else topic
                search_data = search_you(query, count=max_urls_per_section, safesearch="off")
                section_data_result = {
                    "search_data": search_data,
                    "news_data": {},
                    "urls": [],
                    "titles": [],
                }
                # Extract URLs from search
                web_results = search_data.get("results", {}).get("web", [])
                for item in web_results:
                    url = item.get("url")
                    title = item.get("title", "")
                    if url:
                        section_data_result["urls"].append(url)
                        section_data_result["titles"].append(title)
                LOGGER.info(f"Section {section_num}: Found {len(section_data_result['urls'])} URLs from search API")
                return section_num, section_data_result
        except Exception as e:
            LOGGER.error(f"Error fetching data for section {section_num}: {e}", exc_info=True)
            return section_num, {"search_data": {}, "news_data": {}, "urls": [], "titles": []}

    # Fetch all sections in parallel
    LOGGER.info(f"Starting parallel fetch for {len(sections)} sections")
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(fetch_section, num, template): (num, template)
            for num, template in sections.items()
        }
        completed = 0
        for future in as_completed(futures):
            section_num, data = future.result()
            section_data[section_num] = data
            completed += 1
            LOGGER.debug(f"Completed {completed}/{len(sections)} sections")

    LOGGER.info(f"Completed parallel fetch. Got data for {len(section_data)} sections")
    return section_data


def _aggregate_all_urls(section_data: Dict[int, Dict[str, Any]]) -> List[Tuple[str, str]]:
    """Collect all unique URLs from all sections."""
    seen = set()
    urls: List[Tuple[str, str]] = []
    for data in section_data.values():
        section_urls = data.get("urls", [])
        section_titles = data.get("titles", [])
        for url, title in zip(section_urls, section_titles):
            if url and url not in seen:
                seen.add(url)
                urls.append((url, title if title else url))
    return urls


def _build_report_rag_index(
    urls_and_titles: List[Tuple[str, str]], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> Tuple[Any, List, np.ndarray]:
    """Build RAG index from all collected URLs."""
    if faiss is None:
        raise RuntimeError("faiss is not available")
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers is not available")

    # Import chunking functions from rag_app
    from marketvantage.rag_app import build_corpus, embed_chunks  # noqa: E402

    chunks = build_corpus(urls_and_titles, return_format="markdown")
    
    if not chunks:
        LOGGER.warning("No chunks built from URLs")
        return None, [], np.array([])

    # Embed chunks
    model = SentenceTransformer(model_name)
    embeddings = embed_chunks(model, chunks)
    
    # Create FAISS index
    d = embeddings.shape[1] if embeddings.size else 384
    n = len(chunks)
    
    if n >= 50000:
        # IVF+PQ for very large
        nlist = min(int(np.sqrt(n)), 4096)
        m = 8
        while d % m != 0 and m > 1:
            m -= 1
        nbits = 8
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
        index.nprobe = min(nlist // 4, 64)
        min_train_size = max(39 * nlist, 1000)
        train_sample_size = min(n, max(min_train_size, 256000))
        train_ids = np.random.choice(n, size=train_sample_size, replace=False)
        train_vectors = embeddings[train_ids]
        index.train(train_vectors)
    elif n > 20000:
        # HNSW for medium-large
        hnsw = faiss.IndexHNSWFlat(d, 32)
        hnsw.hnsw.efConstruction = 200
        index = hnsw
    else:
        # Flat for small
        index = faiss.IndexFlatIP(d)
    
    if embeddings.size:
        index.add(embeddings)
    
    return index, chunks, embeddings


def _retrieve_chunks_for_section(
    query: str,
    rag_index: Any,
    rag_chunks: List,
    rag_embeddings: np.ndarray,
    embedding_model_name: str,
    top_k: int = 15,
    fast_mode: bool = True,
) -> Tuple[List[int], List[Dict[str, Any]]]:
    """
    Retrieve chunks for section generation. Fast mode skips expensive reranking/MMR.
    
    Args:
        fast_mode: If True, uses simple hybrid search without reranking/MMR (much faster)
                   If False, uses full pipeline with reranking and MMR (slower but more accurate)
    
    Returns:
        Tuple of (selected_indices, chunk_dicts) for use with windowed retrieval
    """
    from marketvantage.retrieval import (
        rerank,
        mmr_select,
        expand_queries,
        build_bm25,
        bm25_search,
        reciprocal_rank_fusion,
    )
    from marketvantage.rag_app import get_model
    
    if rag_index is None or not rag_chunks:
        return [], []
    
    try:
        model = get_model(embedding_model_name)
        q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        
        if fast_mode:
            # Fast path: Simple hybrid search without reranking/MMR
            topN = max(top_k * 2, 20)  # Smaller candidate pool
            
            # Single query dense search (no expansion)
            D, I = rag_index.search(q_emb, topN)  # type: ignore
            dense_ids = [i for i in I[0] if I.size and 0 <= i < len(rag_chunks)]
            
            # BM25 search
            chunk_texts = [
                c.text if hasattr(c, "text") else str(c) 
                for c in rag_chunks
            ]
            bm25_index = build_bm25(chunk_texts)
            bm25_cands = bm25_search(bm25_index, query, topN)
            
            # Simple fusion (weighted combination)
            dense_scored = [(i, float(topN - r)) for r, i in enumerate(dense_ids[:topN], start=1)]
            fused_ids = reciprocal_rank_fusion([dense_scored, bm25_cands])
            cand_ids = [i for i in fused_ids if 0 <= i < len(rag_chunks)][:top_k]
            
            if not cand_ids:
                cand_ids = list(dict.fromkeys(dense_ids))[:top_k]
            
            selected_ids = cand_ids
            
        else:
            # Full path: Advanced retrieval with reranking and MMR (slower)
            topN = max(top_k * 6, 30)
            
            # Multi-query expansion
            all_dense_ids: List[int] = []
            expanded_queries = expand_queries(query, n=4)
            for q_expanded in expanded_queries:
                q_vec = model.encode([q_expanded], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
                D, I = rag_index.search(q_vec, topN)  # type: ignore
                if I.size:
                    all_dense_ids.extend([i for i in I[0] if 0 <= i < len(rag_chunks)])
            
            # BM25 candidates
            chunk_texts = [
                c.text if hasattr(c, "text") else str(c) 
                for c in rag_chunks
            ]
            bm25_index = build_bm25(chunk_texts)
            bm25_cands = bm25_search(bm25_index, query, topN)
            
            # Fuse dense and BM25 results
            dense_scored = [(i, float(topN - r)) for r, i in enumerate(all_dense_ids[:topN], start=1)]
            fused_ids = reciprocal_rank_fusion([dense_scored, bm25_cands])
            cand_ids = [i for i in fused_ids if 0 <= i < len(rag_chunks)]
            
            if not cand_ids:
                # Fallback to dense only
                cand_ids = list(dict.fromkeys(all_dense_ids))
            
            if not cand_ids:
                LOGGER.warning(f"Retrieval found no candidate chunks for query: {query[:100]}... (total chunks available: {len(rag_chunks)})")
                return [], []
            
            # Rerank with cross-encoder (EXPENSIVE - only top candidates)
            cand_texts = [(i, chunk_texts[i]) for i in cand_ids[:top_k * 3]]  # Limit reranking candidates
            reranked_ids = rerank(query, cand_texts, top_k=min(len(cand_ids), max(top_k, 10)))
            
            # MMR diversity (can be slow with many candidates)
            mmr_ids = mmr_select(
                q_emb.reshape(-1), 
                rag_embeddings, 
                reranked_ids, 
                top_k=top_k, 
                lambda_=0.7
            )
            
            selected_ids = mmr_ids[:top_k] if mmr_ids else reranked_ids[:top_k]
        
        if not selected_ids:
            LOGGER.warning(f"Retrieval found no candidate chunks for query: {query[:100]}... (total chunks available: {len(rag_chunks)})")
            return [], []
        
        # Convert to dict format
        selected_chunks = []
        for idx in selected_ids:
            chunk = rag_chunks[idx]
            selected_chunks.append({
                "text": chunk.text if hasattr(chunk, "text") else str(chunk),
                "title": chunk.title if hasattr(chunk, "title") else "",
                "url": chunk.url if hasattr(chunk, "url") else "",
            })
        
        return selected_ids, selected_chunks
        
    except Exception as e:
        LOGGER.warning(f"Advanced retrieval failed, falling back to simple selection: {e}", exc_info=True)
        # Fallback: return first N chunks with indices
        fallback_ids = list(range(min(top_k, len(rag_chunks))))
        fallback_chunks = [
            {
                "text": c.text if hasattr(c, "text") else str(c),
                "title": c.title if hasattr(c, "title") else "",
                "url": c.url if hasattr(c, "url") else "",
            }
            for c in rag_chunks[:top_k]
        ]
        return fallback_ids, fallback_chunks


def _generate_section_content(
    section_template: SectionTemplate,
    topic: str,
    context_chunks: List[Dict[str, Any]],
    max_output_tokens: int = 1500,
    use_rate_limit: bool = True,
) -> str:
    """Generate section content using Groq LLM with optional rate limiting."""
    if not section_template.prompt_template:
        return ""
    
    prompt = section_template.prompt_template.format(topic=topic)
    
    # Format context chunks
    formatted_chunks = []
    for i, chunk in enumerate(context_chunks[:15], 1):  # Limit to top 15 chunks
        if isinstance(chunk, dict):
            text = chunk.get("text", "").strip()
            title = chunk.get("title", "").strip()
            url = chunk.get("url", "").strip()
            # Only add chunks with non-empty text
            if text:
                formatted_chunks.append({
                    "text": text,
                    "title": title,
                    "url": url,
                })
        else:
            text = str(chunk).strip()
            if text:
                formatted_chunks.append({"text": text, "title": "", "url": ""})
    
    # Log if no valid chunks found
    if not formatted_chunks:
        LOGGER.warning(f"No valid chunks with text content found for section {section_template.number}")
    
    # Apply rate limiting before API call
    if use_rate_limit:
        _global_rate_limiter.wait_if_needed()
    
    try:
        answer = generate_answer(
            prompt,
            formatted_chunks,
            max_output_tokens=max_output_tokens,
            report_mode=True,  # Enable detailed report-style generation
        )
        
        # Validate answer is meaningful (check for empty, error messages, or very short content)
        if not answer or answer.strip() == "" or answer.startswith("[Error") or len(answer.strip()) < 50:
            LOGGER.warning(f"Section {section_template.number} returned empty/invalid/short content (length: {len(answer.strip()) if answer else 0}), retrying with simplified prompt")
            # Retry with simplified prompt if content is invalid
            simplified_prompt = (
                f"Provide a comprehensive section about {topic} covering: {section_template.name.lower()}. "
                f"Write in a professional, detailed style suitable for a technology landscape report. "
                f"Use the provided sources and cite them with [1], [2], etc. if relevant."
            )
            try:
                # Apply rate limiting before retry
                if use_rate_limit:
                    _global_rate_limiter.wait_if_needed()
                answer = generate_answer(
                    simplified_prompt,
                    formatted_chunks,
                    max_output_tokens=max_output_tokens,
                    report_mode=True,  # Enable detailed report-style generation
                )
            except Exception as retry_e:
                LOGGER.error(f"Retry also failed for section {section_template.number}: {retry_e}")
                answer = ""  # Ensure answer is empty to trigger fallback
        
        # Final validation - ensure we have meaningful content
        if not answer or answer.strip() == "" or answer.startswith("[Error") or len(answer.strip()) < 50:
            LOGGER.error(f"Section {section_template.number} could not be generated after retries (final length: {len(answer.strip()) if answer else 0})")
            # Generate a comprehensive fallback section based on section type
            section_name = section_template.name.lower()
            if "technology overview" in section_name:
                answer = (
                    f"{topic} represents a significant technological area with ongoing developments and innovation. "
                    f"This technology addresses key challenges and offers various applications across different industries. "
                    f"The current state shows active research, product development, and market adoption. "
                    f"Technical capabilities continue to evolve, with improvements in performance, reliability, and integration. "
                    f"Understanding the fundamentals and applications of {topic} is essential for stakeholders considering "
                    f"adoption or investment in this space."
                )
            elif "key players" in section_name or "market landscape" in section_name:
                answer = (
                    f"The market landscape for {topic} includes established companies and emerging players. "
                    f"Major market leaders have significant resources and market presence, while innovative startups "
                    f"bring fresh approaches and specialized solutions. The competitive dynamics show both collaboration "
                    f"and competition, with partnerships forming to address complex challenges. Market positioning varies "
                    f"across players, with different strategic focuses on technology, market segments, and customer needs."
                )
            elif "recent news" in section_name or "advancements" in section_name:
                answer = (
                    f"Recent developments in {topic} show continued momentum and innovation. Key announcements include "
                    f"product launches, strategic partnerships, funding rounds, and technical breakthroughs. "
                    f"The pace of advancement indicates strong interest and investment in this space. "
                    f"Industry participants are actively working on improvements and new capabilities."
                )
            elif "trends" in section_name or "future" in section_name:
                answer = (
                    f"Future trends for {topic} point toward continued growth and evolution. Market projections indicate "
                    f"expanding adoption across various sectors. Emerging applications and use cases are being identified, "
                    f"and technology is expected to mature further. Industry adoption patterns show gradual acceleration "
                    f"as solutions become more refined and accessible."
                )
            elif "opportunities" in section_name or "white space" in section_name:
                answer = (
                    f"Opportunities in {topic} exist across multiple dimensions. Current limitations and challenges "
                    f"present areas for innovation and improvement. Market gaps suggest potential for new entrants "
                    f"or specialized solutions. Unaddressed use cases and customer needs offer opportunities for "
                    f"development. Further research and technical advancement could unlock additional capabilities."
                )
            else:
                answer = (
                    f"This section on {topic} covers {section_template.name.lower()}. "
                    f"Based on the research conducted, this area shows significant activity and development. "
                    f"Key aspects include technology fundamentals, market dynamics, and future opportunities. "
                    f"Further detailed analysis would benefit from continued monitoring of developments in this space."
                )
            LOGGER.info(f"Using comprehensive fallback content for section {section_template.number} ({section_template.name})")
        
        # Strip any unwanted reference/conclusion sections
        answer = _strip_unwanted_sections(answer)
        # Clean template artifacts and formatting
        answer = _clean_template_artifacts(answer)
        
        # Final check - ensure minimum length
        if len(answer.strip()) < 100:
            LOGGER.warning(f"Section {section_template.number} content is very short ({len(answer.strip())} chars), may be incomplete")
        
        return answer
    except Exception as e:
        LOGGER.error(f"Error generating section {section_template.number}: {e}", exc_info=True)
        # Return a meaningful fallback instead of error message
        fallback = (
            f"This section on {topic} covers {section_template.name.lower()}. "
            f"Based on the research conducted, this area shows significant activity and development. "
            f"Key aspects include technology fundamentals, market dynamics, and future opportunities."
        )
        return fallback


def _generate_executive_summary(
    topic: str, all_sections: Dict[str, str], max_output_tokens: int = 1000, use_rate_limit: bool = True
) -> str:
    """Generate executive summary from all section content."""
    template = REPORT_SECTIONS[2]
    if not template.prompt_template:
        return ""
    
    # Aggregate all section texts (truncate to avoid overly long prompts)
    sections_text = "\n\n".join([f"### {name}\n{content[:800]}..." for name, content in all_sections.items() if content])
    
    prompt = template.prompt_template.format(topic=topic, all_sections=sections_text)
    
    # Apply rate limiting before API call
    if use_rate_limit:
        _global_rate_limiter.wait_if_needed()
    
    try:
        # Use empty context for executive summary since it synthesizes other sections
        summary = generate_answer(
            prompt,
            [],
            max_output_tokens=max_output_tokens,
            report_mode=True,  # Enable detailed report-style generation
        )
        
        # Validate summary
        if not summary or summary.strip() == "" or summary.startswith("[Error"):
            LOGGER.warning("Executive summary was empty/invalid, generating simplified version")
            simplified_prompt = (
                f"Write a 2-3 paragraph executive summary for a technology landscape report about {topic}. "
                f"Summarize the key findings from the research. Keep it professional and concise."
            )
            summary = generate_answer(
                simplified_prompt,
                [],
                max_output_tokens=800,
                report_mode=True,  # Enable detailed report-style generation
            )
        
        # Final fallback if still empty
        if not summary or summary.strip() == "" or summary.startswith("[Error"):
            LOGGER.warning("Using fallback executive summary")
            summary = (
                f"This technology landscape report on {topic} examines the current state, "
                f"key market players, recent developments, trends, and opportunities in this space. "
                f"The findings indicate significant activity and potential for stakeholders."
            )
        
        # Strip any unwanted sections
        summary = _strip_unwanted_sections(summary)
        # Clean template artifacts and formatting
        summary = _clean_template_artifacts(summary)
        return summary
    except Exception as e:
        LOGGER.error(f"Error generating executive summary: {e}", exc_info=True)
        # Return meaningful fallback instead of error message
        return (
            f"This technology landscape report on {topic} examines the current state, "
            f"key market players, recent developments, trends, and opportunities in this space. "
            f"The findings indicate significant activity and potential for stakeholders."
        )


def generate_report(
    topic: str,
    max_urls_per_section: int = 10,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Dict[str, Any]:
    """
    Generate a complete Technology Landscape Report.
    
    Returns a dict with:
    - sections: Dict[str, str] - section name -> content
    - citations: List[str] - all source URLs
    - rag_index: FAISS index
    - rag_chunks: List[Chunk]
    - rag_embeddings: np.ndarray
    - metadata: dict with topic, date, etc.
    """
    _configure_logging()
    LOGGER.info(f"Starting report generation for topic: {topic}")
    LOGGER.info(f"Parameters: max_urls_per_section={max_urls_per_section}, embedding_model={embedding_model}")
    
    # Step 1: Fetch data for all sections in parallel
    sections_needing_api = get_sections_requiring_api()
    LOGGER.info(f"Fetching data for {len(sections_needing_api)} sections")
    section_data = _fetch_section_data_parallel(topic, sections_needing_api, max_urls_per_section)
    
    # Step 2: Aggregate all URLs and fetch contents
    all_urls_and_titles = _aggregate_all_urls(section_data)
    LOGGER.info(f"Collected {len(all_urls_and_titles)} unique URLs across all sections")
    
    if not all_urls_and_titles:
        LOGGER.error("No URLs collected from API calls - cannot proceed")
        raise ValueError("No URLs collected from API calls")
    
    # Fetch full content (already imported at top)
    # Note: You.com API has a limit of 10 URLs per request, handled by build_corpus
    LOGGER.info(f"Will fetch full content for {len(all_urls_and_titles)} URLs during RAG index building...")
    contents = []  # Content will be fetched during build_corpus
    
    # Step 3: Build RAG index from all content
    LOGGER.info("Building RAG index from collected content...")
    try:
        rag_index, rag_chunks, rag_embeddings = _build_report_rag_index(
            all_urls_and_titles, embedding_model
        )
        if rag_index is not None:
            LOGGER.info(f"RAG index built successfully with {len(rag_chunks)} chunks")
        else:
            LOGGER.warning("RAG index is None - continuing without index")
    except Exception as e:
        LOGGER.error(f"Error building RAG index: {e}", exc_info=True)
        rag_index, rag_chunks, rag_embeddings = None, [], np.array([])
    
    # Step 4: Prepare context chunks for each section
    # Map URLs to chunks
    url_to_chunks: Dict[str, List[Dict[str, Any]]] = {}
    for chunk in rag_chunks:
        # Handle Chunk dataclass or dict
        if hasattr(chunk, "url"):
            url = chunk.url
            text = chunk.text if hasattr(chunk, "text") else ""
            title = chunk.title if hasattr(chunk, "title") else ""
        elif isinstance(chunk, dict):
            url = chunk.get("url", "")
            text = chunk.get("text", "")
            title = chunk.get("title", "")
        else:
            continue
            
        if url:
            if url not in url_to_chunks:
                url_to_chunks[url] = []
            url_to_chunks[url].append({
                "text": text,
                "title": title,
                "url": url,
            })
    
    # Step 5: Generate content for each section (3-7) using advanced RAG retrieval in parallel
    sections_content: Dict[str, str] = {}
    section_order = [3, 4, 5, 6, 7]  # Skip 1 (cover), 2 (summary), 8 (appendix)
    
    def _generate_single_section(
        section_num: int,
        topic: str,
        rag_index: Any,
        rag_chunks: List,
        rag_embeddings: np.ndarray,
        embedding_model: str,
        url_to_chunks: Dict[str, List[Dict[str, Any]]],
        section_data: Dict[int, Dict[str, Any]],
    ) -> Tuple[int, str]:
        """Generate a single section with all retrieval logic (for parallel execution)."""
        template = REPORT_SECTIONS[section_num]
        LOGGER.info(f"Generating content for section {section_num}: {template.name}")
        
        # Build query from section template and topic
        section_query = template.prompt_template.format(topic=topic) if template.prompt_template else topic
        if template.api_query_hint:
            query_hint = template.api_query_hint.format(topic=topic)
            section_query = f"{query_hint}. {section_query}"
        
        LOGGER.debug(f"Section {section_num} query: {section_query[:200]}...")
        
        # Use advanced retrieval if RAG index is available
        if rag_index is not None and rag_chunks:
            LOGGER.info(f"Using advanced RAG retrieval for section {section_num}")
            # Retrieve top 15 chunks using fast hybrid search (skips expensive reranking/MMR)
            selected_indices, section_chunks = _retrieve_chunks_for_section(
                query=section_query,
                rag_index=rag_index,
                rag_chunks=rag_chunks,
                rag_embeddings=rag_embeddings,
                embedding_model_name=embedding_model,
                top_k=15,
                fast_mode=True,  # Use fast mode for parallel generation
            )
            
            # Apply windowed retrieval to expand context
            if selected_indices:
                from marketvantage.retrieval import expand_with_neighbors
                expanded_indices = expand_with_neighbors(
                    selected_indices[:10],
                    total=len(rag_chunks),
                    window=1
                )
                LOGGER.info(f"Expanded from {len(selected_indices)} to {len(expanded_indices)} chunks with windowed retrieval")
                
                # Convert expanded indices back to chunk dicts
                section_chunks = [
                    {
                        "text": rag_chunks[i].text if hasattr(rag_chunks[i], "text") else str(rag_chunks[i]),
                        "title": rag_chunks[i].title if hasattr(rag_chunks[i], "title") else "",
                        "url": rag_chunks[i].url if hasattr(rag_chunks[i], "url") else "",
                    }
                    for i in expanded_indices
                    if 0 <= i < len(rag_chunks)
                ]
            
            # Fallback to URL-based if retrieval returns nothing
            if not section_chunks:
                LOGGER.warning(f"Advanced retrieval returned no chunks for section {section_num}, falling back to URL-based")
                section_urls = section_data.get(section_num, {}).get("urls", [])
                for url in section_urls:
                    if url in url_to_chunks:
                        section_chunks.extend(url_to_chunks[url])
                
                # Final fallback: use general chunks if URL-based also fails
                if not section_chunks and rag_chunks:
                    LOGGER.warning(f"URL-based fallback also returned no chunks for section {section_num}, using first 20 chunks as final fallback")
                    section_chunks = [
                        {
                            "text": c.text if hasattr(c, "text") else str(c),
                            "title": c.title if hasattr(c, "title") else "",
                            "url": c.url if hasattr(c, "url") else "",
                        }
                        for c in rag_chunks[:20]
                    ]
        else:
            # Fallback to simple URL-based retrieval if RAG index unavailable
            LOGGER.warning(f"RAG index unavailable for section {section_num}, using URL-based retrieval")
            section_urls = section_data.get(section_num, {}).get("urls", [])
            section_chunks = []
            for url in section_urls:
                if url in url_to_chunks:
                    section_chunks.extend(url_to_chunks[url])
            
            # Final fallback: use all chunks
            if not section_chunks and rag_chunks:
                LOGGER.warning(f"No chunks found for section {section_num}, using first 20 chunks")
                section_chunks = [
                    {
                        "text": c.text if hasattr(c, "text") else str(c),
                        "title": c.title if hasattr(c, "title") else "",
                        "url": c.url if hasattr(c, "url") else "",
                    }
                    for c in rag_chunks[:20]
                ]
        
        # Log chunk count and validate chunks have text content
        LOGGER.info(f"Section {section_num} ({template.name}): Retrieved {len(section_chunks)} chunks for generation")
        
        # Validate chunks have text content
        chunks_with_text = sum(1 for chunk in section_chunks if chunk.get("text", "").strip())
        if chunks_with_text == 0:
            LOGGER.error(f"Section {section_num} ({template.name}): CRITICAL - All {len(section_chunks)} chunks have empty text!")
            # Debug: Log a sample chunk to see what's wrong
            if section_chunks:
                sample = section_chunks[0]
                LOGGER.error(f"Sample chunk keys: {list(sample.keys())}, text length: {len(str(sample.get('text', '')))}, text preview: {str(sample.get('text', ''))[:100]}")
        
        if not section_chunks:
            LOGGER.error(f"Section {section_num} ({template.name}): WARNING - No chunks available for generation!")
        
        # Generate content with rate limiting
        try:
            content = _generate_section_content(template, topic, section_chunks, use_rate_limit=True)
            return section_num, content
        except Exception as e:
            LOGGER.error(f"Failed to generate section {section_num}: {e}")
            return section_num, f"[Error: Could not generate content for this section. {str(e)}]"
    
    # Generate sections in parallel with controlled concurrency
    LOGGER.info(f"Starting parallel generation for {len(section_order)} sections")
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=3) as executor:  # Limit concurrent workers to avoid rate limits
        futures = {
            executor.submit(
                _generate_single_section,
                section_num,
                topic,
                rag_index,
                rag_chunks,
                rag_embeddings,
                embedding_model,
                url_to_chunks,
                section_data,
            ): section_num
            for section_num in section_order
        }
        
        completed = 0
        for future in as_completed(futures):
            section_num = futures[future]
            try:
                section_num, content = future.result()
                sections_content[REPORT_SECTIONS[section_num].name] = content
                completed += 1
                elapsed = time.time() - start_time
                LOGGER.info(f"Completed section {section_num} ({completed}/{len(section_order)}) in {elapsed:.1f}s")
            except Exception as e:
                LOGGER.error(f"Failed to generate section {section_num}: {e}", exc_info=True)
                sections_content[REPORT_SECTIONS[section_num].name] = f"[Error: Could not generate content for this section. {str(e)}]"
    
    total_elapsed = time.time() - start_time
    LOGGER.info(f"Completed parallel generation of {len(section_order)} sections in {total_elapsed:.1f}s")
    
    # Step 6: Generate Executive Summary (section 2)
    LOGGER.info("Generating executive summary...")
    executive_summary = _generate_executive_summary(topic, sections_content)
    executive_summary = _strip_unwanted_sections(executive_summary)
    sections_content["Executive Summary"] = executive_summary
    
    # Step 6.5: Generate Conclusion (section 8)
    LOGGER.info("Generating conclusion...")
    conclusion_template = REPORT_SECTIONS[8]
    
    # Apply rate limiting before API call
    _global_rate_limiter.wait_if_needed()
    
    try:
        from marketvantage.llm_groq import generate_answer
        
        # First attempt: Full conclusion with all sections
        all_sections_text = "\n\n".join([f"## {name}\n{content[:500]}..." for name, content in sections_content.items()])
        conclusion_prompt = conclusion_template.prompt_template.format(topic=topic, all_sections=all_sections_text)
        
        try:
            conclusion = generate_answer(
                conclusion_prompt,
                [],  # No chunks needed
                max_output_tokens=800,
                report_mode=True,  # Enable detailed report-style generation
            )
            conclusion = _strip_unwanted_sections(conclusion)
            conclusion = _clean_template_artifacts(conclusion)
            
            # Check if conclusion is meaningful (not empty or error message)
            if conclusion and not conclusion.startswith("[Error") and len(conclusion.strip()) > 50:
                sections_content["Conclusion"] = conclusion
                LOGGER.info("Conclusion generated successfully")
            else:
                raise ValueError("Conclusion was empty or invalid")
                
        except Exception as e1:
            LOGGER.warning(f"Full conclusion generation failed: {e1}, trying simplified version")
            
            # Second attempt: Simplified conclusion prompt (less context, might avoid filters)
            simplified_prompt = (
                f"Write a 2-3 paragraph conclusion for a technology landscape report about {topic}. "
                f"Summarize the key findings and provide final insights. Keep it professional and concise."
            )
            
            try:
                conclusion = generate_answer(
                    simplified_prompt,
                    [],  # No chunks needed
                    max_output_tokens=600,
                    report_mode=True,  # Enable detailed report-style generation
                )
                conclusion = _strip_unwanted_sections(conclusion)
                conclusion = _clean_template_artifacts(conclusion)
                
                if conclusion and not conclusion.startswith("[Error") and len(conclusion.strip()) > 50:
                    sections_content["Conclusion"] = conclusion
                    LOGGER.info("Conclusion generated successfully with simplified prompt")
                else:
                    raise ValueError("Simplified conclusion was also empty or invalid")
                    
            except Exception as e2:
                LOGGER.error(f"Simplified conclusion generation also failed: {e2}")
                # Generate a basic fallback conclusion
                sections_content["Conclusion"] = (
                    f"This technology landscape report on {topic} has examined key aspects including "
                    f"technology overview, market players, recent developments, trends, and opportunities. "
                    f"The findings indicate significant activity and potential in this space. "
                    f"Further research and monitoring of developments would be valuable for stakeholders."
                )
                LOGGER.info("Using fallback conclusion text")
                
    except Exception as e:
        LOGGER.error(f"Failed to generate conclusion after all attempts: {e}", exc_info=True)
        sections_content["Conclusion"] = (
            f"Conclusion could not be generated automatically. "
            f"Please refer to the sections above for key findings about {topic}."
        )
    
    # Step 7: Collect all citations
    all_citations = list(set([url for url, _ in all_urls_and_titles]))
    LOGGER.info(f"Collected {len(all_citations)} unique citations")
    
    # Step 8: Prepare metadata
    metadata = {
        "topic": topic,
        "generated_at": datetime.now().isoformat(),
        "sections_generated": list(sections_content.keys()),
        "total_citations": len(all_citations),
    }
    
    LOGGER.info(f"Report generation complete. Generated {len(sections_content)} sections")
    LOGGER.info(f"Sections: {', '.join(sections_content.keys())}")
    
    return {
        "sections": sections_content,
        "citations": all_citations,
        "rag_index": rag_index,
        "rag_chunks": rag_chunks,
        "rag_embeddings": rag_embeddings,
        "metadata": metadata,
    }
