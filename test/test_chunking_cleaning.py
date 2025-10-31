from marketvantage.rag_app import _clean_markdown, _split_markdown_blocks, chunk_markdown, Chunk
from marketvantage.retrieval import reciprocal_rank_fusion


def test_clean_markdown_removes_boilerplate():
    raw = """
    Sign in | Subscribe | Newsletter

    # Title
    Useful content here.

    Accept cookies to continue.
    Footer — All rights reserved.
    """.strip()
    cleaned = _clean_markdown(raw)
    assert "Accept cookies".lower() not in cleaned.lower()
    assert "all rights reserved" not in cleaned.lower()
    assert "Useful content" in cleaned


def test_split_preserves_code_fences():
    md = """
    # H1
    Intro

    ```python
    print('code')
    ```

    ## H2
    More text
    """.strip()
    blocks = _split_markdown_blocks(md)
    # Expect at least 3 blocks: H1+Intro, code fence, H2+text
    assert any("```" in b for b in blocks)
    assert any(b.startswith("# ") for b in blocks)


def test_chunk_markdown_merges_short_chunks():
    md = "# A\n" + ("word " * 40) + "\n## B\n" + ("word " * 40)
    chunks = chunk_markdown(md, url="u", title="t", token_target=80)
    # Should produce at least one chunk and not produce many tiny chunks
    assert len(chunks) <= 2
    assert all(isinstance(c, Chunk) for c in chunks)


def test_rrf_basic():
    a = [(1, 10.0), (2, 9.0), (3, 8.0)]
    b = [(3, 9.0), (2, 8.5), (4, 7.0)]
    fused = reciprocal_rank_fusion([a, b])
    # 2 and 3 should be near the top due to presence in both lists
    assert fused[0] in (2, 3)
    assert set(fused[:3]).issuperset({2, 3})


