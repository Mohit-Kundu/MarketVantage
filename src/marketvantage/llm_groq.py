import logging
import os
from pathlib import Path
from typing import List, Optional

from groq import Groq

try:
    from dotenv import load_dotenv, find_dotenv, dotenv_values
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore
    find_dotenv = None  # type: ignore
    dotenv_values = None  # type: ignore

LOGGER = logging.getLogger("marketvantage.llm_groq")


def _sanitize(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    v = value.strip()
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        v = v[1:-1]
    return v.strip()


def _load_env_if_possible() -> None:
    # Try current working dir .env discovery
    if load_dotenv is not None and find_dotenv is not None:
        try:
            dotenv_path = find_dotenv(usecwd=True)
            if dotenv_path:
                load_dotenv(dotenv_path=dotenv_path, override=False)
                LOGGER.debug("Loaded .env via find_dotenv", extra={"dotenv_path": dotenv_path})
        except Exception as e:
            LOGGER.debug("find_dotenv load failed", extra={"error": str(e)})
    # Try repo-root relative to package file
    try:
        pkg_root = Path(__file__).resolve().parents[2]
        repo_env = pkg_root / ".env"
        if load_dotenv is not None and repo_env.exists():
            load_dotenv(dotenv_path=str(repo_env), override=False)
            LOGGER.debug("Loaded .env via package-root path", extra={"dotenv_path": str(repo_env)})
    except Exception as e:
        LOGGER.debug("package-root .env load failed", extra={"error": str(e)})

    # If still missing, read values directly and inject once
    if dotenv_values is not None and not os.getenv("GROQ_API_KEY"):
        for candidate in [
            (find_dotenv(usecwd=True) if find_dotenv is not None else ""),
            str(Path(__file__).resolve().parents[2] / ".env"),
        ]:
            try:
                if candidate and os.path.exists(candidate):
                    vals = dotenv_values(dotenv_path=candidate)
                    raw = vals.get("GROQ_API_KEY")
                    if raw:
                        os.environ["GROQ_API_KEY"] = _sanitize(raw) or raw
                        LOGGER.info(
                            "Injected GROQ_API_KEY from dotenv_values",
                            extra={"source": candidate, "length": len(os.environ["GROQ_API_KEY"])},
                        )
                        break
            except Exception as e:
                LOGGER.debug("dotenv_values load failed", extra={"error": str(e)})


def _get_groq_api_key() -> str:
    _load_env_if_possible()
    api_key = _sanitize(os.getenv("GROQ_API_KEY"))
    if not api_key:
        LOGGER.error("GROQ_API_KEY missing after env load attempts")
        raise RuntimeError(
            "Missing Groq API key. Set GROQ_API_KEY in environment or .env to enable answer generation."
        )
    LOGGER.info("GROQ_API_KEY detected", extra={"length": len(api_key)})
    return api_key


def generate_answer(
    question: str,
    context_chunks: List,  # list of str or list of dicts with text/title/url
    *,
    model: str = "llama-3.1-8b-instant",
    max_output_tokens: int = 1024,
    report_mode: bool = False,  # If True, use detailed report-style prompts instead of brief Q&A
) -> str:
    if not question or not question.strip():
        raise ValueError("question must be non-empty")

    # Build context with citations if dicts are provided
    blocks: List[str] = []
    for i, ctx in enumerate(context_chunks[:10]):
        if isinstance(ctx, dict):
            title = str(ctx.get("title", "")).strip()
            url = str(ctx.get("url", "")).strip()
            text = str(ctx.get("text", ""))
            header = f"[{i+1}] {title} ({url})".strip()
            blocks.append(f"{header}\n\n{text}")
        else:
            blocks.append(str(ctx))
    joined = "\n\n---\n\n".join(blocks)

    # Use different prompts for report generation vs Q&A
    if report_mode:
        system_prompt = (
            "You are a professional technical writer creating comprehensive content for technology landscape reports. "
            "Write detailed, well-structured sections that thoroughly analyze the topic. "
            "Use multiple paragraphs as needed to provide comprehensive coverage. "
            "Base your response on the provided sources and cite them using bracketed numbers like [1], [2]. "
            "Write in a professional, analytical style suitable for business and technical audiences."
        )
        user_prompt = (
            (f"Sources (markdown):\n\n{joined}\n\n" if joined else "")
            + f"{question}\n\n"
            + "Write a comprehensive, detailed section based on the sources above. Use citations [1], [2], etc. where appropriate. "
            + "Provide thorough analysis with multiple paragraphs covering all relevant aspects."
        )
    else:
        system_prompt = (
            "You are a helpful assistant that provides direct, concise answers. "
            "Be brief and to the point. Answer in 2-3 sentences maximum. "
            "Prefer the provided sources and cite using bracketed numbers like [1], [2]. "
            "If the answer is not supported by the sources, say so briefly."
        )
        user_prompt = (
            (f"Sources (markdown):\n\n{joined}\n\n" if joined else "")
            + f"Question: {question}\n\n"
            + "Answer the question directly and concisely in 2-3 sentences. Include citations [1], [2] if relevant."
        )

    api_key = _get_groq_api_key()
    # Groq client doesn't accept proxies parameter, so create http_client explicitly
    # to avoid httpx trying to use environment proxy variables
    try:
        import httpx
        # Create httpx client with trust_env=False to prevent using environment proxy settings
        http_client = httpx.Client(
            timeout=httpx.Timeout(30.0, connect=10.0),
            trust_env=False,  # Don't use HTTP_PROXY/HTTPS_PROXY environment variables
        )
        client = Groq(api_key=api_key, http_client=http_client)
    except Exception as e:
        # Fallback to simple initialization if custom http_client fails
        LOGGER.debug(f"Failed to create custom http_client, using default: {e}")
        client = Groq(api_key=api_key)

    def _call(messages, temp: float, max_tok: int):
        return client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temp,
            max_tokens=max_tok,
            tools=[],
            tool_choice="none",
        )

    LOGGER.info("Calling Groq model", extra={"model": model, "context_chunks": len(context_chunks)})
    def _has_citation(ans: str) -> bool:
        import re as _re
        return bool(_re.search(r"\[(?:\d+)\]", ans))

    # First attempt - adjust temperature based on mode
    completion = _call(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temp=0.3 if report_mode else 0.1,  # Slightly higher temp for report mode for more variety
        max_tok=max_output_tokens,
    )
    content = (completion.choices[0].message.content or "").strip()

    # Grader/validation loop: enforce at least one citation if sources exist (only for Q&A mode)
    if blocks and not _has_citation(content) and not report_mode:
        LOGGER.info("No citations found, retrying with stricter instruction")
        reinforce = (
            (f"Sources (markdown):\n\n{joined}\n\n")
            + f"Question: {question}\n\n"
            + "Answer directly in 2-3 sentences. Include citations [1], [2] from the sources above."
        )
        completion = _call(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": reinforce},
            ],
            temp=0.1,  # Keep low temperature for concise answers
            max_tok=max_output_tokens,
        )
        content = (completion.choices[0].message.content or "").strip()

    if not content:
        LOGGER.warning("Empty Groq content, retrying with simpler prompt")
        completion = _call(
            messages=[{"role": "user", "content": question}],
            temp=0.5,
            max_tok=max(256, max_output_tokens),
        )
        content = (completion.choices[0].message.content or "").strip()
        LOGGER.debug("Retry content length", extra={"chars": len(content)})

    return content
