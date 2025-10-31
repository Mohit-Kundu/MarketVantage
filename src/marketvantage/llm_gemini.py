import logging
import os
from pathlib import Path
from typing import List, Optional

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from dotenv import load_dotenv, find_dotenv, dotenv_values
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore
    find_dotenv = None  # type: ignore
    dotenv_values = None  # type: ignore

LOGGER = logging.getLogger("marketvantage.llm_gemini")


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
    if dotenv_values is not None and not os.getenv("GEMINI_API_KEY"):
        for candidate in [
            (find_dotenv(usecwd=True) if find_dotenv is not None else ""),
            str(Path(__file__).resolve().parents[2] / ".env"),
        ]:
            try:
                if candidate and os.path.exists(candidate):
                    vals = dotenv_values(dotenv_path=candidate)
                    raw = vals.get("GEMINI_API_KEY")
                    if raw:
                        os.environ["GEMINI_API_KEY"] = _sanitize(raw) or raw
                        LOGGER.info(
                            "Injected GEMINI_API_KEY from dotenv_values",
                            extra={"source": candidate, "length": len(os.environ["GEMINI_API_KEY"])},
                        )
                        break
            except Exception as e:
                LOGGER.debug("dotenv_values load failed", extra={"error": str(e)})


def _get_gemini_api_key() -> str:
    _load_env_if_possible()
    api_key = _sanitize(os.getenv("GEMINI_API_KEY"))
    if not api_key:
        LOGGER.error("GEMINI_API_KEY missing after env load attempts")
        raise RuntimeError(
            "Missing Gemini API key. Set GEMINI_API_KEY in environment or .env to enable answer generation."
        )
    LOGGER.info("GEMINI_API_KEY detected", extra={"length": len(api_key)})
    return api_key


def generate_answer(
    question: str,
    context_chunks: List,  # list of str or list of dicts with text/title/url
    *,
    model: str = "gemini-2.5-flash",
    max_output_tokens: int = 1024,
) -> str:
    if genai is None:
        raise RuntimeError(
            "google-generativeai package not installed. Install it with: pip install google-generativeai"
        )

    if not question or not question.strip():
        raise ValueError("question must be non-empty")

    # Build context with citations if dicts are provided
    blocks: List[str] = []
    valid_chunk_count = 0
    for i, ctx in enumerate(context_chunks[:10]):
        if isinstance(ctx, dict):
            title = str(ctx.get("title", "")).strip()
            url = str(ctx.get("url", "")).strip()
            text = str(ctx.get("text", "")).strip()  # Strip text to check for empty
            
            # Only include chunks with meaningful text content (at least 10 chars)
            if text and len(text) >= 10:
                header = f"[{valid_chunk_count + 1}] {title} ({url})".strip()
                blocks.append(f"{header}\n\n{text}")
                valid_chunk_count += 1
            else:
                # Log skipped chunks with empty text for debugging
                LOGGER.debug(f"Skipping chunk {i+1} - empty or too short text (length: {len(text) if text else 0})")
        else:
            text = str(ctx).strip()
            if text and len(text) >= 10:
                blocks.append(text)
                valid_chunk_count += 1
    
    joined = "\n\n---\n\n".join(blocks)
    
    # Log if no valid chunks were found
    if not blocks:
        LOGGER.warning(f"No valid chunks with text content found in {len(context_chunks)} provided chunks")
        # If we have chunks but they're all empty, log details for debugging
        if context_chunks:
            sample_chunk = context_chunks[0]
            if isinstance(sample_chunk, dict):
                LOGGER.warning(f"Sample chunk: keys={list(sample_chunk.keys())}, text_type={type(sample_chunk.get('text'))}, text_len={len(str(sample_chunk.get('text', '')))}")
                LOGGER.warning(f"Sample chunk text preview: {str(sample_chunk.get('text', ''))[:200]}")

    system_prompt = (
        "You are a helpful assistant that provides direct, concise answers. "
        "Be brief and to the point. Answer in 2-3 sentences maximum. "
        "Prefer the provided sources and cite using bracketed numbers like [1], [2]. "
        "If the answer is not supported by the sources, say so briefly."
    )
    
    # Build user prompt - ensure sources section only appears if we have valid content
    if joined and len(joined.strip()) > 20:  # Ensure joined has meaningful content (not just headers)
        user_prompt = (
            f"Sources (markdown):\n\n{joined}\n\n"
            + f"Question: {question}\n\n"
            + "Answer the question directly and concisely in 2-3 sentences. Include citations [1], [2] if relevant."
        )
    else:
        # No valid sources available
        LOGGER.warning(f"No valid sources available for question: {question[:100]}...")
        user_prompt = (
            f"Question: {question}\n\n"
            + "Answer the question directly and concisely based on general knowledge. "
            + "Since no specific sources were provided, provide a general answer."
        )

    api_key = _get_gemini_api_key()
    genai.configure(api_key=api_key)

    def _call(prompt: str, temp: float, max_tok: int):
        model_instance = genai.GenerativeModel(model)
        # Combine system prompt and user prompt for Gemini
        full_prompt = f"{system_prompt}\n\n{prompt}"
        response = model_instance.generate_content(
            full_prompt,
            generation_config={
                "temperature": temp,
                "max_output_tokens": max_tok,
            },
        )
        # Check if response has valid content
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if candidate.finish_reason and candidate.finish_reason != 1:  # 1 = STOP (success)
                # Map finish_reason codes to human-readable messages
                finish_reason_map = {
                    1: "STOP (success)",
                    2: "SAFETY (content filtered)",
                    3: "RECITATION (detected recitation)",
                    4: "OTHER",
                    5: "MAX_TOKENS",
                }
                reason_msg = finish_reason_map.get(candidate.finish_reason, f"UNKNOWN ({candidate.finish_reason})")
                LOGGER.warning(
                    f"Gemini finish_reason: {reason_msg}",
                    extra={"finish_reason": candidate.finish_reason, "reason_msg": reason_msg},
                )
        return response

    LOGGER.info("Calling Gemini model", extra={"model": model, "context_chunks": len(context_chunks)})
    def _has_citation(ans: str) -> bool:
        import re as _re
        return bool(_re.search(r"\[(?:\d+)\]", ans))

    # First attempt - lower temperature for more focused, concise answers
    try:
        completion = _call(user_prompt, temp=0.1, max_tok=max_output_tokens)
        # Safely extract text from response
        try:
            content = (completion.text or "").strip()
        except ValueError as ve:
            # Handle case where response.text fails (e.g., finish_reason != STOP)
            LOGGER.warning(f"Could not extract text from response: {ve}")
            
            # Check if content was filtered by safety filters
            was_safety_filtered = False
            if completion.candidates and len(completion.candidates) > 0:
                candidate = completion.candidates[0]
                if candidate.finish_reason == 2:  # SAFETY
                    was_safety_filtered = True
                    LOGGER.warning(
                        "Content filtered by safety filters on first attempt. "
                        "This may be due to sensitive keywords or topics in the prompt/sources. "
                        "Will retry with a more neutral, simplified prompt."
                    )
                
                if candidate.content and candidate.content.parts:
                    content = candidate.content.parts[0].text if candidate.content.parts else ""
                else:
                    content = ""
            else:
                content = ""
            
            # If safety filtered, skip to simplified retry immediately
            if was_safety_filtered:
                content = ""  # Force retry path
    except Exception as e:
        error_str = str(e)
        LOGGER.warning(f"Gemini API call failed: {e}", extra={"error": error_str})
        
        # Check if it's a quota/rate limit error
        if "429" in error_str or "quota" in error_str.lower() or "rate" in error_str.lower():
            # Extract retry delay if available
            import re
            retry_match = re.search(r"retry.*?(\d+(?:\.\d+)?)\s*s", error_str, re.IGNORECASE)
            if retry_match:
                retry_seconds = float(retry_match.group(1))
                LOGGER.warning(f"Quota exceeded, would need to wait {retry_seconds}s. Returning empty for caller fallback.")
            else:
                LOGGER.warning("Quota exceeded, returning empty for caller fallback.")
        
        content = ""

    # Grader/validation loop: enforce at least one citation if sources exist
    if blocks and not _has_citation(content) and content:
        LOGGER.info("No citations found, retrying with stricter instruction")
        reinforce = (
            (f"Sources (markdown):\n\n{joined}\n\n")
            + f"Question: {question}\n\n"
            + "Answer directly in 2-3 sentences. Include citations [1], [2] from the sources above."
        )
        try:
            completion = _call(reinforce, temp=0.1, max_tok=max_output_tokens)
            try:
                content = (completion.text or "").strip()
            except ValueError:
                if completion.candidates and len(completion.candidates) > 0:
                    candidate = completion.candidates[0]
                    if candidate.content and candidate.content.parts:
                        content = candidate.content.parts[0].text if candidate.content.parts else ""
                    else:
                        content = ""
                else:
                    content = ""
        except Exception as e:
            LOGGER.warning(f"Gemini retry failed: {e}", extra={"error": str(e)})

    if not content:
        LOGGER.warning("Empty Gemini content, retrying with simpler prompt")
        try:
            # Try with a neutral, simplified approach to avoid safety filters
            # Extract just the core question, remove potentially sensitive parts
            simplified_question = question.split("\n")[0] if "\n" in question else question
            # Remove sources section if present (might contain filtered content)
            if "Sources (markdown):" in simplified_question:
                simplified_question = simplified_question.split("Question:")[-1].strip()
            # Further simplify to avoid sensitive keywords
            simplified_question = simplified_question[:300]  # Limit length
            completion = _call(simplified_question, temp=0.5, max_tok=max(256, max_output_tokens))
            try:
                content = (completion.text or "").strip()
            except ValueError:
                if completion.candidates and len(completion.candidates) > 0:
                    candidate = completion.candidates[0]
                    if candidate.content and candidate.content.parts:
                        content = candidate.content.parts[0].text if candidate.content.parts else ""
                    else:
                        content = ""
                else:
                    content = ""
            
            # If still empty, check finish_reason
            if not content and completion.candidates and len(completion.candidates) > 0:
                candidate = completion.candidates[0]
                if candidate.finish_reason and candidate.finish_reason == 2:
                    LOGGER.warning(
                        "Content was filtered by safety filters (finish_reason: 2) on retry. "
                        "This indicates the topic or sources may contain content that triggers safety filters. "
                        "Trying with ultra-safe, neutral prompt."
                    )
                    # Try one more time with extremely neutral, safe prompt
                    try:
                        # Extract just the topic/keyword, avoid any sensitive phrasing
                        topic_words = question.split("?")[0].split(".")[0].split(":")[-1].strip()[:150]
                        safe_prompt = f"Provide a neutral, factual overview of {topic_words}. Focus on general information."
                        completion = _call(safe_prompt, temp=0.7, max_tok=512)
                        try:
                            content = (completion.text or "").strip()
                        except ValueError:
                            content = ""
                    except Exception:
                        content = ""
            
            LOGGER.debug("Retry content length", extra={"chars": len(content)})
        except Exception as e:
            LOGGER.error(f"Gemini fallback failed: {e}", extra={"error": str(e)})
            # Don't return error message, return empty string so caller can handle fallback
            content = ""

    # Final validation - if still empty, try one more time with very simple prompt
    if not content or len(content.strip()) < 10:
        LOGGER.warning("Final content is empty or too short after all retries, trying one last simplified attempt")
        try:
            # Ultra-simple prompt to avoid filters
            simple_q = question.split("?")[0].split(".")[0][:100] if "?" in question or "." in question else question[:100]
            ultra_simple = f"Write about {simple_q}"
            completion = _call(ultra_simple, temp=0.8, max_tok=400)
            try:
                content = (completion.text or "").strip()
            except ValueError:
                if completion.candidates and len(completion.candidates) > 0:
                    candidate = completion.candidates[0]
                    if candidate.content and candidate.content.parts:
                        content = candidate.content.parts[0].text if candidate.content.parts else ""
                    else:
                        content = ""
                else:
                    content = ""
        except Exception:
            content = ""
    
    # If still empty, return empty string so caller can provide fallback
    if not content or len(content.strip()) < 10:
        LOGGER.warning("Final content is empty or too short after all retries")
        return ""

    return content

