"""Streamlit app for Technology Landscape Report Generation."""
import logging
import pathlib
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
import numpy as np

# Ensure package imports work
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Load environment variables
try:
    from dotenv import load_dotenv
    _dotenv_path = PROJECT_ROOT / ".env"
    if _dotenv_path.exists():
        load_dotenv(dotenv_path=str(_dotenv_path), override=False)
except Exception:
    pass

from marketvantage.report_engine import generate_report  # noqa: E402
from marketvantage.pdf_generator import create_pdf_report  # noqa: E402
from marketvantage.you_search import configure_logging  # noqa: E402
from marketvantage.rag_app import (  # noqa: E402
    get_model,
    embed_chunks,
    search_top_k,
)
from marketvantage.retrieval import expand_with_neighbors  # noqa: E402
from marketvantage.llm_groq import generate_answer  # noqa: E402

# Configure logging
configure_logging(verbosity=1)
LOGGER = logging.getLogger("marketvantage.report_generator_app")
LOGGER.info("Report generator app module loaded")

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "output" / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def slugify(value: str) -> str:
    """Create URL-safe slug."""
    import re
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return value or "topic"


def clean_markdown_for_ui(text: str) -> str:
    """
    Clean markdown formatting for consistent UI display.
    Removes single-asterisk italics and ensures proper spacing.
    """
    import re
    
    # First, protect double asterisks (bold) temporarily
    text = text.replace('**', '<<<BOLD>>>')
    
    # Remove single asterisk italics: *text* -> text
    # This pattern matches *word* but ensures we don't match **bold**
    text = re.sub(r'\*([^*]+?)\*', r'\1', text)
    
    # Restore double asterisks (bold)
    text = text.replace('<<<BOLD>>>', '**')
    
    # Remove underscore italics: _text_ -> text
    text = re.sub(r'_([^_]+?)_', r'\1', text)
    
    # Fix spacing issues: ensure space after numbers before words
    # This handles cases like "13.9*billion*by2027" -> "13.9 billion by 2027"
    text = re.sub(r'(\d+(?:\.\d+)?)([a-zA-Z])', r'\1 \2', text)
    
    # Fix spacing: ensure space before numbers after words
    text = re.sub(r'([a-zA-Z])(\d+(?:\.\d+)?)', r'\1 \2', text)
    
    # Clean up multiple spaces
    text = re.sub(r' {2,}', ' ', text)
    
    return text.strip()


def render_qa_sidebar(report_data: Dict[str, Any]) -> None:
    """Render Q&A sidebar using built RAG index."""
    with st.sidebar:
        st.markdown("## Ask Questions")
        
        # Check if RAG index is available
        rag_index = report_data.get("rag_index")
        rag_chunks = report_data.get("rag_chunks", [])
        
        # Debug logging
        LOGGER.info(f"Sidebar: rag_index is None? {rag_index is None}")
        LOGGER.info(f"Sidebar: rag_chunks length: {len(rag_chunks) if rag_chunks else 0}")
        LOGGER.info(f"Sidebar: report_data keys: {list(report_data.keys())}")
        
        if rag_index is None or not rag_chunks:
            st.info("Generate a report to ask questions about it.")
            LOGGER.warning(f"Q&A unavailable - rag_index: {rag_index is None}, rag_chunks: {len(rag_chunks) if rag_chunks else 0}")
            return
        
        question = st.text_input(
            "What would you like to know?",
            placeholder="Enter your question...",
            key="qa_question",
            label_visibility="collapsed",
        )
        
        # Top-K passages fixed at 4
        top_k = 4
        
        if st.button("Search", key="qa_search", type="primary", use_container_width=True):
            if not question or not question.strip():
                st.warning("Please enter a question.")
                return
            
            LOGGER.info(f"Q&A search initiated: question='{question}', top_k={top_k}")
            # Search using existing RAG functions
            # Temporarily set session state for search_top_k
            st.session_state["faiss_index"] = rag_index
            st.session_state["faiss_chunks"] = rag_chunks
            st.session_state["faiss_embeddings"] = report_data.get("rag_embeddings")
            # Build BM25 index if available
            from marketvantage.retrieval import build_bm25  # noqa: E402
            chunk_texts = [
                c.text if hasattr(c, "text") else str(c) for c in rag_chunks
            ]
            st.session_state["bm25_index"] = build_bm25(chunk_texts)
            st.session_state["embedding_model"] = "sentence-transformers/all-MiniLM-L6-v2"
            
            with st.spinner("Searching..."):
                try:
                    ids = search_top_k(question, top_k)
                    LOGGER.info(f"Retrieved {len(ids)} chunk IDs")
                    # Windowed retrieval
                    win_ids = expand_with_neighbors(ids, total=len(rag_chunks), window=1)
                    LOGGER.info(f"Expanded to {len(win_ids)} chunks with windowed retrieval")
                    selected = [
                        {
                            "text": rag_chunks[i].text if hasattr(rag_chunks[i], "text") else str(rag_chunks[i]),
                            "title": rag_chunks[i].title if hasattr(rag_chunks[i], "title") else "",
                            "url": rag_chunks[i].url if hasattr(rag_chunks[i], "url") else "",
                        }
                        for i in win_ids
                        if 0 <= i < len(rag_chunks)
                    ]
                    
                    if selected:
                        LOGGER.info(f"Generating answer from {len(selected)} selected chunks")
                        with st.spinner("Generating answer..."):
                            answer = generate_answer(
                                question, 
                                selected,
                                max_output_tokens=300  # Limit to 300 tokens for concise answers
                            )
                            st.session_state["qa_answer"] = answer
                            st.session_state["qa_selected_chunks"] = selected
                            LOGGER.info("Answer generated successfully")
                    else:
                        LOGGER.warning("No relevant passages found for Q&A query")
                        st.warning("No relevant passages found.")
                except Exception as e:
                    LOGGER.error(f"Error during Q&A search: {e}", exc_info=True)
                    st.error(f"Error during search: {e}")
        
        # Display answer if available
        if "qa_answer" in st.session_state and st.session_state.get("qa_answer"):
            st.markdown("---")
            st.markdown("### Answer")
            qa_answer_cleaned = clean_markdown_for_ui(st.session_state["qa_answer"])
            st.write(qa_answer_cleaned)
            
            # Show sources
            if "qa_selected_chunks" in st.session_state and st.session_state.get("qa_selected_chunks"):
                chunks = st.session_state["qa_selected_chunks"]
                if chunks:
                    st.markdown("### Sources")
                    for i, chunk in enumerate(chunks[:5], 1):
                        url = chunk.get("url", "")
                        title = chunk.get("title", "")
                        if url or title:
                            st.markdown(f"{i}. **{title}** - {url}")




def render():
    """Main render function."""
    st.set_page_config(
        page_title="MarketVantage",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Custom CSS to change buttons, hover effects, and highlights to lavender
    st.markdown("""
    <style>
    /* Lavender color constants */
    :root {
        --lavender: #9370DB;
        --lavender-hover: #8360CB;
        --lavender-light: #B19CD9;
        --lavender-dark: #6A4C93;
    }
    
    /* Target primary buttons - change to lavender */
    button[kind="primary"],
    button[data-baseweb="button"][kind="primary"],
    [data-baseweb="button"][kind="primary"],
    .stButton > button[kind="primary"],
    .stButton > button[data-baseweb="button"][kind="primary"] {
        background-color: var(--lavender) !important;
        border-color: var(--lavender) !important;
        color: white !important;
    }
    
    /* Button hover effects - darker lavender */
    button[kind="primary"]:hover,
    button[data-baseweb="button"][kind="primary"]:hover,
    [data-baseweb="button"][kind="primary"]:hover,
    .stButton > button[kind="primary"]:hover {
        background-color: var(--lavender-hover) !important;
        border-color: var(--lavender-hover) !important;
        color: white !important;
    }
    
    /* Button active/pressed state */
    button[kind="primary"]:active,
    button[data-baseweb="button"][kind="primary"]:active,
    [data-baseweb="button"][kind="primary"]:active {
        background-color: var(--lavender-dark) !important;
        border-color: var(--lavender-dark) !important;
    }
    
    /* Button focus state with lavender glow */
    button[kind="primary"]:focus,
    button[kind="primary"]:focus-visible,
    button[data-baseweb="button"][kind="primary"]:focus,
    .stButton > button[kind="primary"]:focus {
        box-shadow: 0 0 0 3px rgba(147, 112, 219, 0.3) !important;
        outline: none !important;
    }
    
    /* Regular buttons hover effect */
    button:not([kind="primary"]):hover,
    [data-baseweb="button"]:not([kind="primary"]):hover {
        border-color: var(--lavender) !important;
        color: var(--lavender) !important;
    }
    
    /* Download button styling */
    .stDownloadButton > button,
    button[kind="secondary"]:has-text("Download") {
        background-color: var(--lavender) !important;
        border-color: var(--lavender) !important;
        color: white !important;
    }
    .stDownloadButton > button:hover {
        background-color: var(--lavender-hover) !important;
        border-color: var(--lavender-hover) !important;
    }
    
    /* Target selected/highlighted states */
    [aria-selected="true"],
    [data-state="selected"],
    .selected,
    [role="option"][aria-selected="true"] {
        background-color: var(--lavender-light) !important;
        color: white !important;
    }
    
    /* Hover effects on selectable items */
    [role="option"]:hover,
    [role="menuitem"]:hover,
    [data-baseweb="menu"] [role="option"]:hover {
        background-color: var(--lavender-light) !important;
        color: white !important;
    }
    
    /* Expander/collapsible hover effects */
    [data-baseweb="accordion"] summary:hover,
    .streamlit-expanderHeader:hover {
        color: var(--lavender) !important;
    }
    
    /* Links and clickable text hover */
    a:hover,
    [role="link"]:hover {
        color: var(--lavender) !important;
    }
    
    /* Checkbox and radio button checked states */
    input[type="checkbox"]:checked,
    input[type="radio"]:checked {
        accent-color: var(--lavender) !important;
    }
    
    /* Override any primary color theme variables */
    * {
        --primary-color: var(--lavender) !important;
        --primary-hover: var(--lavender-hover) !important;
    }
    
    /* Vertical alignment for search bar and button */
    /* Make the button column align properly with input */
    div[data-testid="column"] {
        display: flex !important;
        flex-direction: column !important;
        justify-content: flex-end !important;
    }
    
    /* Remove extra spacing from button container */
    div[data-testid="column"] .stButton {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    
    /* Ensure input and button have same height and alignment */
    .stTextInput > div > div > input {
        height: 38px !important;
    }
    
    button[kind="primary"] {
        height: 38px !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    
    /* Remove the label space from text input to align better */
    .stTextInput > label {
        display: none !important;
    }
    
    /* Underline headings in Report Summary section */
    h2:has-text("Report Summary"),
    h3:has-text("Overall Summary"),
    h3:has-text("Report Statistics") {
        text-decoration: underline !important;
    }
    
    /* Alternative selector for headings */
    h2:contains("Report Summary"),
    h3:contains("Overall Summary"),
    h3:contains("Report Statistics") {
        text-decoration: underline !important;
    }
    </style>
    <script>
    // Underline summary section headings
    function underlineSummaryHeadings() {
        const headings = document.querySelectorAll('h2, h3');
        headings.forEach(heading => {
            const text = heading.textContent.trim();
            if (text === 'Report Summary' || 
                text === 'Overall Summary' || 
                text === 'Report Statistics') {
                heading.style.textDecoration = 'underline';
            }
        });
    }
    
    // Run on page load
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', underlineSummaryHeadings);
    } else {
        underlineSummaryHeadings();
    }
    
    // Watch for Streamlit re-renders
    let timeoutId;
    const observer = new MutationObserver(() => {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(underlineSummaryHeadings, 100);
    });
    observer.observe(document.body, { childList: true, subtree: true });
    </script>
    """, unsafe_allow_html=True)
    
    st.title("MarketVantage")
    st.markdown("Generate comprehensive technology landscape reports and market analysis in a matter of seconds")
    
    # Main controls
    # URLs per section fixed at 10
    max_urls = 10
    
    # Place search bar and button side by side
    col_search, col_button = st.columns([4, 1])
    with col_search:
        topic = st.text_input(
            label="",
            placeholder="Enter topic you'd like to research (eg: AI Video Generation, AI healthcare...)",
            help="Enter the technology or topic you want to analyze",
        )
    
    with col_button:
        generate_clicked = st.button("Generate Report", type="primary", use_container_width=True)
    
    # Generate button logic
    if generate_clicked:
        if not topic or not topic.strip():
            st.error("Please enter a technology topic.")
            st.stop()
        
        # Clear old report data when generating new report
        if "report_data" in st.session_state:
            del st.session_state["report_data"]
        if "overall_summary" in st.session_state:
            del st.session_state["overall_summary"]
        if "summary_topic" in st.session_state:
            del st.session_state["summary_topic"]
        if "key_highlights" in st.session_state:
            del st.session_state["key_highlights"]
        if "highlights_topic" in st.session_state:
            del st.session_state["highlights_topic"]
        if "pdf_path" in st.session_state:
            del st.session_state["pdf_path"]
        if "pdf_filename" in st.session_state:
            del st.session_state["pdf_filename"]
        # Clear Q&A data when generating new report
        if "qa_answer" in st.session_state:
            del st.session_state["qa_answer"]
        if "qa_selected_chunks" in st.session_state:
            del st.session_state["qa_selected_chunks"]
        
        # Status message and progress bar
        status_text = st.empty()
        progress_bar = st.progress(0)
        
        try:
            # Show "Generating report..." throughout the entire process
            status_text.markdown("**Generating report...**")
            progress_bar.progress(10)
            
            report_data = generate_report(
                topic=topic.strip(),
                max_urls_per_section=max_urls,
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            )
            
            progress_bar.progress(80)
            
            # Generate PDF automatically
            metadata = report_data.get("metadata", {})
            topic_slug = slugify(metadata.get("topic", "report"))
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pdf_filename = f"{topic_slug}_{timestamp}.pdf"
            pdf_path = OUTPUT_DIR / pdf_filename
            
            create_pdf_report(report_data, pdf_path)
            LOGGER.info(f"PDF generated successfully: {pdf_path}")
            
            # Store report data and PDF info in session state
            st.session_state["report_data"] = report_data
            st.session_state["pdf_path"] = str(pdf_path)
            st.session_state["pdf_filename"] = pdf_filename
            
            progress_bar.progress(100)
            status_text.empty()
            
            st.success("Report generated successfully!")
            
        except ValueError as e:
            status_text.empty()
            progress_bar.empty()
            st.error(f"Report generation failed: {e}")
            LOGGER.exception("Report generation ValueError")
            st.stop()
        except Exception as e:
            status_text.empty()
            progress_bar.empty()
            st.error(f"Unexpected error: {e}. Check logs for details.")
            LOGGER.exception("Report generation error")
            st.stop()
    
    # Display report if available
    if "report_data" in st.session_state:
        report_data = st.session_state["report_data"]
        
        # Render Q&A sidebar
        render_qa_sidebar(report_data)
        
        # Report Summary
        st.markdown("---")
        st.markdown("## Report Summary")
        
        sections = report_data.get("sections", {})
        citations = report_data.get("citations", [])
        metadata = report_data.get("metadata", {})
        
        topic = metadata.get('topic', 'N/A')
        generated_at = metadata.get('generated_at', 'N/A')
        
        # Format date
        try:
            dt = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
            date_str = dt.strftime("%B %d, %Y at %I:%M %p")
        except Exception:
            date_str = generated_at
        
        # Report metadata
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Topic:** {topic}")
        with col2:
            st.markdown(f"**Generated:** {date_str}")
        
        # Generate Overall Summary using LLM
        st.markdown("### Overall Summary")
        if "overall_summary" not in st.session_state or st.session_state.get("summary_topic") != topic:
            # Generate summary from all sections using LLM
            all_sections_text = "\n\n".join([
                f"## {name}\n{content}" 
                for name, content in sections.items() 
                if name != "Executive Summary" and content
            ])
            
            if all_sections_text:
                summary_prompt = (
                    f"Provide a very concise 2-paragraph summary of this technology landscape report about {topic}. "
                    "Focus only on the most critical insights. Keep it brief and direct.\n\n"
                    "IMPORTANT: Write ONLY the summary paragraphs. Do NOT include any introductory phrases like "
                    "'Here is a summary' or 'Based on the report'. Start directly with the content.\n\n"
                    "Report Content:\n{all_sections}"
                ).format(all_sections=all_sections_text)
                
                with st.spinner("Generating summary..."):
                    try:
                        overall_summary = generate_answer(
                            summary_prompt,
                            [],  # No additional context chunks needed
                            max_output_tokens=300
                        )
                        # Clean up any introductory phrases that might still appear
                        import re
                        # Remove common introductory patterns - more aggressive matching
                        patterns = [
                            r"^Here'?s?\s+(?:a\s+)?(?:brief\s+)?(?:concise\s+)?(?:overall\s+)?(?:2-paragraph\s+)?(?:paragraph\s+)?summary\s+(?:of\s+)?(?:the\s+)?(?:[^:]*technology\s+landscape\s+)?(?:[^:]*report)?[:\s]*",
                            r"^Here'?s?\s+.*?summary\s+of.*?report[:\s]*",
                            r"^(?:Based on|According to|From)\s+(?:the|this)\s+(?:comprehensive\s+)?(?:technology\s+landscape\s+)?report[:\s]*",
                            r"^(?:This|The)\s+report\s+(?:shows|indicates|reveals)[:\s]*",
                            r"^(?:Following|Below)\s+is\s+(?:a|the)\s+summary[:\s]*",
                            r"^.*?summary\s+of\s+.*?report[:\s]*",
                        ]
                        for pattern in patterns:
                            overall_summary = re.sub(pattern, "", overall_summary, flags=re.IGNORECASE | re.DOTALL).strip()
                        # Remove any leading colons, dashes, or newlines
                        overall_summary = re.sub(r"^[:\-\s\n]+", "", overall_summary).strip()
                        # Remove any lines that are just introductory text (lines that end with colon and are short)
                        lines = overall_summary.split('\n')
                        cleaned_lines = []
                        for line in lines:
                            line_stripped = line.strip()
                            # Skip lines that look like introductions (short, end with colon, contain "summary" or "report")
                            if (len(line_stripped) < 80 and 
                                line_stripped.endswith(':') and 
                                (re.search(r'summary|report', line_stripped, re.IGNORECASE))):
                                continue
                            cleaned_lines.append(line)
                        overall_summary = '\n'.join(cleaned_lines).strip()
                        
                        # Clean markdown formatting for consistent UI display
                        overall_summary = clean_markdown_for_ui(overall_summary)
                        
                        st.session_state["overall_summary"] = overall_summary
                        st.session_state["summary_topic"] = topic
                    except Exception as e:
                        LOGGER.error(f"Error generating overall summary: {e}", exc_info=True)
                        st.error(f"Error generating summary: {e}")
                        overall_summary = "Summary generation failed. Please try again."
                        st.session_state["overall_summary"] = overall_summary
            else:
                overall_summary = "Report content is being generated. Please wait..."
                st.session_state["overall_summary"] = overall_summary
        else:
            overall_summary = st.session_state["overall_summary"]
        
        # Clean markdown before displaying to ensure consistency
        overall_summary_cleaned = clean_markdown_for_ui(overall_summary)
        st.markdown(overall_summary_cleaned)
        
        # Report Statistics
        st.markdown("### Report Statistics")
        total_words = sum(len(str(content).split()) for content in sections.values())
        # Calculate reading time (average reading speed: 200 words per minute)
        reading_time_minutes = max(1, round(total_words / 200))
        
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        with stat_col1:
            st.metric("Sections", len(sections))
        with stat_col2:
            st.metric("Citations", len(citations))
        with stat_col3:
            st.metric("Total Words", f"{total_words:,}")
        with stat_col4:
            if reading_time_minutes == 1:
                time_display = "1 minute"
            else:
                time_display = f"{reading_time_minutes} minutes"
            st.metric("Time to Read", time_display)
        
        # PDF Download
        # PDF Download - PDF is already generated, just provide download button
        st.markdown("---")
        st.markdown("## Download Report")
        
        if "pdf_path" in st.session_state and "pdf_filename" in st.session_state:
            pdf_path = st.session_state["pdf_path"]
            pdf_filename = st.session_state["pdf_filename"]
            
            try:
                if Path(pdf_path).exists():
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            label="Download PDF",
                            data=f.read(),
                            file_name=pdf_filename,
                            mime="application/pdf",
                            type="primary",
                            use_container_width=True,
                        )
                else:
                    st.error(f"PDF file not found: {pdf_path}")
            except Exception as e:
                LOGGER.error(f"Error reading PDF: {e}", exc_info=True)
                st.error(f"Error reading PDF: {e}")


if __name__ == "__main__":
    render()
