"""PDF generation using ReportLab."""
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    Table,
    TableStyle,
)
from reportlab.pdfgen import canvas
import re

LOGGER = logging.getLogger("marketvantage.pdf_generator")


def _strip_markdown(text: str) -> str:
    """Simple markdown to plain text conversion."""
    import re
    # Remove markdown links but keep text
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    # Remove markdown headers
    text = re.sub(r"^#+\s+", "", text, flags=re.MULTILINE)
    # Remove markdown bold/italic
    text = re.sub(r"\*\*([^\*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^\*]+)\*", r"\1", text)
    # Remove markdown code blocks
    text = re.sub(r"```[^\n]*\n([^`]+)```", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Clean up extra whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _clean_text_for_pdf(text: str, citation_map: Dict[int, str]) -> str:
    """Clean text for PDF rendering with hyperlinked citations."""
    # First, convert markdown bold (**text**) to HTML bold (<b>text</b>)
    # Protect from being escaped later
    bold_pattern = r"\*\*([^*]+)\*\*"
    bold_replacements = {}
    
    def store_bold(match):
        bold_text = match.group(1)
        key = f"__BOLD_{len(bold_replacements)}__"
        bold_replacements[key] = bold_text
        return key
    
    text = re.sub(bold_pattern, store_bold, text)
    
    # Handle markdown italic (*text*) if needed
    italic_pattern = r"(?<!\*)\*([^*]+)\*(?!\*)"
    italic_replacements = {}
    
    def store_italic(match):
        italic_text = match.group(1)
        key = f"__ITALIC_{len(italic_replacements)}__"
        italic_replacements[key] = italic_text
        return key
    
    text = re.sub(italic_pattern, store_italic, text)
    
    # First, handle citation ranges like [1-3] before single citations
    range_pattern = r"\[(\d+)-(\d+)\]"
    range_replacements = {}
    
    def store_range(match):
        start_num = int(match.group(1))
        end_num = int(match.group(2))
        key = f"__RANGE_{start_num}_{end_num}__"
        if start_num in citation_map:
            url = citation_map[start_num]
            url_escaped = url.replace("&", "&amp;").replace('"', "&quot;")
            range_replacements[key] = f'<a href="{url_escaped}" color="blue"><u>[{start_num}-{end_num}]</u></a>'
        else:
            range_replacements[key] = f"[{start_num}-{end_num}]"
        return key
    
    # Replace ranges first
    text = re.sub(range_pattern, store_range, text)
    
    # Now handle single citations [1], [2], etc.
    citation_pattern = r"\[(\d+)\]"
    citations_found = []
    
    def store_citation(match):
        cite_num = int(match.group(1))
        citations_found.append(cite_num)
        return f"__CITATION_{len(citations_found)-1}__"
    
    # Replace citations with placeholders
    text = re.sub(citation_pattern, store_citation, text)
    
    # Escape HTML in the text (but protect placeholders first)
    # Temporarily replace placeholders with safe markers that won't be escaped
    placeholder_map = {}
    
    # Include all placeholders: bold, italic, range, and citations
    citation_placeholders = [f"__CITATION_{idx}__" for idx in range(len(citations_found))]
    all_placeholders = (
        list(bold_replacements.keys()) + 
        list(italic_replacements.keys()) + 
        list(range_replacements.keys()) + 
        citation_placeholders
    )
    
    safe_markers = []
    for placeholder in all_placeholders:
        safe_marker = f"__SAFE_{len(placeholder_map)}__"
        placeholder_map[safe_marker] = placeholder
        safe_markers.append((placeholder, safe_marker))
    
    # Replace placeholders with safe markers (in reverse order to avoid conflicts)
    for placeholder, safe_marker in reversed(safe_markers):
        text = text.replace(placeholder, safe_marker)
    
    # Now escape HTML
    text = _escape_html(text)
    
    # Restore placeholders (in order)
    for safe_marker, placeholder in placeholder_map.items():
        text = text.replace(safe_marker, placeholder)
    
    # Restore bold formatting - escape the bold text content
    for placeholder, bold_text in bold_replacements.items():
        # Bold text was already captured, just escape it for HTML
        escaped_bold = _escape_html(bold_text)
        # Replace placeholder with HTML bold tag - replace all occurrences
        replacement = f"<b>{escaped_bold}</b>"
        if placeholder in text:
            text = text.replace(placeholder, replacement)
        else:
            LOGGER.warning(f"Bold placeholder {placeholder} not found in text after HTML escaping")
    
    # Restore italic formatting
    for placeholder, italic_text in italic_replacements.items():
        escaped_italic = _escape_html(italic_text)
        replacement = f"<i>{escaped_italic}</i>"
        if placeholder in text:
            text = text.replace(placeholder, replacement)
        else:
            LOGGER.warning(f"Italic placeholder {placeholder} not found in text after HTML escaping")
    
    # Restore range replacements
    for placeholder, replacement in range_replacements.items():
        text = text.replace(placeholder, replacement)
    
    # Restore citations as hyperlinks
    for idx, cite_num in enumerate(citations_found):
        url = citation_map.get(cite_num, "")
        placeholder = f"__CITATION_{idx}__"
        if url:
            # Escape URL for HTML attribute
            url_escaped = url.replace("&", "&amp;").replace('"', "&quot;")
            link_tag = f'<a href="{url_escaped}" color="blue"><u>[{cite_num}]</u></a>'
            text = text.replace(placeholder, link_tag)
        else:
            text = text.replace(placeholder, f"[{cite_num}]")
    
    # Final cleanup: remove any remaining placeholders that weren't replaced
    # This is a safety net in case something went wrong
    # Only remove if they're still there after all replacements
    remaining_placeholders = re.findall(r'__(?:BOLD|ITALIC|CITATION|RANGE|PLACEHOLDER)_\d+(?:_\d+)?__', text)
    if remaining_placeholders:
        LOGGER.warning(f"Found {len(remaining_placeholders)} unreplaced placeholders in PDF text: {remaining_placeholders[:5]}")
        # Remove them to prevent them appearing in the PDF
        text = re.sub(r'__(?:BOLD|ITALIC|CITATION|RANGE|PLACEHOLDER)_\d+(?:_\d+)?__', '', text)
    
    # Convert line breaks
    text = text.replace("\\n\\n", "<br/><br/>")
    text = text.replace("\\n", "<br/>")
    text = text.replace("\n\n", "<br/><br/>")
    text = text.replace("\n", "<br/>")
    
    return text


def create_pdf_report(report_data: Dict[str, Any], output_path: Path) -> None:
    """
    Create a PDF report from report data.
    
    Args:
        report_data: Dict with keys:
            - sections: Dict[str, str] - section name -> content
            - citations: List[str] - source URLs
            - metadata: Dict with topic, generated_at, etc.
        output_path: Path where PDF will be saved
    """
    LOGGER.info(f"Starting PDF generation for: {output_path}")
    LOGGER.info(f"Report has {len(report_data.get('sections', {}))} sections and {len(report_data.get('citations', []))} citations")
    
    styles = getSampleStyleSheet()
    
    # Define custom styles
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontSize=24,
        textColor=colors.HexColor("#1f4788"),
        spaceAfter=30,
        alignment=1,  # Center
    )
    
    subtitle_style = ParagraphStyle(
        "CustomSubtitle",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=colors.HexColor("#666666"),
        spaceAfter=20,
        alignment=1,  # Center
    )
    
    heading1_style = ParagraphStyle(
        "CustomHeading1",
        parent=styles["Heading1"],
        fontSize=16,
        textColor=colors.HexColor("#1f4788"),
        spaceAfter=12,
        spaceBefore=24,
    )
    
    heading2_style = ParagraphStyle(
        "CustomHeading2",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=colors.HexColor("#333333"),
        spaceAfter=10,
        spaceBefore=18,
    )
    
    normal_style = ParagraphStyle(
        "CustomNormal",
        parent=styles["Normal"],
        fontSize=11,
        leading=14,
        spaceAfter=12,
    )
    
    # Style for subheadings (bold text paragraphs) - reduced spacing
    subheading_style = ParagraphStyle(
        "Subheading",
        parent=styles["Normal"],
        fontSize=11,
        leading=14,
        spaceAfter=6,  # Reduced space after subheadings (was 12)
        spaceBefore=10,  # Space before subheading
    )
    
    citation_style = ParagraphStyle(
        "Citation",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#666666"),
        leftIndent=20,
        spaceAfter=6,
    )
    
    # Build citation map (number -> URL) for hyperlinking
    citations = report_data.get("citations", [])
    citation_map: Dict[int, str] = {i + 1: url for i, url in enumerate(citations)}
    LOGGER.info(f"Created citation map with {len(citation_map)} citations")
    
    # Build story (content elements)
    story: List[Any] = []
    
    # Cover Page
    topic = report_data.get("metadata", {}).get("topic", "Technology Topic")
    generated_at = report_data.get("metadata", {}).get("generated_at", datetime.now().isoformat())
    
    try:
        dt = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
        date_str = dt.strftime("%B %d, %Y at %I:%M %p")
    except Exception:
        date_str = generated_at
    
    story.append(Spacer(1, 2 * inch))
    story.append(Paragraph("Technology Landscape Report", title_style))
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph(f"{topic}", heading1_style))
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph("An Automated Analysis", subtitle_style))
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph(f"Generated on: {date_str}", styles["Normal"]))
    story.append(Paragraph("Generated by: MarketVantage RAG System", styles["Normal"]))
    story.append(PageBreak())
    
    # Table of Contents with numbered formatting
    story.append(Paragraph("Table of Contents", heading1_style))
    story.append(Spacer(1, 0.3 * inch))
    
    sections = report_data.get("sections", {})
    
    # Define TOC style for numbered items
    toc_style = ParagraphStyle(
        "TOC",
        parent=styles["Normal"],
        fontSize=11,
        leading=16,
        spaceAfter=8,
        leftIndent=0,
        textColor=colors.HexColor("#333333"),
    )
    
    # Use the section_order to ensure consistent numbering
    section_order = [
        "Executive Summary",
        "Technology Overview",
        "Key Players & Market Landscape",
        "Recent News & Advancements",
        "Market Trends & Future Outlook",
        "Opportunities & White Space",
        "Conclusion",
    ]
    
    # Create numbered list of sections that exist
    numbered_sections = []
    for idx, section_name in enumerate(section_order, start=1):
        if section_name in sections and sections[section_name]:
            numbered_sections.append((idx, section_name))
    
    if numbered_sections:
        for number, section_name in numbered_sections:
            # Format as numbered bullet point: "1. Section Name"
            toc_text = f"{number}. {section_name}"
            story.append(Paragraph(toc_text, toc_style))
    else:
        # Fallback: use all sections in order they appear
        for idx, section_name in enumerate(sections.keys(), start=1):
            toc_text = f"{idx}. {section_name}"
            story.append(Paragraph(toc_text, toc_style))
    
    story.append(Spacer(1, 0.2 * inch))
    story.append(PageBreak())
    
    # Sections
    section_order = [
        "Executive Summary",
        "Technology Overview",
        "Key Players & Market Landscape",
        "Recent News & Advancements",
        "Market Trends & Future Outlook",
        "Opportunities & White Space",
        "Conclusion",  # Added conclusion before references
    ]
    
    # Render all content sections (references only appear in dedicated section at end)
    for section_name in section_order:
        if section_name in sections:
            content = sections[section_name]
            if content:
                story.append(Paragraph(section_name, heading1_style))
                # Split content into paragraphs
                paragraphs = content.split("\n\n")
                for para in paragraphs:
                    if para.strip():
                        # Convert in-text citations [1], [2] to hyperlinks (not full reference lists)
                        cleaned = _clean_text_for_pdf(para.strip(), citation_map)
                        
                        # Check if this paragraph is a subheading (starts with <b> tag after cleaning)
                        is_subheading = cleaned.strip().startswith('<b>') and cleaned.strip().endswith('</b>')
                        
                        # Use appropriate style
                        style = subheading_style if is_subheading else normal_style
                        story.append(Paragraph(cleaned, style))
                story.append(Spacer(1, 0.2 * inch))
    
    # References Section - All full URLs listed here at end of document
    # Add spacer instead of page break to avoid empty pages
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph("References", heading1_style))
    story.append(Spacer(1, 0.2 * inch))
    
    if citations:
        reference_style = ParagraphStyle(
            "Reference",
            parent=styles["Normal"],
            fontSize=10,
            leftIndent=0,
            spaceAfter=10,
            leading=14,
        )
        
        for i, url in enumerate(citations, 1):
            # Create clickable URL in reference
            url_escaped_for_href = url.replace("&", "&amp;").replace('"', "&quot;")
            # Escape URL for display but keep it readable
            url_display = _escape_html(url)
            # Format as numbered reference with clickable URL
            ref_text = f'<b>[{i}]</b> <a href="{url_escaped_for_href}" color="blue"><u>{url_display}</u></a>'
            story.append(Paragraph(ref_text, reference_style))
    else:
        story.append(Paragraph("No sources cited.", normal_style))
    
    # Create PDF
    output_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info(f"Building PDF document with {len(story)} elements...")
    try:
        doc = SimpleDocTemplate(str(output_path), pagesize=letter, topMargin=0.75 * inch, bottomMargin=0.75 * inch)
        doc.build(story)
        LOGGER.info(f"PDF report successfully saved to {output_path}")
    except Exception as e:
        LOGGER.error(f"Error building PDF: {e}", exc_info=True)
        raise
