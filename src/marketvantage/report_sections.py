"""Section templates and metadata for Technology Landscape Reports."""
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class SectionTemplate:
    """Template for a report section."""
    name: str
    number: int
    prompt_template: str
    api_query_hint: Optional[str] = None
    uses_news_api: bool = False


REPORT_SECTIONS: Dict[int, SectionTemplate] = {
    1: SectionTemplate(
        name="Cover Page",
        number=1,
        prompt_template="",  # No generation needed
        api_query_hint=None,
    ),
    2: SectionTemplate(
        name="Executive Summary",
        number=2,
        prompt_template=(
            "Based on the following comprehensive research about {topic}, write a concise executive summary "
            "highlighting the most critical findings. Focus on:\n"
            "- Technology overview and current state\n"
            "- Key market players and competitive landscape\n"
            "- Recent developments and momentum\n"
            "- Future outlook and opportunities\n\n"
            "Keep it to 2-3 paragraphs, written for executive audiences.\n\n"
            "CRITICAL: Write ONLY the executive summary paragraphs. Do NOT include any additional points like:\n"
            "- Any 'References' or 'Cited Sources' section\n"
            "- Any 'Conclusion' paragraph\n"
            "- Any 'Recommendations' section or list of recommendations\n"
            "- Any action items or bullet-pointed suggestions\n"
            "- Any list of URLs or citations\n"
            "Stop immediately after the summary content.\n\n"
            "Research Content:\n{all_sections}"
        ),
        api_query_hint=None,  # Generated last from other sections
    ),
    3: SectionTemplate(
        name="Technology Overview",
        number=3,
        prompt_template=(
            "Provide a detailed overview of {topic}. Your response should cover:\n"
            "- What is {topic} and what problem does it solve?\n"
            "- Primary applications and use cases\n"
            "- Current maturity level (emerging, growing, mature)\n"
            "- Technical capabilities and limitations\n\n"
            "Write in clear, professional language suitable for a technology landscape report. "
            "Use the provided sources and cite them appropriately with [1], [2], etc.\n\n"
            "IMPORTANT: Do NOT include a 'References' section, 'Cited Sources' list, or any 'Conclusion' paragraph. "
            "Present only the main content - references will be compiled separately at the end of the report."
        ),
        api_query_hint="{topic} overview applications",
    ),
    4: SectionTemplate(
        name="Key Players & Market Landscape",
        number=4,
        prompt_template=(
            "Provide a comprehensive analysis of the competitive landscape and key market players for {topic}. "
            "Write this as a well-structured narrative that covers:\n\n"
            "1. Market Overview: Start with 1-2 paragraphs describing the overall competitive landscape, "
            "market structure, and key dynamics. Identify whether the market is consolidated, fragmented, "
            "or dominated by a few players.\n\n"
            "2. Major Market Leaders: Describe the top 3-5 major established companies or organizations in this space. "
            "For each, explain their market position, core offerings, competitive advantages, and strategic approach. "
            "Connect these into a cohesive narrative rather than listing them.\n\n"
            "3. Emerging Players: Discuss significant startups, newcomers, or innovative entrants that are gaining "
            "traction. Explain what makes them notable and how they're differentiating themselves.\n\n"
            "4. Market Dynamics: Describe the competitive dynamics, including partnerships, acquisitions, "
            "market share trends, and how players are positioning themselves.\n\n"
            "Write in a narrative, analytical style suitable for a professional technology landscape report. "
            "Use paragraphs and connecting text rather than lists. Use the provided sources and cite appropriately with [1], [2], etc.\n\n"
            "IMPORTANT: Do NOT format as a bullet list or simple enumeration. Write as flowing narrative paragraphs. "
            "Do NOT include a 'References' section, 'Cited Sources' list, or any 'Conclusion' paragraph. "
            "Present only the main content - references will be compiled separately at the end of the report."
        ),
        api_query_hint="{topic} companies startups market leaders",
    ),
    5: SectionTemplate(
        name="Recent News & Advancements",
        number=5,
        prompt_template=(
            "Summarize the latest news and most recent breakthroughs for {topic} in 2024 and 2025. "
            "Include:\n"
            "- Top 3-5 recent developments\n"
            "- Significant product launches, partnerships, or announcements\n"
            "- Technical breakthroughs or research findings\n"
            "- Market movements or funding news\n\n"
            "Organize chronologically or by significance. Use the provided news sources and cite appropriately.\n\n"
            "IMPORTANT: Do NOT include a 'References' section, 'Cited Sources' list, or any 'Conclusion' paragraph. "
            "Present only the main content - references will be compiled separately at the end of the report."
        ),
        api_query_hint="{topic}",
        uses_news_api=True,
    ),
    6: SectionTemplate(
        name="Market Trends & Future Outlook",
        number=6,
        prompt_template=(
            "Analyze the future market trends and outlook for {topic}. Address:\n"
            "- Long-term market projections and growth potential\n"
            "- Emerging trends and directions\n"
            "- New applications or use cases on the horizon\n"
            "- Industry adoption patterns\n"
            "- Technology evolution trajectory\n\n"
            "Provide forward-looking insights based on current developments. "
            "Use the provided sources and cite appropriately.\n\n"
            "IMPORTANT: Do NOT include a 'References' section, 'Cited Sources' list, or any 'Conclusion' paragraph. "
            "Present only the main content - references will be compiled separately at the end of the report."
        ),
        api_query_hint="{topic} future trends outlook 2025 2026",
    ),
    7: SectionTemplate(
        name="Opportunities & White Space",
        number=7,
        prompt_template=(
            "Identify the primary challenges, limitations, and unsolved problems associated with {topic}. "
            "Also highlight opportunities and 'white space' for innovation:\n"
            "- Current limitations or pain points\n"
            "- Gaps in the market or technology\n"
            "- Unaddressed use cases or customer needs\n"
            "- Opportunities for new entrants or innovation\n"
            "- Areas requiring further research or development\n\n"
            "Focus on actionable insights for strategic decision-making. Use the provided sources and cite appropriately.\n\n"
            "IMPORTANT: Do NOT include a 'References' section, 'Cited Sources' list, or any 'Conclusion' paragraph. "
            "Present only the main content - references will be compiled separately at the end of the report."
        ),
        api_query_hint="{topic} challenges limitations opportunities gaps",
    ),
    8: SectionTemplate(
        name="Conclusion",
        number=8,
        prompt_template=(
            "Based on the following comprehensive research about {topic}, write a concise conclusion. "
            "Synthesize the key findings and provide final insights:\n"
            "- Overall assessment of the current state and maturity of {topic}\n"
            "- Most significant trends and developments\n"
            "- Strategic implications for stakeholders\n"
            "- Final outlook and key takeaways\n\n"
            "Keep it to 2-3 paragraphs. This is the closing section before references.\n\n"
            "CRITICAL: Write ONLY the conclusion paragraphs. Do NOT include:\n"
            "- Any 'References' or 'Cited Sources' section\n"
            "- Any 'Recommendations' section or list of recommendations\n"
            "- Any action items, bullet points, or numbered suggestions\n"
            "- Any list of URLs or citations\n"
            "Stop immediately after the conclusion content.\n\n"
            "Research Content:\n{all_sections}"
        ),
        api_query_hint=None,  # Generated last from other sections
    ),
    9: SectionTemplate(
        name="References",
        number=9,
        prompt_template="",  # No generation, just list citations
        api_query_hint=None,
    ),
}


def get_section_template(section_number: int) -> Optional[SectionTemplate]:
    """Get section template by number."""
    return REPORT_SECTIONS.get(section_number)


def get_sections_requiring_api() -> Dict[int, SectionTemplate]:
    """Get all sections that require API queries (sections 3-7)."""
    return {k: v for k, v in REPORT_SECTIONS.items() if v.api_query_hint is not None}
