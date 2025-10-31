"""Test report engine functionality."""
import logging
import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from marketvantage.report_sections import REPORT_SECTIONS, get_sections_requiring_api
from marketvantage.report_engine import slugify, _aggregate_all_urls


def test_report_sections():
    """Test that report sections are properly defined."""
    assert len(REPORT_SECTIONS) == 8, f"Expected 8 sections, got {len(REPORT_SECTIONS)}"
    
    sections_needing_api = get_sections_requiring_api()
    assert len(sections_needing_api) == 5, f"Expected 5 sections needing API, got {len(sections_needing_api)}"
    
    # Check section numbers
    assert 1 in REPORT_SECTIONS, "Section 1 (Cover Page) missing"
    assert 2 in REPORT_SECTIONS, "Section 2 (Executive Summary) missing"
    assert 3 in REPORT_SECTIONS, "Section 3 (Technology Overview) missing"
    
    print("✓ Report sections structure OK")


def test_slugify():
    """Test slugify function."""
    assert slugify("AI Lip Sync") == "ai-lip-sync"
    assert slugify("Quantum Computing 2024") == "quantum-computing-2024"
    assert slugify("") == "topic"
    print("✓ slugify function OK")


def test_aggregate_urls():
    """Test URL aggregation."""
    section_data = {
        3: {"urls": ["url1", "url2"], "titles": ["t1", "t2"]},
        4: {"urls": ["url2", "url3"], "titles": ["t2", "t3"]},
        5: {"urls": ["url4"], "titles": ["t4"]},
    }
    urls = _aggregate_all_urls(section_data)
    assert len(urls) == 4, f"Expected 4 unique URLs, got {len(urls)}"
    assert ("url1", "t1") in urls
    assert ("url4", "t4") in urls
    print("✓ URL aggregation OK")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing report engine components...\n")
    
    test_report_sections()
    test_slugify()
    test_aggregate_urls()
    
    print("\n✓ All basic tests passed!")
