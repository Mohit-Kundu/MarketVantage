import sys
import pathlib

import pytest

# Ensure the 'src' directory is importable when running tests from repo root
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from marketvantage.you_search import configure_logging, extract_web_results_titles, search_you  # noqa: E402


@pytest.fixture(autouse=True)
def _setup_logging():
    configure_logging(0)
    yield


def test_search_you_live_basic():
    data = search_you("python programming", count=3, freshness="year")
    assert isinstance(data, dict)
    titles = extract_web_results_titles(data)
    # We cannot guarantee non-empty, but API should return structure
    assert isinstance(titles, list)


def test_search_you_live_with_options():
    data = search_you("news this week", count=2, freshness="week", safesearch="moderate")
    assert "results" in data
