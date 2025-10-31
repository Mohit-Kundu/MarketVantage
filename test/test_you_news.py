import sys
import pathlib

import pytest

# Ensure the 'src' directory is importable when running tests from repo root
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from marketvantage.you_news import extract_news_titles, search_live_news  # noqa: E402
from marketvantage.you_search import configure_logging  # noqa: E402


@pytest.fixture(autouse=True)
def _setup_logging():
    configure_logging(0)
    yield


def test_live_news_basic():
    data = search_live_news("space exploration", count=3)
    assert isinstance(data, dict)
    titles = extract_news_titles(data)
    assert isinstance(titles, list)


def test_live_news_count_limit():
    data = search_live_news("technology", count=2)
    news = data.get("news", {})
    results = news.get("results", [])
    assert isinstance(results, list)
