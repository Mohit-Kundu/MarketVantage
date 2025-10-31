import sys
import pathlib

import pytest

# Ensure the 'src' directory is importable when running tests from repo root
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from marketvantage.you_contents import fetch_contents, extract_titles  # noqa: E402
from marketvantage.you_search import configure_logging  # noqa: E402


@pytest.fixture(autouse=True)
def _setup_logging():
    configure_logging(0)
    yield


def test_contents_fetch_html():
    data = fetch_contents(["https://www.you.com"], format="html")
    assert isinstance(data, list)
    titles = extract_titles(data)
    assert isinstance(titles, list)


def test_contents_fetch_markdown_multiple():
    data = fetch_contents(["https://www.you.com", "https://example.com"], format="markdown")
    assert isinstance(data, list)
