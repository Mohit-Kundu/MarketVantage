import logging
from typing import Any, Dict, List, Optional

import requests

from .you_search import _get_api_key, configure_logging


LOGGER = logging.getLogger("marketvantage.you_news")


def search_live_news(
    query: str,
    *,
    count: int = 5,
    timeout_seconds: float = 15.0,
    api_key_env=("YOUCOM_API_KEY", "YOU_API_KEY"),
) -> Dict[str, Any]:
    """Query You.com Live News API and return parsed JSON.

    Docs: https://documentation.you.com/api-reference/news
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")

    api_key = _get_api_key(api_key_env)

    url = "https://api.ydc-index.io/livenews"
    headers = {"X-API-Key": api_key}
    params: Dict[str, Any] = {"q": query, "count": count}

    LOGGER.info("Sending You.com live news request", extra={"query": query})

    response = requests.get(url, headers=headers, params=params, timeout=timeout_seconds)
    try:
        response.raise_for_status()
    except requests.HTTPError as http_err:
        LOGGER.error(
            "You.com Live News API request failed",
            extra={
                "status_code": response.status_code,
                "reason": response.reason,
                "text": response.text[:500],
            },
        )
        raise http_err

    data = response.json()
    LOGGER.debug("You.com Live News API response received", extra={"keys": list(data.keys())})
    return data


def extract_news_titles(api_response: Dict[str, Any]) -> List[str]:
    """Extract news result titles from Live News API response."""
    news = api_response.get("news", {})
    results = news.get("results", [])
    titles = [item.get("title", "") for item in results if isinstance(item, dict)]
    return [t for t in titles if t]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Fetch live news via You.com API and print titles.")
    parser.add_argument("query", help="News query text")
    parser.add_argument("--count", type=int, default=5, help="Max news results (default: 5)")
    parser.add_argument("--timeout", type=float, default=15.0, help="HTTP timeout in seconds (default: 15)")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (-v, -vv)")

    args = parser.parse_args()

    configure_logging(args.verbose)

    data = search_live_news(args.query, count=args.count, timeout_seconds=args.timeout)
    titles = extract_news_titles(data)

    if titles:
        for title in titles:
            print(title)
    else:
        print("No news titles found in response.")


if __name__ == "__main__":
    main()
