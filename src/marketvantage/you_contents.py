import logging
from typing import Any, Dict, List, Literal, Optional

import requests

from .you_search import _get_api_key, configure_logging


LOGGER = logging.getLogger("marketvantage.you_contents")


def fetch_contents(
    urls: List[str],
    *,
    format: Literal["html", "markdown"] = "html",
    timeout_seconds: float = 20.0,
    api_key_env=("YOUCOM_API_KEY", "YOU_API_KEY"),
) -> List[Dict[str, Any]]:
    """Fetch page contents via You.com Contents API.

    Docs: https://documentation.you.com/api-reference/contents
    """
    if not urls or not all(isinstance(u, str) and u.strip() for u in urls):
        raise ValueError("urls must be a non-empty list of non-empty strings")

    api_key = _get_api_key(api_key_env)

    url = "https://api.ydc-index.io/v1/contents"
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    payload: Dict[str, Any] = {"urls": urls, "format": format}

    LOGGER.info("Sending You.com contents request", extra={"num_urls": len(urls), "format": format})

    response = requests.post(url, headers=headers, json=payload, timeout=timeout_seconds)
    try:
        response.raise_for_status()
    except requests.HTTPError as http_err:
        LOGGER.error(
            "You.com Contents API request failed",
            extra={
                "status_code": response.status_code,
                "reason": response.reason,
                "text": response.text[:500],
            },
        )
        raise http_err

    data = response.json()
    if not isinstance(data, list):
        LOGGER.warning("Unexpected contents response shape", extra={"type": type(data).__name__, "data_keys": list(data.keys()) if isinstance(data, dict) else None})
        # If it's a dict with a 'contents' key, extract that
        if isinstance(data, dict) and "contents" in data:
            data = data["contents"]
        elif isinstance(data, dict) and "results" in data:
            data = data["results"]
        else:
            # Return empty list if we can't figure out the structure
            LOGGER.error(f"Could not extract contents from response: {data}")
            return []
    return data  # list of {url,title,html,markdown}


def extract_titles(contents_response: List[Dict[str, Any]]) -> List[str]:
    titles: List[str] = []
    for item in contents_response:
        if isinstance(item, dict):
            title = item.get("title")
            if isinstance(title, str) and title.strip():
                titles.append(title)
    return titles


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Fetch web page contents via You.com Contents API.")
    parser.add_argument("urls", nargs="+", help="One or more URLs to fetch")
    parser.add_argument("--format", choices=["html", "markdown"], default="html", help="Return format")
    parser.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout in seconds (default: 20)")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (-v, -vv)")

    args = parser.parse_args()

    configure_logging(args.verbose)

    data = fetch_contents(args.urls, format=args.format, timeout_seconds=args.timeout)
    titles = extract_titles(data)

    if titles:
        for t in titles:
            print(t)
    else:
        print("No titles found in contents response.")


if __name__ == "__main__":
    main()
