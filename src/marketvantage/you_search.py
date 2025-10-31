import os
import logging
from typing import Any, Dict, List, Optional, Sequence

import requests

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore

from .logging_utils import setup_logging


LOGGER = logging.getLogger("marketvantage.you_search")


def _get_api_key(env_var_name: Sequence[str] | str = ("YOUCOM_API_KEY", "YOU_API_KEY")) -> str:
    """Return the You.com API key from environment or .env.

    Parameters
    ----------
    env_var_name: Sequence[str] | str
        Name(s) of the environment variable that hold the API key. Defaults to
        checking both "YOUCOM_API_KEY" and "YOU_API_KEY".

    Raises
    ------
    RuntimeError
        If the API key is not found in the environment or .env.
    """
    # Load .env if python-dotenv is available
    if load_dotenv is not None:
        try:
            load_dotenv(override=False)
        except Exception:
            pass

    names = (env_var_name,) if isinstance(env_var_name, str) else tuple(env_var_name)
    for name in names:
        api_key = os.getenv(name)
        if api_key:
            return api_key

    joined = ", ".join(names)
    raise RuntimeError(
        f"Missing API key. Set one of [{joined}] in environment or .env with your You.com API key."
    )


def search_you(
    query: str,
    *,
    count: int = 5,
    freshness: Optional[str] = None,
    country: Optional[str] = None,
    safesearch: Optional[str] = None,
    livecrawl: Optional[str] = None,
    livecrawl_formats: Optional[str] = None,
    timeout_seconds: float = 15.0,
    api_key_env: Sequence[str] | str = ("YOUCOM_API_KEY", "YOU_API_KEY"),
) -> Dict[str, Any]:
    """Query You.com search API and return parsed JSON.

    Notes
    -----
    Docs: https://documentation.you.com/api-reference/search

    Parameters
    ----------
    query: str
        The search query.
    count: int
        Max number of results per section (1..100).
    freshness: Optional[str]
        One of {"day", "week", "month", "year"}.
    country: Optional[str]
        Country code (e.g., "US", "IN").
    safesearch: Optional[str]
        One of {"off", "moderate", "strict"}.
    livecrawl: Optional[str]
        One of {"web", "news", "all"}.
    livecrawl_formats: Optional[str]
        One of {"html", "markdown"}.
    timeout_seconds: float
        HTTP timeout in seconds.
    api_key_env: Sequence[str] | str
        Environment variable name(s) holding the API key.

    Returns
    -------
    Dict[str, Any]
        Parsed JSON response from the API.

    Raises
    ------
    requests.HTTPError
        If the response status is not OK.
    RuntimeError
        If API key is missing.
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")

    api_key = _get_api_key(api_key_env)

    url = "https://api.ydc-index.io/v1/search"
    headers = {"X-API-Key": api_key}

    params: Dict[str, Any] = {"query": query, "count": count}
    if freshness:
        params["freshness"] = freshness
    if country:
        params["country"] = country
    if safesearch:
        params["safesearch"] = safesearch
    if livecrawl:
        params["livecrawl"] = livecrawl
    if livecrawl_formats:
        params["livecrawl_formats"] = livecrawl_formats

    LOGGER.info("Sending You.com search request", extra={"query": query})

    response = requests.get(url, headers=headers, params=params, timeout=timeout_seconds)
    try:
        response.raise_for_status()
    except requests.HTTPError as http_err:
        LOGGER.error(
            "You.com API request failed",
            extra={
                "status_code": response.status_code,
                "reason": response.reason,
                "text": response.text[:500],
            },
        )
        raise http_err

    data = response.json()
    LOGGER.debug("You.com API response received", extra={"keys": list(data.keys())})
    return data


def extract_web_results_titles(api_response: Dict[str, Any]) -> List[str]:
    """Extract web result titles from API response.

    Parameters
    ----------
    api_response: Dict[str, Any]
        JSON object returned by `search_you`.

    Returns
    -------
    List[str]
        Titles of web results if present; otherwise empty list.
    """
    results = api_response.get("results", {})
    web = results.get("web", [])
    titles = [item.get("title", "") for item in web if isinstance(item, dict)]
    return [t for t in titles if t]


def configure_logging(verbosity: int = 0, log_file: str = "logs/marketvantage.log") -> None:
    """Configure logging via central setup."""
    setup_logging(verbosity=verbosity, log_file=log_file)


def main() -> None:
    """CLI entry point: searches You.com and prints titles."""
    import argparse

    parser = argparse.ArgumentParser(description="Search the web via You.com API and print result titles.")
    parser.add_argument("query", help="Search query text")
    parser.add_argument("--count", type=int, default=5, help="Max results per section (default: 5)")
    parser.add_argument("--freshness", choices=["day", "week", "month", "year"], help="Result freshness filter")
    parser.add_argument("--country", help="Country code, e.g., US, IN")
    parser.add_argument("--safesearch", choices=["off", "moderate", "strict"], help="SafeSearch level")
    parser.add_argument("--livecrawl", choices=["web", "news", "all"], help="Livecrawl section")
    parser.add_argument("--livecrawl-formats", dest="livecrawl_formats", choices=["html", "markdown"], help="Livecrawl format")
    parser.add_argument("--timeout", type=float, default=15.0, help="HTTP timeout in seconds (default: 15)")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (-v, -vv)")

    args = parser.parse_args()

    configure_logging(args.verbose)

    data = search_you(
        args.query,
        count=args.count,
        freshness=args.freshness,
        country=args.country,
        safesearch=args.safesearch,
        livecrawl=args.livecrawl,
        livecrawl_formats=args.livecrawl_formats,
        timeout_seconds=args.timeout,
    )

    titles = extract_web_results_titles(data)
    if titles:
        for title in titles:
            print(title)
    else:
        print("No web titles found in response.")


if __name__ == "__main__":
    main()
