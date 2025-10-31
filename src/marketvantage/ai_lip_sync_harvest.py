import os
import logging
import pathlib
from typing import List, Dict, Any, Set

from .you_search import configure_logging, search_you
from .you_news import search_live_news
from .you_contents import fetch_contents


LOGGER = logging.getLogger("marketvantage.ai_lip_sync_harvest")


def _extract_urls_from_search(search_response: Dict[str, Any]) -> List[str]:
    urls: List[str] = []
    results = search_response.get("results", {})
    for item in results.get("web", []) or []:
        url = item.get("url")
        if isinstance(url, str) and url.strip():
            urls.append(url)
    return urls


def _extract_urls_from_news(news_response: Dict[str, Any]) -> List[str]:
    urls: List[str] = []
    news = news_response.get("news", {})
    for item in news.get("results", []) or []:
        url = item.get("url")
        if isinstance(url, str) and url.strip():
            urls.append(url)
    return urls


def _write_markdown(outfile: pathlib.Path, contents: List[Dict[str, Any]]) -> None:
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with outfile.open("w", encoding="utf-8") as f:
        for entry in contents:
            url = entry.get("url", "")
            title = entry.get("title", "")
            md = entry.get("markdown") or ""
            if not md and entry.get("html"):
                md = entry.get("html")
            f.write(f"# {title}\n\n")
            f.write(f"Source: {url}\n\n")
            if md:
                f.write(str(md))
                if not str(md).endswith("\n"):
                    f.write("\n")
            f.write("\n\n---\n\n")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Search 'ai lip sync' (or custom query), fetch markdown contents of results, and write to a file."
    )
    parser.add_argument("--query", default="ai lip sync", help="Query string (default: 'ai lip sync')")
    parser.add_argument("--count", type=int, default=5, help="Max results per section to collect (default: 5)")
    parser.add_argument("--freshness", choices=["day", "week", "month", "year"], default="year", help="Freshness filter for search")
    parser.add_argument("--outfile", default="output/ai_lip_sync.md", help="Output markdown file path")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (-v, -vv)")

    args = parser.parse_args()

    configure_logging(args.verbose)

    LOGGER.info("Running search", extra={"query": args.query})
    search_data = search_you(args.query, count=args.count, freshness=args.freshness)
    news_data = search_live_news(args.query, count=args.count)

    urls: List[str] = []
    urls.extend(_extract_urls_from_search(search_data))
    urls.extend(_extract_urls_from_news(news_data))

    # Dedupe and cap to a reasonable number to avoid rate limits
    seen: Set[str] = set()
    deduped: List[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    deduped = deduped[: max(1, min(10, args.count * 2))]

    LOGGER.info("Fetching contents", extra={"num_urls": len(deduped)})
    contents = fetch_contents(deduped, format="markdown")

    outfile = pathlib.Path(args.outfile)
    _write_markdown(outfile, contents)

    print(str(outfile.resolve()))


if __name__ == "__main__":
    main()
