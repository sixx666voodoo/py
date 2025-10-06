"""Utilities for scraping the Borderlands 4 wiki from game8.co.

The :class:`WikiScraper` class implements a focused web crawler that walks all
pages under ``https://game8.co/games/Borderlands-4`` and stores the downloaded
HTML documents on disk.  The crawler keeps the scope limited to the wiki by
only following hyperlinks that share the same host and path prefix as the base
URL.  It optionally respects ``robots.txt`` directives, throttles requests to a
configurable delay, and exposes a simple command line interface.

The implementation relies solely on the Python standard library so it can be
used as a standalone script in addition to the existing tooling in this
repository.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import time
from collections import deque
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Deque, Iterable, Iterator, Optional, Set
from urllib import robotparser
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen


LOGGER = logging.getLogger(__name__)


def _sanitize_path_segment(segment: str) -> str:
    """Return a filesystem friendly version of ``segment``.

    The helper removes characters that are problematic on common filesystems
    and guarantees that an empty result is replaced by ``"index"``.
    """

    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", segment)
    return cleaned or "index"


def _path_from_url(base_dir: Path, url: str) -> Path:
    """Derive an output path for ``url`` rooted at ``base_dir``.

    URLs that point to directories (with or without trailing slashes) are saved
    as ``index.html`` inside the corresponding directory.  URLs that point to a
    file without an HTML extension are normalised to ``.html``.  Query strings
    are encoded as an eight-character SHA1 digest appended to the filename so
    that distinct URLs map to distinct files.
    """

    parsed = urlparse(url)
    path = parsed.path or "/"
    parts = [segment for segment in path.split("/") if segment]
    sanitized = [_sanitize_path_segment(part) for part in parts]

    if not parts or path.endswith("/"):
        # Directory index
        file_path = base_dir.joinpath(*sanitized, "index.html")
    else:
        filename = sanitized[-1]
        stem, suffix = os.path.splitext(filename)
        if suffix.lower() not in {".html", ".htm"}:
            filename = f"{filename}.html"
        sanitized[-1] = filename
        file_path = base_dir.joinpath(*sanitized)

    if parsed.query:
        digest = hashlib.sha1(parsed.query.encode("utf-8")).hexdigest()[:8]
        stem, suffix = os.path.splitext(file_path.name)
        file_path = file_path.with_name(f"{stem}_{digest}{suffix}")

    return file_path


class _LinkExtractor(HTMLParser):
    """Lightweight HTML parser that collects ``href`` values from anchors."""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if tag.lower() != "a":
            return
        for name, value in attrs:
            if name.lower() == "href" and value:
                self.links.append(value)
                break


class SimpleResponse:
    """Container for HTTP response data."""

    def __init__(self, url: str, text: str):
        self.url = url
        self.text = text

    def raise_for_status(self) -> None:  # pragma: no cover - compatibility shim
        return None


class SimpleSession:
    """Minimal HTTP client used when no custom session is provided."""

    def __init__(self, user_agent: str) -> None:
        self.user_agent = user_agent

    def get(
        self, url: str, headers: Optional[dict[str, str]] = None, timeout: float = 20.0
    ) -> SimpleResponse:
        request_headers = {"User-Agent": self.user_agent}
        if headers:
            request_headers.update(headers)
        request = Request(url, headers=request_headers)
        with urlopen(request, timeout=timeout) as handle:  # pragma: no cover - network I/O
            charset = handle.headers.get_content_charset() or "utf-8"
            text = handle.read().decode(charset, errors="replace")
        return SimpleResponse(url=url, text=text)


@dataclass
class WikiScraper:
    """Focused crawler that downloads pages from a specific wiki."""

    base_url: str
    output_dir: Path
    delay: float = 1.0
    session: Optional[SimpleSession] = None
    user_agent: str = "Borderlands4WikiScraper/1.0"
    respect_robots: bool = True
    robot_parser: Optional[robotparser.RobotFileParser] = None

    def __post_init__(self) -> None:
        parsed = urlparse(self.base_url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid base URL: {self.base_url}")
        self._base_netloc = parsed.netloc
        self._base_path = parsed.path.rstrip("/") or "/"
        self._session = self.session or SimpleSession(self.user_agent)
        self._visited: Set[str] = set()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # Public API -----------------------------------------------------------------
    def scrape(self, max_pages: Optional[int] = None) -> int:
        """Download the wiki starting from :pyattr:`base_url`.

        Args:
            max_pages: Optional limit for the number of pages to download.

        Returns:
            Number of successfully downloaded pages.
        """

        queue: Deque[str] = deque([self.base_url])
        pages_downloaded = 0

        while queue:
            current = queue.popleft()
            if current in self._visited:
                continue
            if max_pages is not None and pages_downloaded >= max_pages:
                break
            self._visited.add(current)

            if self.respect_robots and not self._allowed_by_robots(current):
                LOGGER.info("Skipping %s due to robots.txt", current)
                continue

            try:
                html = self._download(current)
            except (URLError, HTTPError) as exc:
                LOGGER.warning("Failed to download %s: %s", current, exc)
                continue

            self._store(current, html)
            pages_downloaded += 1

            for link in self._extract_links(current, html):
                if link not in self._visited and link not in queue:
                    queue.append(link)

            if self.delay:
                time.sleep(self.delay)

        return pages_downloaded

    # Internal helpers -----------------------------------------------------------
    def _download(self, url: str) -> str:
        LOGGER.info("Downloading %s", url)
        headers = {"User-Agent": self.user_agent}
        response = self._session.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        return response.text

    def _store(self, url: str, html: str) -> None:
        path = _path_from_url(self.output_dir, url)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")
        LOGGER.debug("Stored %s -> %s", url, path)

    def _extract_links(self, current_url: str, html: str) -> Iterable[str]:
        extractor = _LinkExtractor()
        extractor.feed(html)
        for href in extractor.links:
            if href.startswith("javascript:") or href.startswith("mailto:"):
                continue
            absolute = urljoin(current_url, href)
            if self._in_scope(absolute):
                yield absolute.split("#", 1)[0]

    def _in_scope(self, url: str) -> bool:
        parsed = urlparse(url)
        if parsed.netloc != self._base_netloc:
            return False
        path = parsed.path or "/"
        return path.startswith(self._base_path)

    def _allowed_by_robots(self, url: str) -> bool:
        parser = self._ensure_robot_parser()
        if parser is None:
            return True
        return parser.can_fetch(self.user_agent, url)

    def _ensure_robot_parser(self) -> Optional[robotparser.RobotFileParser]:
        if not self.respect_robots:
            return None
        if self.robot_parser is not None:
            return self.robot_parser
        robots_url = urljoin(f"https://{self._base_netloc}", "/robots.txt")
        parser = robotparser.RobotFileParser()
        try:
            response = self._session.get(robots_url, timeout=10)
            response.raise_for_status()
        except (URLError, HTTPError) as exc:
            LOGGER.warning("Unable to fetch robots.txt (%s): %s", robots_url, exc)
            self.robot_parser = parser
            return parser
        parser.parse(response.text.splitlines())
        self.robot_parser = parser
        return parser


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scrape the Borderlands 4 wiki from game8.co",
    )
    parser.add_argument(
        "--base-url",
        default="https://game8.co/games/Borderlands-4",
        help="Root URL of the wiki to scrape",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("borderlands4_wiki"),
        help="Directory where the HTML files will be stored",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Optional limit for the number of pages to download",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay in seconds between requests (default: 1.0)",
    )
    parser.add_argument(
        "--ignore-robots",
        action="store_true",
        help="Ignore robots.txt directives",
    )
    return parser


def main(argv: Optional[Iterator[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _build_argument_parser().parse_args(argv)
    scraper = WikiScraper(
        base_url=args.base_url,
        output_dir=args.output,
        delay=args.delay,
        respect_robots=not args.ignore_robots,
    )
    pages = scraper.scrape(max_pages=args.max_pages)
    LOGGER.info("Downloaded %d pages", pages)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
