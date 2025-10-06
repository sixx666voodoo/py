"""Tests for :mod:`wiki_scraper`."""

from __future__ import annotations

from pathlib import Path

from wiki_scraper import WikiScraper, _path_from_url


class FakeResponse:
    def __init__(self, text: str, status_code: int = 200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP error: {self.status_code}")


class FakeSession:
    def __init__(self, responses: dict[str, FakeResponse]):
        self.responses = responses
        self.requested: list[str] = []

    def get(self, url: str, headers=None, timeout=None):  # pragma: no cover - simple stub
        self.requested.append(url)
        try:
            return self.responses[url]
        except KeyError as exc:  # pragma: no cover - easier debugging
            raise AssertionError(f"Unexpected URL requested: {url}") from exc


def test_path_from_url_handles_directories_and_queries(tmp_path: Path) -> None:
    base = tmp_path
    file_path = _path_from_url(base, "https://example.com/wiki")
    assert file_path == base / "wiki.html"

    directory_path = _path_from_url(base, "https://example.com/wiki/page-one/")
    assert directory_path == base / "wiki" / "page-one" / "index.html"

    query_path = _path_from_url(base, "https://example.com/wiki/page-one?variant=1")
    assert query_path.parent == base / "wiki"
    assert query_path.suffix == ".html"
    assert "variant" not in query_path.name  # Query converted to digest


def test_scraper_downloads_in_scope_pages(tmp_path: Path) -> None:
    base_url = "https://example.com/wiki"
    responses = {
        base_url: FakeResponse(
            """
            <html><body>
                <a href="/wiki/page-one">Page 1</a>
                <a href="https://example.com/wiki/page-two/">Page 2</a>
                <a href="https://example.com/wiki/page-one?variant=1">Variant</a>
                <a href="https://other.com/page">Out of scope</a>
                <a href="mailto:info@example.com">Mail</a>
                <a href="#fragment">Fragment</a>
            </body></html>
            """
        ),
        "https://example.com/wiki/page-one": FakeResponse(
            "<html><body><a href='/wiki/page-three'>Page 3</a></body></html>"
        ),
        "https://example.com/wiki/page-two/": FakeResponse("<html><body>Two</body></html>"),
        "https://example.com/wiki/page-one?variant=1": FakeResponse("<html><body>Variant</body></html>"),
        "https://example.com/wiki/page-three": FakeResponse("<html><body>Three</body></html>"),
    }

    scraper = WikiScraper(
        base_url=base_url,
        output_dir=tmp_path,
        delay=0.0,
        session=FakeSession(responses),
        respect_robots=False,
    )

    pages = scraper.scrape()
    assert pages == 5

    assert (tmp_path / "wiki.html").exists()
    assert (tmp_path / "wiki" / "page-one.html").exists()
    assert (tmp_path / "wiki" / "page-two" / "index.html").exists()
    assert (tmp_path / "wiki" / "page-three.html").exists()

    # Query string variant stored separately using a digest suffix
    variant_files = list((tmp_path / "wiki").glob("page-one_*.html"))
    assert len(variant_files) == 1

    # Ensure out-of-scope URLs were never requested
    assert all("other.com" not in url for url in scraper._session.requested)
