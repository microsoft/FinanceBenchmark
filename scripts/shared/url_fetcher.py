"""
url_fetcher.py

Shared URL fetching utilities for inference-time source capture and
eval-time groundedness checking.

Exports:
    fetch_url(url, cache_dir, timeout) -> str
        Synchronous Playwright fetch with disk cache. Thread-safe.

    is_useful_content(content) -> bool
        Returns True if content is substantive enough for grounding evaluation.

    async_fetch_url_list(urls, cache_dir, timeout, max_workers) -> dict[str, str]
        Async: fetches a pre-built list of URLs concurrently, returns
        {url: content} for URLs with useful content.

    async_fetch_sources(answer, cache_dir, timeout, max_workers) -> dict[str, str]
        Async: extracts all cited URLs from an answer's markdown hyperlinks, fetches
        them with up to max_workers concurrent Playwright instances, returns
        {url: content} for URLs with useful content.
"""

import asyncio
import hashlib
import io
import re
import threading
import urllib.request
from pathlib import Path

import fitz  # pymupdf
from playwright.sync_api import sync_playwright

_url_cache_lock = threading.Lock()

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


def _extract_pdf_text(url: str, timeout: int) -> str:
    """Download a PDF by URL and extract its text with PyMuPDF."""
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        pdf_bytes = resp.read()
    doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
    return "\n".join(page.get_text() for page in doc)


def fetch_url(url: str, cache_dir: Path, timeout: int = 30) -> str:
    """Fetch a URL and return its text content, with disk caching.

    Uses Playwright to navigate to the URL and checks the Content-Type of the
    navigation response. PDFs (detected by Content-Type regardless of URL extension)
    are re-fetched with urllib and extracted with PyMuPDF. HTML pages use
    Playwright's inner_text.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key  = hashlib.sha256(url.encode()).hexdigest()
    cache_file = cache_dir / f"{cache_key}.txt"

    with _url_cache_lock:
        if cache_file.exists():
            return cache_file.read_text(encoding="utf-8", errors="replace")

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(user_agent=_USER_AGENT)
            page = context.new_page()
            # wait_until="commit" stops after receiving response headers — enough to
            # detect Content-Type without rendering the full page or downloading a PDF.
            response = page.goto(url, wait_until="commit", timeout=timeout * 1000)
            ct = response.headers.get("content-type", "").lower() if response else ""
            if "pdf" in ct:
                browser.close()
                text = _extract_pdf_text(url, timeout)
            else:
                page.wait_for_load_state("networkidle", timeout=timeout * 1000)
                text = page.inner_text("body")
                browser.close()
    except Exception as exc:
        return f"[fetch error: {exc}]"

    if text.strip():
        with _url_cache_lock:
            cache_file.write_text(text, encoding="utf-8")
    return text


def is_useful_content(content: str) -> bool:
    """Return True if content contains enough text to be useful for grounding evaluation.

    Filters out:
    - Empty content
    - Fetch errors
    - Content shorter than 200 chars
    - Common bot-blocking patterns (SEC.gov, Cloudflare, generic access denied)
    """
    if not content or content.startswith("[fetch error:") or len(content.strip()) < 200:
        return False

    # Detect common blocking patterns
    blocking_patterns = [
        "Undeclared Automated Tool",           # SEC.gov
        "Access Denied",                        # Cloudflare / generic
        "You don't have permission",            # Generic access block
        "Request failed with status code 403",  # HTTP 403 errors
        "Security Check Required",              # Additional bot detection
    ]

    content_lower = content.lower()
    for pattern in blocking_patterns:
        if pattern.lower() in content_lower:
            return False

    return True


async def async_fetch_url_list(
    urls: list[str],
    cache_dir: Path,
    timeout: int = 30,
    max_workers: int = 5,
) -> dict[str, str]:
    """Fetch a pre-built list of URLs concurrently.

    Deduplicates the input list, then fetches in parallel using a semaphore to
    cap concurrent Playwright instances at max_workers.
    Returns a dict mapping URL → page content for URLs with useful content.
    """
    seen: set = set()
    unique_urls = [u for u in urls if not (u in seen or seen.add(u))]

    sem = asyncio.Semaphore(max_workers)

    async def fetch_one(url: str) -> tuple[str, str]:
        async with sem:
            content = await asyncio.to_thread(fetch_url, url, cache_dir, timeout)
        return url, content

    results = await asyncio.gather(*[fetch_one(u) for u in unique_urls])
    return {url: content for url, content in results if is_useful_content(content)}


async def async_fetch_sources(
    answer: str,
    cache_dir: Path,
    timeout: int = 30,
    max_workers: int = 5,
) -> dict[str, str]:
    """Fetch all cited URLs from an answer's markdown hyperlinks concurrently.

    Extracts all unique URLs from markdown hyperlinks and fetches them in parallel,
    using a semaphore to cap concurrent Playwright instances at max_workers.
    Returns a dict mapping URL → page content for URLs with useful content.
    """
    raw_urls = re.findall(r'\[(?:[^\]]+)\]\((https?://[^\)]+)\)', answer)
    return await async_fetch_url_list(raw_urls, cache_dir, timeout, max_workers)
