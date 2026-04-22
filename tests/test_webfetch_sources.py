"""
test_webfetch_sources.py

Unit tests for WebFetch source extraction in Claude inference.
"""
import pytest
from scripts.shared.url_fetcher import is_useful_content


class TestIsUsefulContent:
    """Tests for is_useful_content function."""

    def test_rejects_empty_string(self):
        """Empty string is not useful content."""
        assert not is_useful_content("")

    def test_rejects_fetch_errors(self):
        """Fetch errors are not useful content."""
        assert not is_useful_content("[fetch error: timeout]")
        assert not is_useful_content("[fetch error: connection refused]")

    def test_rejects_short_content(self):
        """Content shorter than 200 chars is not useful."""
        short_text = "This is a short text."
        assert len(short_text) < 200
        assert not is_useful_content(short_text)

    def test_accepts_long_content(self):
        """Content with 200+ characters is useful."""
        long_text = "A" * 200
        assert is_useful_content(long_text)

    def test_accepts_realistic_content(self):
        """Realistic page content is accepted."""
        content = """
        This is a typical page content from a financial report.
        It contains information about revenue, expenses, and net income.
        The company reported revenue of $10 million in Q4 2024.
        Operating expenses were $6 million, resulting in net income of $4 million.
        The board approved a dividend of $0.50 per share.
        """
        assert is_useful_content(content)


class TestWebFetchSourceExtraction:
    """Tests for source extraction from tool_calls_log."""

    def test_extracts_webfetch_sources(self):
        """Successfully extracts sources from WebFetch tool calls."""
        tool_calls_log = [
            {
                "tool": "WebFetch",
                "input": {"url": "https://example.com/page1", "prompt": "Extract info"},
                "output": "A" * 300,  # Sufficient content
                "success": True,
            },
            {
                "tool": "WebSearch",
                "input": {"query": "test"},
                "output": "Search results",
                "success": True,
            },
        ]

        sources = {}
        for tc in tool_calls_log:
            if tc.get("tool") == "WebFetch" and tc.get("success"):
                url = tc.get("input", {}).get("url", "")
                content = tc.get("output", "")
                if url and is_useful_content(content):
                    sources[url] = content

        assert len(sources) == 1
        assert "https://example.com/page1" in sources
        assert sources["https://example.com/page1"] == "A" * 300

    def test_skips_failed_webfetch(self):
        """Failed WebFetch calls are not included in sources."""
        tool_calls_log = [
            {
                "tool": "WebFetch",
                "input": {"url": "https://example.com/fail"},
                "output": "[fetch error: timeout]",
                "success": False,
            },
        ]

        sources = {}
        for tc in tool_calls_log:
            if tc.get("tool") == "WebFetch" and tc.get("success"):
                url = tc.get("input", {}).get("url", "")
                content = tc.get("output", "")
                if url and is_useful_content(content):
                    sources[url] = content

        assert len(sources) == 0

    def test_skips_short_content(self):
        """WebFetch with short content is not included."""
        tool_calls_log = [
            {
                "tool": "WebFetch",
                "input": {"url": "https://example.com/short"},
                "output": "Too short",
                "success": True,
            },
        ]

        sources = {}
        for tc in tool_calls_log:
            if tc.get("tool") == "WebFetch" and tc.get("success"):
                url = tc.get("input", {}).get("url", "")
                content = tc.get("output", "")
                if url and is_useful_content(content):
                    sources[url] = content

        assert len(sources) == 0

    def test_handles_multiple_webfetch_calls(self):
        """Multiple WebFetch calls are all included."""
        tool_calls_log = [
            {
                "tool": "WebFetch",
                "input": {"url": "https://example.com/page1"},
                "output": "A" * 300,
                "success": True,
            },
            {
                "tool": "WebFetch",
                "input": {"url": "https://example.com/page2"},
                "output": "B" * 300,
                "success": True,
            },
            {
                "tool": "WebSearch",
                "input": {"query": "test"},
                "output": "Search results",
                "success": True,
            },
        ]

        sources = {}
        for tc in tool_calls_log:
            if tc.get("tool") == "WebFetch" and tc.get("success"):
                url = tc.get("input", {}).get("url", "")
                content = tc.get("output", "")
                if url and is_useful_content(content):
                    sources[url] = content

        assert len(sources) == 2
        assert "https://example.com/page1" in sources
        assert "https://example.com/page2" in sources

    def test_handles_missing_url(self):
        """WebFetch without URL is skipped."""
        tool_calls_log = [
            {
                "tool": "WebFetch",
                "input": {},  # No URL
                "output": "A" * 300,
                "success": True,
            },
        ]

        sources = {}
        for tc in tool_calls_log:
            if tc.get("tool") == "WebFetch" and tc.get("success"):
                url = tc.get("input", {}).get("url", "")
                content = tc.get("output", "")
                if url and is_useful_content(content):
                    sources[url] = content

        assert len(sources) == 0
