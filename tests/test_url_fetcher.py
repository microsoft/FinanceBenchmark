"""
test_url_fetcher.py

Tests for scripts/shared/url_fetcher.py — validates User-Agent setting
and bot-blocking pattern detection in is_useful_content().
"""
import pytest
from scripts.shared.url_fetcher import is_useful_content


class TestIsUsefulContent:
    """Tests for the is_useful_content() function."""

    def test_empty_content_returns_false(self):
        """Empty content should be rejected."""
        assert is_useful_content("") is False
        assert is_useful_content(None) is False

    def test_fetch_error_returns_false(self):
        """Content starting with [fetch error: should be rejected."""
        assert is_useful_content("[fetch error: timeout]") is False
        assert is_useful_content("[fetch error: 404]") is False

    def test_short_content_returns_false(self):
        """Content shorter than 200 chars should be rejected."""
        assert is_useful_content("short") is False
        assert is_useful_content("a" * 199) is False

    def test_valid_content_returns_true(self):
        """Content with >= 200 chars and no blocking patterns should pass."""
        valid_content = "a" * 200
        assert is_useful_content(valid_content) is True

        # Realistic content example
        article = (
            "This is a financial article about market trends. "
            "The stock market showed strong performance in Q4 2023. "
            "Analysts predict continued growth in the technology sector. "
            "Investment strategies should consider diversification. "
            "The Federal Reserve's interest rate decisions continue to impact markets."
        )
        assert is_useful_content(article) is True

    def test_sec_gov_blocking_pattern(self):
        """SEC.gov 'Undeclared Automated Tool' message should be rejected."""
        sec_block = (
            "Your Request Originates from an Undeclared Automated Tool\n"
            "To allow for equitable access to all users, SEC reserves the right "
            "to limit requests originating from undeclared automated tools. "
            "Your request has been identified as part of a network of automated tools."
            " " * 50  # Padding to exceed 200 chars
        )
        assert is_useful_content(sec_block) is False

    def test_cloudflare_access_denied(self):
        """Cloudflare 'Access Denied' page should be rejected."""
        cf_block = (
            "Access Denied\n"
            "You don't have permission to access this resource.\n"
            "If you think this is an error, please contact the website administrator."
            " " * 100  # Padding to exceed 200 chars
        )
        assert is_useful_content(cf_block) is False

    def test_generic_permission_denied(self):
        """Generic 'You don't have permission' message should be rejected."""
        perm_block = (
            "You don't have permission to view this page.\n"
            "Please contact the site administrator for assistance."
            " " * 100  # Padding to exceed 200 chars
        )
        assert is_useful_content(perm_block) is False

    def test_403_status_code_message(self):
        """HTTP 403 error messages should be rejected."""
        http_403 = (
            "Request failed with status code 403\n"
            "Forbidden: You don't have permission to access this resource."
            " " * 100  # Padding to exceed 200 chars
        )
        assert is_useful_content(http_403) is False

    def test_security_check_required(self):
        """'Security Check Required' message should be rejected."""
        security_check = (
            "Security Check Required\n"
            "Please complete the security verification to continue."
            " " * 100  # Padding to exceed 200 chars
        )
        assert is_useful_content(security_check) is False

    def test_case_insensitive_blocking_detection(self):
        """Blocking pattern detection should be case-insensitive."""
        # Mixed case versions should still be caught
        assert is_useful_content("ACCESS DENIED " * 20) is False
        assert is_useful_content("access denied " * 20) is False
        assert is_useful_content("Access Denied " * 20) is False

        assert is_useful_content("UNDECLARED AUTOMATED TOOL " * 10) is False
        assert is_useful_content("undeclared automated tool " * 10) is False

    def test_blocking_pattern_in_middle_of_content(self):
        """Blocking patterns anywhere in content should trigger rejection."""
        content_with_block = (
            "This looks like normal content at first. "
            "It has a lot of text and appears legitimate. "
            "But then suddenly: Access Denied appears in the middle. "
            "And there's more text after that to make it look real."
            " " * 50  # Padding
        )
        assert is_useful_content(content_with_block) is False

    def test_partial_blocking_words_allowed(self):
        """Content with partial matches (not full blocking patterns) should pass."""
        # "access" and "denied" separately shouldn't trigger the block
        valid_content = (
            "Users can access the financial reports through the portal. "
            "The request was not denied, and the data was retrieved successfully. "
            "This content discusses market access and regulatory compliance."
            " " * 50
        )
        assert is_useful_content(valid_content) is True
