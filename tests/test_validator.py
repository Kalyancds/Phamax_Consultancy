"""
Unit tests for validator.py — Chandra OCR statistical validation.

These tests cover:
* compute_page_stats   — per-page stat extraction
* validate_document    — cross-page flagging (IQR, z-score, absolute rules)
* edge cases           — empty pages, deeply nested blocks, inference_failed
"""

from __future__ import annotations

import pytest

import validator as v
from validator import (
    PageStats,
    compute_page_stats,
    validate_document,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def make_page(
    page_number: int = 1,
    inference_failed: bool = False,
    blocks: list | None = None,
) -> dict:
    """Return a minimal Chandra page dict."""
    return {
        "page_number": page_number,
        "inference_failed": inference_failed,
        "blocks": blocks or [],
    }


def text_block(content: str, children: list | None = None) -> dict:
    """Return a Chandra text block dict."""
    return {"type": "paragraph", "content": content, "children": children or []}


# ---------------------------------------------------------------------------
# compute_page_stats
# ---------------------------------------------------------------------------


class TestComputePageStats:
    def test_empty_page_has_zero_words_and_no_blocks(self):
        page = make_page(page_number=3)
        stats = compute_page_stats(page)
        assert stats.page_number == 3
        assert stats.word_count == 0
        assert stats.block_count == 0
        assert stats.empty_block_count == 0
        assert stats.empty_block_ratio == 1.0  # no blocks → 100 % empty

    def test_inference_failed_is_captured(self):
        page = make_page(inference_failed=True)
        stats = compute_page_stats(page)
        assert stats.inference_failed is True

    def test_word_count_from_plain_html(self):
        page = make_page(
            blocks=[text_block("<p>Hello world foo bar</p>")]
        )
        stats = compute_page_stats(page)
        assert stats.word_count == 4

    def test_html_tags_are_stripped(self):
        page = make_page(
            blocks=[text_block("<div><p>one</p><span>two three</span></div>")]
        )
        stats = compute_page_stats(page)
        assert stats.word_count == 3

    def test_empty_block_counted(self):
        page = make_page(
            blocks=[
                text_block("<p>real content here</p>"),
                text_block(""),
                text_block("   "),
            ]
        )
        stats = compute_page_stats(page)
        assert stats.block_count == 3
        assert stats.empty_block_count == 2
        assert abs(stats.empty_block_ratio - 2 / 3) < 1e-9

    def test_nested_blocks_flattened(self):
        child = text_block("<span>child text</span>")
        parent = text_block("<p>parent text</p>", children=[child])
        page = make_page(blocks=[parent])
        stats = compute_page_stats(page)
        # Both parent and child are counted
        assert stats.block_count == 2
        assert stats.word_count == 4  # "parent text child text"

    def test_deeply_nested_blocks(self):
        deep = text_block("<em>deep</em>")
        mid = text_block("<em>mid</em>", children=[deep])
        top = text_block("<em>top</em>", children=[mid])
        page = make_page(blocks=[top])
        stats = compute_page_stats(page)
        assert stats.block_count == 3
        assert stats.word_count == 3

    def test_none_content_treated_as_empty(self):
        block = {"type": "unknown", "content": None, "children": []}
        page = make_page(blocks=[block])
        stats = compute_page_stats(page)
        assert stats.word_count == 0
        assert stats.empty_block_count == 1

    def test_missing_blocks_key(self):
        page = {"page_number": 5, "inference_failed": False}
        stats = compute_page_stats(page)
        assert stats.block_count == 0


# ---------------------------------------------------------------------------
# validate_document — flag rules
# ---------------------------------------------------------------------------


class TestValidateDocumentFlags:
    def _reset_thresholds(self):
        v.MIN_WORD_COUNT = 10
        v.IQR_MULTIPLIER = 1.5
        v.ZSCORE_THRESHOLD = -2.0
        v.EMPTY_BLOCK_RATIO_THRESHOLD = 0.5

    def setup_method(self):
        self._reset_thresholds()

    def test_inference_failed_always_flagged(self):
        pages = [
            make_page(1, inference_failed=True, blocks=[text_block("<p>" + "word " * 50 + "</p>")]),
        ]
        result = validate_document(pages)
        assert result[0].needs_review is True
        assert any("inference_failed" in f for f in result[0].flags)

    def test_low_word_count_flagged(self):
        v.MIN_WORD_COUNT = 20
        page = make_page(1, blocks=[text_block("<p>only five words here</p>")])
        result = validate_document([page])
        assert result[0].needs_review is True
        assert any("low_word_count" in f for f in result[0].flags)

    def test_no_blocks_flagged(self):
        page = make_page(1, blocks=[])
        result = validate_document([page])
        assert result[0].needs_review is True
        assert any("no_blocks" in f for f in result[0].flags)

    def test_high_empty_block_ratio_flagged(self):
        v.EMPTY_BLOCK_RATIO_THRESHOLD = 0.5
        page = make_page(
            1,
            blocks=[text_block(""), text_block(""), text_block("<p>one word</p>")],
        )
        result = validate_document([page])
        assert result[0].needs_review is True
        assert any("high_empty_block_ratio" in f for f in result[0].flags)

    def test_good_page_not_flagged(self):
        v.MIN_WORD_COUNT = 5
        page = make_page(
            1,
            blocks=[text_block("<p>" + "word " * 20 + "</p>")],
        )
        result = validate_document([page])
        assert result[0].needs_review is False
        assert result[0].flags == []

    def test_iqr_outlier_flagged(self):
        """A page with far fewer words than the rest should be IQR-flagged."""
        v.MIN_WORD_COUNT = 0
        normal_words = "<p>" + "word " * 100 + "</p>"
        pages = [
            make_page(i + 1, blocks=[text_block(normal_words)])
            for i in range(7)
        ]
        # Page 8 is a severe low outlier
        pages.append(make_page(8, blocks=[text_block("<p>one</p>")]))

        result = validate_document(pages)
        outlier_stats = result[7]
        assert outlier_stats.page_number == 8
        assert any("iqr_low_outlier" in f for f in outlier_stats.flags)

    def test_zscore_outlier_flagged(self):
        """A page drastically below the mean should trigger the z-score flag."""
        v.MIN_WORD_COUNT = 0
        v.ZSCORE_THRESHOLD = -1.5
        normal_words = "<p>" + "word " * 100 + "</p>"
        pages = [
            make_page(i + 1, blocks=[text_block(normal_words)])
            for i in range(5)
        ]
        pages.append(make_page(6, blocks=[text_block("<p>x</p>")]))

        result = validate_document(pages)
        outlier_stats = result[5]
        assert any("zscore_low" in f for f in outlier_stats.flags)

    def test_no_iqr_flag_with_fewer_than_four_pages(self):
        """IQR detection is silenced for documents with fewer than 4 pages."""
        v.MIN_WORD_COUNT = 0
        pages = [
            make_page(1, blocks=[text_block("<p>" + "w " * 100 + "</p>")]),
            make_page(2, blocks=[text_block("<p>x</p>")]),
            make_page(3, blocks=[text_block("<p>" + "w " * 100 + "</p>")]),
        ]
        result = validate_document(pages)
        # Page 2 should NOT get an IQR flag (only 3 pages)
        flags_p2 = result[1].flags
        assert not any("iqr_low_outlier" in f for f in flags_p2)

    def test_multiple_flags_can_be_set(self):
        """A page can accumulate more than one flag."""
        v.MIN_WORD_COUNT = 50
        page = make_page(1, inference_failed=True, blocks=[])
        result = validate_document([page])
        flags = result[0].flags
        assert any("inference_failed" in f for f in flags)
        assert any("no_blocks" in f for f in flags)
        assert any("low_word_count" in f for f in flags)


# ---------------------------------------------------------------------------
# validate_document — ordering and completeness
# ---------------------------------------------------------------------------


class TestValidateDocumentOrdering:
    def test_result_length_matches_input(self):
        pages = [make_page(i + 1) for i in range(10)]
        result = validate_document(pages)
        assert len(result) == 10

    def test_result_order_matches_input(self):
        pages = [make_page(i + 1) for i in range(5)]
        result = validate_document(pages)
        for i, stats in enumerate(result):
            assert stats.page_number == i + 1

    def test_empty_document(self):
        result = validate_document([])
        assert result == []


# ---------------------------------------------------------------------------
# exporter — smoke test
# ---------------------------------------------------------------------------


class TestExporter:
    def test_export_returns_bytes(self):
        from exporter import export_review_package

        pages = [
            make_page(1, inference_failed=True, blocks=[text_block("<p>hello</p>")]),
            make_page(2, blocks=[text_block("<p>" + "w " * 50 + "</p>")]),
        ]
        stats_list = validate_document(pages)
        result = export_review_package(pages, stats_list, source_filename="test.json")
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_export_is_valid_zip(self):
        import io
        import zipfile

        from exporter import export_review_package

        import validator as v
        v.MIN_WORD_COUNT = 0

        pages = [
            make_page(1, inference_failed=True, blocks=[text_block("<p>text</p>")]),
        ]
        stats_list = validate_document(pages)
        zip_bytes = export_review_package(pages, stats_list)

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = zf.namelist()
        assert "review_summary.json" in names
        assert "pages/1.json" in names
        assert "pages/1.html" in names

    def test_export_summary_contents(self):
        import io
        import json
        import zipfile

        from exporter import export_review_package

        import validator as v
        v.MIN_WORD_COUNT = 0

        pages = [
            make_page(1, inference_failed=True, blocks=[text_block("<p>hi</p>")]),
            make_page(2, blocks=[text_block("<p>" + "w " * 50 + "</p>")]),
        ]
        stats_list = validate_document(pages)
        zip_bytes = export_review_package(pages, stats_list, source_filename="doc.json")

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            summary = json.loads(zf.read("review_summary.json"))

        assert summary["source"] == "doc.json"
        assert summary["total_pages"] == 2
        # Page 1 (inference_failed) must appear in the summary
        flagged_nums = [p["page_number"] for p in summary["pages"]]
        assert 1 in flagged_nums

    def test_no_flagged_pages_empty_zip_entries(self):
        import io
        import zipfile

        from exporter import export_review_package

        import validator as v
        v.MIN_WORD_COUNT = 0
        v.EMPTY_BLOCK_RATIO_THRESHOLD = 1.1  # nothing flagged by ratio

        pages = [
            make_page(1, blocks=[text_block("<p>" + "w " * 50 + "</p>")]),
        ]
        stats_list = validate_document(pages)
        # Ensure no flags
        assert not stats_list[0].needs_review

        zip_bytes = export_review_package(pages, stats_list)
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = zf.namelist()
        assert "review_summary.json" in names
        # No page entries when nothing is flagged
        assert not any(n.startswith("pages/") for n in names)
