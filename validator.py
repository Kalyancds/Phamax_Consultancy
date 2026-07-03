"""
Statistical validation of Chandra OCR page output.

Chandra schema
--------------
Top-level document::

    {"pages": [...]}

Each page::

    {
        "page_number": <int>,
        "inference_failed": <bool>,
        "blocks": [<block>, ...]
    }

Each block (nested)::

    {
        "type": <str>,
        "content": "<html-string>",
        "children": [<block>, ...],
        ...
    }

Chandra does **not** emit per-token confidence scores; validation is therefore
based on structural and word-count statistics only.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import Any

from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Tuneable constants (overridable at runtime from the Streamlit sidebar)
# ---------------------------------------------------------------------------

#: Absolute word-count floor — pages with fewer words are always flagged.
MIN_WORD_COUNT: int = 10

#: Tukey IQR fence multiplier used for cross-page outlier detection.
IQR_MULTIPLIER: float = 1.5

#: Pages whose word-count z-score falls below this value are flagged.
ZSCORE_THRESHOLD: float = -2.0

#: If the proportion of empty blocks on a page exceeds this, the page is flagged.
EMPTY_BLOCK_RATIO_THRESHOLD: float = 0.5


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PageStats:
    """Per-page statistics and review flags derived from a Chandra OCR page."""

    page_number: int
    inference_failed: bool
    word_count: int
    block_count: int
    empty_block_count: int
    empty_block_ratio: float
    flags: list[str] = field(default_factory=list)
    needs_review: bool = False

    @property
    def non_empty_blocks(self) -> int:
        """Number of blocks that contain at least one word."""
        return self.block_count - self.empty_block_count


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _html_to_text(html: str) -> str:
    """Strip HTML markup and return plain text."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator=" ")


def _count_words(text: str) -> int:
    """Return the number of whitespace-delimited tokens in *text*."""
    return len(re.findall(r"\S+", text))


def _flatten_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Depth-first traversal of a nested block tree; returns every node."""
    result: list[dict[str, Any]] = []
    stack = list(blocks)
    while stack:
        block = stack.pop()
        result.append(block)
        children = block.get("children") or []
        stack.extend(children)
    return result


# ---------------------------------------------------------------------------
# Core validation
# ---------------------------------------------------------------------------


def compute_page_stats(page: dict[str, Any]) -> PageStats:
    """Compute raw statistics for a single Chandra OCR page dict.

    No cross-page information is needed at this stage; flags that require
    document-level context (IQR, z-score) are added later by
    :func:`validate_document`.
    """
    page_number: int = int(page.get("page_number", 0))
    inference_failed: bool = bool(page.get("inference_failed", False))

    blocks = _flatten_blocks(page.get("blocks") or [])
    block_count = len(blocks)

    all_text_parts: list[str] = []
    empty_block_count = 0

    for block in blocks:
        raw_content: str = block.get("content", "") or ""
        text = _html_to_text(raw_content)
        if _count_words(text) == 0:
            empty_block_count += 1
        all_text_parts.append(text)

    word_count = _count_words(" ".join(all_text_parts))
    empty_block_ratio = (empty_block_count / block_count) if block_count > 0 else 1.0

    return PageStats(
        page_number=page_number,
        inference_failed=inference_failed,
        word_count=word_count,
        block_count=block_count,
        empty_block_count=empty_block_count,
        empty_block_ratio=empty_block_ratio,
    )


def _apply_flags(
    stats_list: list[PageStats],
    word_counts: list[int],
) -> None:
    """Attach review flags to each :class:`PageStats` object *in-place*.

    Flags applied
    ~~~~~~~~~~~~~
    * ``inference_failed``           — Chandra set the inference_failed flag
    * ``low_word_count``             — absolute floor (MIN_WORD_COUNT)
    * ``no_blocks``                  — page has zero blocks
    * ``high_empty_block_ratio``     — too many structurally empty blocks
    * ``iqr_low_outlier``            — below the Tukey lower fence (≥4 pages)
    * ``zscore_low``                 — z-score below ZSCORE_THRESHOLD (≥2 pages)
    """
    n = len(word_counts)

    # Document-level: IQR fence (requires at least 4 data points)
    low_fence: float | None = None
    if n >= 4:
        sorted_wc = sorted(word_counts)
        q1 = statistics.quantiles(sorted_wc, n=4)[0]
        q3 = statistics.quantiles(sorted_wc, n=4)[2]
        iqr = q3 - q1
        low_fence = q1 - IQR_MULTIPLIER * iqr

    # Document-level: z-score (requires at least 2 data points)
    mean_wc: float | None = None
    stdev_wc: float | None = None
    if n >= 2:
        mean_wc = statistics.mean(word_counts)
        stdev_wc = statistics.pstdev(word_counts)

    for stats in stats_list:
        flags: list[str] = []

        if stats.inference_failed:
            flags.append("inference_failed")

        if stats.word_count < MIN_WORD_COUNT:
            flags.append(
                f"low_word_count ({stats.word_count} < {MIN_WORD_COUNT})"
            )

        if stats.block_count == 0:
            flags.append("no_blocks")

        if stats.empty_block_ratio > EMPTY_BLOCK_RATIO_THRESHOLD:
            flags.append(
                f"high_empty_block_ratio ({stats.empty_block_ratio:.0%})"
            )

        if low_fence is not None and stats.word_count < low_fence:
            flags.append(f"iqr_low_outlier (fence={low_fence:.1f})")

        if (
            mean_wc is not None
            and stdev_wc is not None
            and stdev_wc > 0
        ):
            z = (stats.word_count - mean_wc) / stdev_wc
            if z < ZSCORE_THRESHOLD:
                flags.append(f"zscore_low ({z:.2f})")

        stats.flags = flags
        stats.needs_review = len(flags) > 0


def validate_document(pages: list[dict[str, Any]]) -> list[PageStats]:
    """Validate all pages of a Chandra OCR document.

    Returns a :class:`PageStats` list in the same order as *pages*, with
    ``flags`` and ``needs_review`` populated.
    """
    stats_list = [compute_page_stats(p) for p in pages]
    word_counts = [s.word_count for s in stats_list]
    _apply_flags(stats_list, word_counts)
    return stats_list
