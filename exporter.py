"""
Export a human-review package (ZIP archive) for flagged Chandra OCR pages.

The archive contains:
* ``review_summary.json``  — JSON index of all flagged pages and their flags.
* ``pages/<n>.json``       — raw Chandra page JSON for each flagged page.
* ``pages/<n>.html``       — rendered HTML preview for each flagged page.
"""

from __future__ import annotations

import io
import json
import zipfile
from typing import Any

from validator import PageStats


# ---------------------------------------------------------------------------
# HTML preview builder
# ---------------------------------------------------------------------------

_HTML_TEMPLATE_HEAD = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Page {page_number} — Chandra OCR Review</title>
  <style>
    body  {{ font-family: sans-serif; margin: 2em; color: #222; }}
    h1    {{ border-bottom: 2px solid #333; padding-bottom: .25em; }}
    .warn {{ color: #c00; font-weight: bold; }}
    .flag {{ display: inline-block; background: #ffe082; border-radius: 4px;
             padding: 2px 6px; margin: 2px; font-size: .85em; }}
    .block {{ border: 1px solid #ccc; border-radius: 4px;
              padding: 8px 12px; margin: 6px 0; }}
    .block-type {{ font-size: .8em; color: #666; }}
    .empty-block {{ background: #fafafa; color: #aaa; }}
  </style>
</head>
<body>
"""


def _render_blocks_html(blocks: list[dict[str, Any]], depth: int = 0) -> str:
    """Recursively render a list of Chandra blocks as HTML fragments."""
    parts: list[str] = []
    indent_px = depth * 24
    for block in blocks:
        btype = block.get("type", "unknown")
        content: str = block.get("content", "") or ""
        is_empty = not content.strip()
        cls = "block empty-block" if is_empty else "block"
        parts.append(
            f'<div class="{cls}" style="margin-left:{indent_px}px">'
            f'<span class="block-type">[{btype}]</span> '
            f'{content if content else "<em>(empty)</em>"}'
        )
        children = block.get("children") or []
        if children:
            parts.append(_render_blocks_html(children, depth + 1))
        parts.append("</div>")
    return "\n".join(parts)


def build_html_preview(
    page: dict[str, Any],
    stats: PageStats | None = None,
) -> str:
    """Return a self-contained HTML page previewing one Chandra OCR page.

    Parameters
    ----------
    page:
        Raw Chandra page dict.
    stats:
        Optional :class:`~validator.PageStats`; when supplied the flag
        badges are shown at the top of the preview.
    """
    page_number = page.get("page_number", "?")
    inference_failed = bool(page.get("inference_failed", False))
    blocks = page.get("blocks") or []

    lines: list[str] = [_HTML_TEMPLATE_HEAD.format(page_number=page_number)]
    lines.append(f"<h1>Page {page_number}</h1>")

    if inference_failed:
        lines.append('<p class="warn">⚠ inference_failed = true</p>')

    if stats and stats.flags:
        flag_html = " ".join(f'<span class="flag">{f}</span>' for f in stats.flags)
        lines.append(f"<p>Flags: {flag_html}</p>")

    if stats:
        lines.append(
            f"<p>Words: <strong>{stats.word_count}</strong> &nbsp;|&nbsp; "
            f"Blocks: <strong>{stats.block_count}</strong> &nbsp;|&nbsp; "
            f"Empty-block ratio: <strong>{stats.empty_block_ratio:.0%}</strong></p>"
        )

    lines.append(_render_blocks_html(blocks))
    lines.append("</body></html>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ZIP package builder
# ---------------------------------------------------------------------------


def export_review_package(
    all_pages: list[dict[str, Any]],
    stats_list: list[PageStats],
    source_filename: str = "document",
) -> bytes:
    """Build an in-memory ZIP review package for every flagged page.

    Parameters
    ----------
    all_pages:
        The complete list of raw Chandra page dicts (as parsed from the
        uploaded JSON).
    stats_list:
        Validated :class:`~validator.PageStats` list, parallel to *all_pages*.
    source_filename:
        Original filename; embedded in the summary JSON for traceability.

    Returns
    -------
    bytes
        Raw ZIP archive content suitable for :func:`streamlit.download_button`.
    """
    flagged = [s for s in stats_list if s.needs_review]
    stats_by_page: dict[int, PageStats] = {s.page_number: s for s in stats_list}
    page_by_number: dict[int, dict[str, Any]] = {
        p.get("page_number"): p for p in all_pages  # type: ignore[misc]
    }

    summary: dict[str, Any] = {
        "source": source_filename,
        "total_pages": len(all_pages),
        "flagged_pages": len(flagged),
        "pages": [
            {
                "page_number": s.page_number,
                "inference_failed": s.inference_failed,
                "word_count": s.word_count,
                "block_count": s.block_count,
                "empty_block_ratio": round(s.empty_block_ratio, 4),
                "flags": s.flags,
            }
            for s in flagged
        ],
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("review_summary.json", json.dumps(summary, indent=2))
        for stats in flagged:
            pn = stats.page_number
            page = page_by_number.get(pn, {"page_number": pn})
            page_stats = stats_by_page.get(pn)
            zf.writestr(f"pages/{pn}.json", json.dumps(page, indent=2))
            zf.writestr(
                f"pages/{pn}.html",
                build_html_preview(page, stats=page_stats),
            )

    buf.seek(0)
    return buf.read()
