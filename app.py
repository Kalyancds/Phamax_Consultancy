"""
Streamlit application — Chandra OCR Statistical Validator.

Upload a Chandra OCR JSON document, review per-page statistics and flags,
then download a ZIP review package for every page that needs human attention.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

import validator as _validator
from exporter import export_review_package
from validator import validate_document, PageStats

# ---------------------------------------------------------------------------
# Page config (must be the first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Chandra OCR Validator",
    page_icon="🔍",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar — tuneable thresholds
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Validation settings")
    st.markdown(
        "These thresholds tune when a page is flagged for human review. "
        "Changes take effect immediately."
    )

    min_words = st.number_input(
        "Min word count (absolute floor)",
        min_value=0,
        value=_validator.MIN_WORD_COUNT,
        step=1,
        help="Pages with fewer words than this are always flagged.",
    )
    iqr_mult = st.number_input(
        "IQR multiplier (Tukey fence)",
        min_value=0.0,
        value=_validator.IQR_MULTIPLIER,
        step=0.1,
        format="%.1f",
        help="Lower values make the IQR fence stricter.",
    )
    zscore_thr = st.number_input(
        "Z-score threshold",
        value=_validator.ZSCORE_THRESHOLD,
        step=0.1,
        format="%.1f",
        help="Pages whose word-count z-score is below this are flagged.",
    )
    empty_ratio_thr = st.slider(
        "Empty-block ratio threshold",
        min_value=0.0,
        max_value=1.0,
        value=_validator.EMPTY_BLOCK_RATIO_THRESHOLD,
        step=0.05,
        help="Flag pages where the fraction of empty blocks exceeds this.",
    )

    # Apply to the module so validate_document picks them up
    _validator.MIN_WORD_COUNT = int(min_words)
    _validator.IQR_MULTIPLIER = float(iqr_mult)
    _validator.ZSCORE_THRESHOLD = float(zscore_thr)
    _validator.EMPTY_BLOCK_RATIO_THRESHOLD = float(empty_ratio_thr)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _load_document(uploaded_file) -> tuple[list[dict], str]:
    """Parse a Chandra OCR JSON upload and return (pages, filename)."""
    try:
        raw = json.loads(uploaded_file.read())
    except json.JSONDecodeError as exc:
        st.error(f"Invalid JSON: {exc}")
        st.stop()

    if isinstance(raw, dict):
        pages = raw.get("pages", [])
    elif isinstance(raw, list):
        pages = raw
    else:
        st.error(
            "Unrecognised JSON structure. "
            'Expected {"pages": [...]} or a bare JSON array of page objects.'
        )
        st.stop()

    return pages, uploaded_file.name


def _stats_to_dataframe(stats_list: list[PageStats]) -> pd.DataFrame:
    rows = [
        {
            "Page": s.page_number,
            "Words": s.word_count,
            "Blocks": s.block_count,
            "Empty block %": f"{s.empty_block_ratio:.0%}",
            "Inference failed": s.inference_failed,
            "Flags": ", ".join(s.flags) if s.flags else "—",
            "Needs review": s.needs_review,
        }
        for s in stats_list
    ]
    return pd.DataFrame(rows)


def _highlight_flagged(row: pd.Series) -> list[str]:
    """Row-wise style: red background for pages that need review."""
    color = "background-color: #ffe0e0" if row["Needs review"] else ""
    return [color] * len(row)


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

st.title("🔍 Chandra OCR Statistical Validator")
st.markdown(
    "Upload a **Chandra OCR output JSON** file to statistically validate every page "
    "and surface the ones that need human review. "
    "Adjust the thresholds in the sidebar, then download the **review package** "
    "(ZIP with JSON + HTML previews) for all flagged pages."
)

uploaded = st.file_uploader(
    "Upload Chandra OCR JSON",
    type=["json"],
    help='Accepts {"pages": [...]} or a bare JSON array of page objects.',
)

if uploaded is None:
    st.info("👆 Upload a Chandra OCR JSON file to get started.")
    st.stop()

# Parse and validate
pages, filename = _load_document(uploaded)

if not pages:
    st.warning("No pages found in the uploaded file.")
    st.stop()

stats_list = validate_document(pages)
flagged = [s for s in stats_list if s.needs_review]

# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total pages", len(pages))
col2.metric("Pages flagged", len(flagged), delta_color="inverse")
col3.metric(
    "Pass rate",
    f"{(1 - len(flagged) / max(len(pages), 1)) * 100:.1f}%",
)
col4.metric(
    "inference_failed",
    sum(1 for s in stats_list if s.inference_failed),
    delta_color="inverse",
)

st.divider()

# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

df = _stats_to_dataframe(stats_list)

tab_all, tab_flagged = st.tabs(["📋 All pages", "⚠️ Flagged pages"])

with tab_all:
    st.dataframe(
        df.style.apply(_highlight_flagged, axis=1),
        use_container_width=True,
        hide_index=True,
    )

with tab_flagged:
    if flagged:
        st.warning(
            f"{len(flagged)} page(s) require human review — "
            "see flag details below."
        )
        flagged_df = df[df["Needs review"]].drop(columns=["Needs review"])
        st.dataframe(flagged_df, use_container_width=True, hide_index=True)
    else:
        st.success("🎉 All pages passed validation — no review needed!")

# ---------------------------------------------------------------------------
# Export review package
# ---------------------------------------------------------------------------

st.divider()
st.subheader("📦 Export review package")

if flagged:
    st.markdown(
        f"The export contains `review_summary.json` plus `pages/<n>.json` and "
        f"`pages/<n>.html` for each of the **{len(flagged)} flagged page(s)**."
    )
    zip_bytes = export_review_package(
        all_pages=pages,
        stats_list=stats_list,
        source_filename=filename,
    )
    st.download_button(
        label=f"⬇️ Download review package ({len(flagged)} pages)",
        data=zip_bytes,
        file_name=f"{Path(filename).stem}_review.zip",
        mime="application/zip",
    )
else:
    st.success("All pages passed — no review package needed.")
