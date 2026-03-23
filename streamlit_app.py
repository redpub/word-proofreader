from __future__ import annotations

import json
from dataclasses import asdict
from typing import List, Dict, Any

import streamlit as st
from docx import Document

from changes_parser import (
    parse_json,
    parse_changes,
    validate_ranges,
    extract_preview_segments,
    revalidate_subset,
    Change,
)
from docx_apply import apply_changes_to_docx, ResolvedChange


# ---------- Helper: text extraction must mirror docx_apply linearization ----------

def extract_full_text(doc_bytes: bytes) -> str:
    """
    Extract plain text EXACTLY matching docx-revisions paragraph traversal.
    Critical: must use identical paragraph.text + '\n' joining.
    """
    from io import BytesIO
    doc = Document(BytesIO(doc_bytes))
    paragraphs = [p.text or "" for p in doc.paragraphs]
    return "\n".join(paragraphs)



# ---------- LLM prompt template ----------

LLM_PROMPT_TEMPLATE = """You are given the full plain text of a Word document and must propose tracked changes.

Output JSON ONLY, matching exactly this schema:

{
  "changes": [
    {
      "type": "insert" | "delete" | "replace" | "comment",
      "author": "string",
      "timestamp": "ISO-8601 string (optional)",
      "position": number,        // insert only
      "start": number,           // delete/replace/comment
      "end": number,             // delete/replace/comment
      "text": "string"           // inserted text, replacement text, or comment text
    }
  ]
}

Rules:

- All indexes are 0-based character offsets into the raw plain text provided to you (not lines, not words).
- For "insert":
  - Use "position" only.
  - Do NOT include "start" or "end".
- For "delete" and "replace":
  - Use "start" and "end" only.
  - Do NOT include "position".
  - The range [start, end) must correspond exactly to the characters you want to delete/replace.
- For "comment":
  - Use "start" and "end" only to indicate the span of text being commented on.
  - Do NOT modify text.
- Replace = delete + insert, but represented as a single JSON object with type "replace":
  - The characters in [start, end) are replaced by "text".
- All ranges must be non-overlapping:
  - For any two non-insert changes, their [start, end) ranges must NOT overlap at all.
- All ranges must satisfy: 0 ≤ start < end ≤ document_length.
- Insert positions must satisfy: 0 ≤ position ≤ document_length.
- Do NOT make formatting-only changes (bold, italic, etc.).
- Do NOT rewrite the entire document.
- Only propose specific, localized edits.
- Ensure the output is valid JSON and contains only the JSON object, no explanations or comments.

Respond with JSON only.
"""


# ---------- Streamlit UI ----------

st.set_page_config(
    page_title="Tracked Changes JSON Validator",
    layout="wide",
)

st.title("Word Tracked Changes Validator & Applier")

col_left, col_right = st.columns([2, 1])

with st.sidebar:
    st.header("LLM Prompt Template")
    st.write("Use this prompt with your LLM to generate valid JSON edits.")
    st.code(LLM_PROMPT_TEMPLATE, language="text")
    st.button(
        "Copy Prompt to Clipboard",
        help="Use your browser's copy from the code block above.",
    )

with col_right:
    st.subheader("Instructions")
    st.markdown(
        """
1. Upload a **.docx** file and a **JSON** file containing proposed changes.
2. Click **Validate** to perform a dry run.
3. Review changes in the **Preview** section.
4. Use checkboxes to select which changes to apply.
5. Click **Apply Selected Changes** to generate a new .docx with real tracked changes.
        """
    )

with col_left:
    st.subheader("Uploads")

    docx_file = st.file_uploader("Upload Word document (.docx)", type=["docx"])
    json_file = st.file_uploader("Upload changes JSON", type=["json"])

    validate_button = st.button("Validate JSON & Document", type="primary")

# State keys: validated_changes, doc_text, doc_bytes, validation_errors, selected_indices
if "validated_changes" not in st.session_state:
    st.session_state.validated_changes: List[Change] | None = None
if "doc_text" not in st.session_state:
    st.session_state.doc_text: str | None = None
if "doc_bytes" not in st.session_state:
    st.session_state.doc_bytes: bytes | None = None
if "validation_errors" not in st.session_state:
    st.session_state.validation_errors: List[str] | None = None
if "selected_indices" not in st.session_state:
    st.session_state.selected_indices: Dict[int, bool] = {}

# ---------- Validation workflow ----------

if validate_button:
    st.session_state.validation_errors = None
    st.session_state.validated_changes = None
    st.session_state.doc_text = None
    st.session_state.doc_bytes = None
    st.session_state.selected_indices = {}

    if not docx_file or not json_file:
        st.error("Please upload both a .docx file and a JSON file before validating.")
    else:
        doc_bytes = docx_file.read()
        st.session_state.doc_bytes = doc_bytes
        doc_text = extract_full_text(doc_bytes)
        st.session_state.doc_text = doc_text
        doc_length = len(doc_text)

        try:
            data = parse_json(json_file.read())
            changes = parse_changes(data)
            validate_ranges(changes, doc_length)
            st.session_state.validated_changes = changes
            st.success(f"Validation successful. Document length: {doc_length} characters. {len(changes)} changes loaded.")
            # Default: all selected
            st.session_state.selected_indices = {ch.index: True for ch in changes}
        except Exception as e:
            errors: List[str]
            msg = str(e)
            if hasattr(e, "errors"):
                errors = getattr(e, "errors")
            else:
                errors = [msg]
            st.session_state.validation_errors = errors
            st.error("Validation failed. See details below.")


# ---------- Show validation messages ----------

if st.session_state.validation_errors:
    with st.expander("Validation Errors", expanded=True):
        for err in st.session_state.validation_errors:
            st.markdown(f"- {err}")

# ---------- Preview & selection ----------

if st.session_state.validated_changes and st.session_state.doc_text is not None:
    st.subheader("Preview Changes")

    doc_text = st.session_state.doc_text
    changes: List[Change] = st.session_state.validated_changes

    # Sort by original index for display
    changes_sorted = sorted(changes, key=lambda c: c.index)

    for ch in changes_sorted:
        seg = extract_preview_segments(doc_text, ch)
        label = ch.display_label()
        checked = st.session_state.selected_indices.get(ch.index, True)

        # Color / style per type
        if ch.type == "insert":
            card_color = "#e6ffed"  # light green
        elif ch.type == "delete":
            card_color = "#ffe6e6"  # light red
        elif ch.type == "replace":
            card_color = "#e6f0ff"  # light blue
        else:  # comment
            card_color = "#fff9e6"  # light yellow

        with st.container():
            st.checkbox(
                f"Apply {label}",
                value=checked,
                key=f"chk_{ch.index}",
                on_change=lambda idx=ch.index: None,
            )
            st.markdown(
                f"""
<div style="border:1px solid #ccc; border-radius:4px; padding:10px; background-color:{card_color};">
<b>Type:</b> {ch.type} &nbsp;&nbsp;
<b>Author:</b> {ch.author} &nbsp;&nbsp;
<b>Timestamp:</b> {ch.timestamp or "N/A"}<br/>
""",
                unsafe_allow_html=True,
            )

            if ch.type == "insert":
                st.markdown(
                    f"""
<b>Position:</b> {ch.position}<br/>
<b>Context:</b><br/>
<code>{seg["before"]}</code><span style="background-color:#a6f3a6;">{ch.text}</span><code>{seg["after"]}</code>
</div>
""",
                    unsafe_allow_html=True,
                )
            elif ch.type == "delete":
                st.markdown(
                    f"""
<b>Range:</b> [{ch.start}, {ch.end})<br/>
<b>Original text:</b><br/>
<code>{seg["before"]}</code><span style="text-decoration:line-through; background-color:#ffb3b3;">{seg["original"]}</span><code>{seg["after"]}</code>
</div>
""",
                    unsafe_allow_html=True,
                )
            elif ch.type == "replace":
                st.markdown(
                    f"""
<b>Range:</b> [{ch.start}, {ch.end})<br/>
<b>Original text:</b><br/>
<code>{seg["before"]}</code><span style="text-decoration:line-through; background-color:#cce0ff;">{seg["original"]}</span><code>{seg["after"]}</code><br/><br/>
<b>Replacement text:</b><br/>
<span style="background-color:#a6f3a6;">{ch.text}</span>
</div>
""",
                    unsafe_allow_html=True,
                )
            else:  # comment
                st.markdown(
                    f"""
<b>Range:</b> [{ch.start}, {ch.end})<br/>
<b>Original text:</b><br/>
<code>{seg["before"]}</code><span style="background-color:#fff2b3;">{seg["original"]}</span><code>{seg["after"]}</code><br/><br/>
<b>Comment:</b><br/>
<span style="display:inline-block; border:1px solid #ccc; border-radius:4px; padding:6px; background-color:#fffaf0;">{ch.text}</span>
</div>
""",
                    unsafe_allow_html=True,
                )

    # Sync selection back to state
    for ch in changes_sorted:
        st.session_state.selected_indices[ch.index] = st.session_state.get(f"chk_{ch.index}", True)

    st.markdown("---")

    # ---------- Apply selected changes ----------

    apply_button = st.button("Apply Selected Changes and Generate .docx", type="primary")

    if apply_button:
        selected_changes: List[Change] = [
            ch
            for ch in changes_sorted
            if st.session_state.selected_indices.get(ch.index, False)
        ]
        if not selected_changes:
            st.warning("No changes selected. Please select at least one change.")
        else:
            doc_text = st.session_state.doc_text
            doc_len = len(doc_text)
            try:
                revalidate_subset(selected_changes, doc_len)
            except Exception as e:
                errors: List[str]
                msg = str(e)
                if hasattr(e, "errors"):
                    errors = getattr(e, "errors")
                else:
                    errors = [msg]
                with st.expander("Apply-time Validation Errors", expanded=True):
                    for err in errors:
                        st.markdown(f"- {err}")
                st.error("Apply-time validation failed. Please adjust your selection or JSON.")
            else:
                # Build ResolvedChange list
                resolved: List[ResolvedChange] = []
                for ch in selected_changes:
                    resolved.append(
                        ResolvedChange(
                            change_idx=ch.index,
                            type=ch.type,
                            author=ch.author,
                            timestamp=ch.timestamp,
                            text=ch.text,
                            start=ch.start,
                            end=ch.end,
                            position=ch.position,
                        )
                    )

                try:
                    new_doc_bytes = apply_changes_to_docx(
                        doc_bytes=st.session_state.doc_bytes,
                        resolved_changes=resolved,
                    )
                    st.success("Successfully applied selected changes.")
                    st.download_button(
                        label="Download .docx with tracked changes",
                        data=new_doc_bytes,
                        file_name="tracked_changes_output.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    )
                except Exception as e:
                    st.error(f"Failed to apply changes to .docx: {e}")
else:
    st.info("Upload files and run validation to see the preview and apply options.")
