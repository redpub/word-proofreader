from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Dict, Any


ChangeType = Literal["insert", "delete", "replace", "comment"]


@dataclass
class Change:
    index: int                     # original order index in JSON
    type: ChangeType
    author: str
    timestamp: Optional[str]
    position: Optional[int]        # for insert
    start: Optional[int]           # for delete/replace/comment
    end: Optional[int]             # for delete/replace/comment
    text: str                      # inserted / replacement / comment text

    def range_tuple(self) -> Optional[Tuple[int, int]]:
        if self.type == "insert":
            return None
        if self.start is None or self.end is None:
            return None
        return self.start, self.end

    def display_label(self) -> str:
        base = f"Change {self.index + 1} ({self.type})"
        return base


class ValidationError(Exception):
    def __init__(self, errors: List[str]):
        super().__init__("\n".join(errors))
        self.errors = errors


def parse_json(json_bytes: bytes) -> Dict[str, Any]:
    try:
        data = json.loads(json_bytes.decode("utf-8"))
    except Exception as e:
        raise ValidationError([f"JSON parse error: {e}"])
    if not isinstance(data, dict):
        raise ValidationError(["Top-level JSON must be an object with key 'changes'."])
    if "changes" not in data:
        raise ValidationError(["JSON must contain a 'changes' array."])
    if not isinstance(data["changes"], list):
        raise ValidationError(["'changes' must be an array."])
    return data


def parse_changes(data: Dict[str, Any]) -> List[Change]:
    errors: List[str] = []
    changes: List[Change] = []
    raw_changes = data.get("changes", [])
    for idx, c in enumerate(raw_changes):
        if not isinstance(c, dict):
            errors.append(f"changes[{idx}] must be an object.")
            continue

        ctype = c.get("type")
        if ctype not in ("insert", "delete", "replace", "comment"):
            errors.append(f"changes[{idx}].type must be one of insert/delete/replace/comment.")
            continue

        author = c.get("author")
        if not isinstance(author, str) or not author.strip():
            errors.append(f"changes[{idx}].author must be a non-empty string.")
            continue

        timestamp = c.get("timestamp")
        if timestamp is not None and not isinstance(timestamp, str):
            errors.append(f"changes[{idx}].timestamp must be a string if provided.")

        text = c.get("text")
        if not isinstance(text, str):
            errors.append(f"changes[{idx}].text must be a string.")
            continue

        position = c.get("position")
        start = c.get("start")
        end = c.get("end")

        if ctype == "insert":
            if position is None or not isinstance(position, int):
                errors.append(f"changes[{idx}] (insert) must have integer 'position'.")
                continue
            if start is not None or end is not None:
                errors.append(f"changes[{idx}] (insert) must not have 'start'/'end'.")
                continue
        else:
            if not isinstance(start, int) or not isinstance(end, int):
                errors.append(f"changes[{idx}] ({ctype}) must have integer 'start' and 'end'.")
                continue
            if start < 0 or end < 0:
                errors.append(f"changes[{idx}] ({ctype}) has negative 'start'/'end'.")
            if start >= end:
                errors.append(f"changes[{idx}] ({ctype}) must satisfy start < end.")
            if position is not None:
                errors.append(f"changes[{idx}] ({ctype}) must not have 'position'.")

        change = Change(
            index=idx,
            type=ctype,  # type: ignore[arg-type]
            author=author,
            timestamp=timestamp,
            position=position if isinstance(position, int) else None,
            start=start if isinstance(start, int) else None,
            end=end if isinstance(end, int) else None,
            text=text,
        )
        changes.append(change)

    if errors:
        raise ValidationError(errors)
    return changes


def validate_ranges(changes: List[Change], doc_length: int) -> None:
    errors: List[str] = []

    # Bounds and non-overlap, character-based on full text.
    intervals: List[Tuple[int, int, int]] = []  # (start, end, idx)
    for ch in changes:
        if ch.type == "insert":
            if ch.position is None:
                errors.append(f"Change {ch.index + 1} (insert) missing 'position'.")
            else:
                if ch.position < 0 or ch.position > doc_length:
                    errors.append(
                        f"Change {ch.index + 1} (insert) position {ch.position} out of bounds [0, {doc_length}]."
                    )
        else:
            if ch.start is None or ch.end is None:
                errors.append(f"Change {ch.index + 1} ({ch.type}) missing 'start'/'end'.")
                continue
            if ch.start < 0 or ch.end > doc_length:
                errors.append(
                    f"Change {ch.index + 1} ({ch.type}) range [{ch.start}, {ch.end}) out of bounds [0, {doc_length}]."
                )
            if ch.start >= ch.end:
                errors.append(
                    f"Change {ch.index + 1} ({ch.type}) must satisfy 0 ≤ start < end ≤ document_length."
                )
            intervals.append((ch.start, ch.end, ch.index))

    # Sort by start then end
    intervals.sort(key=lambda t: (t[0], t[1]))
    for i in range(1, len(intervals)):
        prev_start, prev_end, prev_idx = intervals[i - 1]
        cur_start, cur_end, cur_idx = intervals[i]
        if cur_start < prev_end:
            errors.append(
                f"Overlapping ranges between Change {prev_idx + 1} [{prev_start}, {prev_end}) "
                f"and Change {cur_idx + 1} [{cur_start}, {cur_end})."
            )

    if errors:
        raise ValidationError(errors)


def extract_preview_segments(doc_text: str, change: Change, context: int = 40) -> Dict[str, Any]:
    """
    Return a dict with original snippet and, for inserts, insertion position snippet.
    Used only for preview; does not modify text.
    """
    if change.type == "insert":
        pos = change.position or 0
        before_start = max(0, pos - context)
        after_end = min(len(doc_text), pos + context)
        before = doc_text[before_start:pos]
        after = doc_text[pos:after_end]
        return {
            "before": before,
            "after": after,
            "original": "",
        }
    else:
        start = change.start or 0
        end = change.end or 0
        snippet_start = max(0, start - context)
        snippet_end = min(len(doc_text), end + context)
        original = doc_text[start:end]
        before = doc_text[snippet_start:start]
        after = doc_text[end:snippet_end]
        return {
            "before": before,
            "after": after,
            "original": original,
        }


def revalidate_subset(changes: List[Change], doc_length: int) -> None:
    """
    Re-run range validation on a subset of changes when user selectively applies.
    """
    validate_ranges(changes, doc_length)
