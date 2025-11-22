from dataclasses import dataclass
import re
from typing import Optional, Tuple


@dataclass
class DomDiff:
    changed: bool
    summary: str
    score: Optional[float] = None


def _count_tags(html: str, tag_patterns: dict[str, re.Pattern[str]]) -> dict[str, int]:
    return {tag: len(pattern.findall(html)) for tag, pattern in tag_patterns.items()}


def compute_dom_diff(prev_html: Optional[str], new_html: str) -> Tuple[str, Optional[float]]:
    """
    Compare two DOM snapshots and return a human-friendly summary and a heuristic score.

    The score ranges from 0 to 1, where higher values indicate more substantial change.
    Returns (summary, None) when a meaningful score cannot be computed.
    """

    if prev_html is None:
        return "Initial state", 1.0

    tag_patterns = {
        "button": re.compile(r"<button\b", re.IGNORECASE),
        "input": re.compile(r"<input\b", re.IGNORECASE),
        "form": re.compile(r"<form\b", re.IGNORECASE),
        "div": re.compile(r"<div\b", re.IGNORECASE),
    }

    prev_counts = _count_tags(prev_html, tag_patterns)
    new_counts = _count_tags(new_html, tag_patterns)

    length_ratio = 0.0
    if len(new_html) or len(prev_html):
        length_ratio = abs(len(new_html) - len(prev_html)) / max(len(new_html), len(prev_html), 1)

    tag_change_ratio = 0.0
    for tag in tag_patterns:
        prev_count = prev_counts[tag]
        new_count = new_counts[tag]
        denominator = max(prev_count, new_count, 1)
        tag_change_ratio = max(tag_change_ratio, abs(prev_count - new_count) / denominator)

    score = max(length_ratio, tag_change_ratio)

    if score > 0.3:
        summary = "New form or modal likely appeared"
    elif score > 0.1:
        summary = "Notable DOM changes detected"
    else:
        summary = "Minor or no structural change"

    return summary, score


def diff_dom(prev_html: Optional[str], new_html: str) -> DomDiff:
    summary, score = compute_dom_diff(prev_html, new_html)
    return DomDiff(changed=bool(score and score > 0.1), summary=summary, score=score)
