from dataclasses import dataclass
import re
from typing import Optional


@dataclass
class DomDiff:
    changed: bool
    summary: str


def diff_dom(prev_html: Optional[str], new_html: str) -> DomDiff:
    if prev_html is None:
        return DomDiff(changed=True, summary="Initial state")

    tag_patterns = {
        "button": re.compile(r"<button\b", re.IGNORECASE),
        "input": re.compile(r"<input\b", re.IGNORECASE),
        "form": re.compile(r"<form\b", re.IGNORECASE),
        "div": re.compile(r"<div\b", re.IGNORECASE),
    }

    def count_tags(html: str) -> dict[str, int]:
        return {tag: len(pattern.findall(html)) for tag, pattern in tag_patterns.items()}

    prev_counts = count_tags(prev_html)
    new_counts = count_tags(new_html)

    length_diff = abs(len(new_html) - len(prev_html))
    significant_length_change = length_diff > max(100, len(prev_html) * 0.05)
    significant_count_change = any(
        abs(prev_counts[tag] - new_counts[tag]) > 0 for tag in tag_patterns
    )

    if significant_length_change or significant_count_change:
        return DomDiff(changed=True, summary="New form or modal likely appeared")

    return DomDiff(changed=False, summary="Minor or no structural change")
