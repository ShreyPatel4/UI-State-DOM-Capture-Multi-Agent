from __future__ import annotations
"""Generic DOM scanner for interactive elements (no app-specific selectors)."""

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Set, Tuple

from playwright.async_api import Page

ActionType = Literal["click", "type"]


@dataclass
class CandidateAction:
    id: str
    action_type: ActionType
    locator: str
    description: str
    tag: Optional[str] = None
    role: Optional[str] = None
    aria_label: Optional[str] = None
    type: Optional[str] = None
    text: Optional[str] = None
    semantics: Set[str] = field(default_factory=set)
    bounding_box: Optional[Tuple[float, float, float, float]] = None
    kind: Optional[ActionType] = None

    def __post_init__(self) -> None:
        # Preserve backward compatibility by mirroring the action_type in kind and
        # ensuring semantics is always a set instance.
        if self.kind is None:
            self.kind = self.action_type
        if self.semantics is None:
            self.semantics = set()
        elif not isinstance(self.semantics, set):
            self.semantics = set(self.semantics)


# This scanner tries to build a generic catalogue of interactive elements on the
# page, similar in spirit to a HAR/trace, but focused on click and text entry
# actions. It must remain generic with no app-specific selectors or workflows.
async def scan_candidate_actions(
    page: Page, max_actions: int = 60
) -> tuple[List[CandidateAction], List[str]]:
    """
    Build a HAR-like catalogue of visible interactive elements, identifying both
    click targets and text entry fields using only generic DOM attributes.
    """
    candidates: List[CandidateAction] = []

    async def is_visible(handle) -> bool:
        try:
            return await handle.is_visible()
        except Exception:
            return False

    async def get_bounding_box(handle) -> Optional[Tuple[float, float, float, float]]:
        try:
            box = await handle.bounding_box()
        except Exception:
            return None
        if not box:
            return None
        return (box.get("x", 0.0), box.get("y", 0.0), box.get("width", 0.0), box.get("height", 0.0))

    def compute_semantics(
        *, tag: str, role: Optional[str], label: str, placeholder: str, text: str, input_type: Optional[str]
    ) -> Set[str]:
        semantics: Set[str] = set()
        normalized = " ".join(
            [tag or "", role or "", label or "", placeholder or "", text or "", input_type or ""]
        ).lower()
        if tag == "button" or (role and "button" in role):
            if any(keyword in normalized for keyword in ["create", "new", "add", "start", "compose"]):
                semantics.add("cta")
            if any(keyword in normalized for keyword in ["delete", "remove", "discard", "danger"]):
                semantics.add("danger")
        if any(keyword in normalized for keyword in ["search", "find", "filter"]):
            semantics.add("search_field")
        if any(keyword in normalized for keyword in ["title", "name", "subject", "heading"]):
            semantics.add("title_field")
        if any(keyword in normalized for keyword in ["save", "submit", "confirm", "done"]):
            semantics.add("primary")
        if any(keyword in normalized for keyword in ["cancel", "back", "close"]):
            semantics.add("nav")
        return semantics

    def trim_text(value: Optional[str], limit: int = 80) -> Optional[str]:
        if not value:
            return None
        value = value.strip()
        if not value:
            return None
        return value[:limit]

    # Discover clickable elements.
    clickable_selector = "button, a[href], [role='button'], [role='link'], [onclick], [tabindex]:not([tabindex='-1'])"
    clickable_locator = page.locator(clickable_selector)
    click_count = await clickable_locator.count()
    for i in range(click_count):
        if len(candidates) >= max_actions:
            return candidates, [c.id for c in candidates if c.action_type == "type"]
        handle = clickable_locator.nth(i)
        if not await is_visible(handle):
            continue

        tag_name = (await handle.evaluate("(el) => el.tagName.toLowerCase()")) or ""
        role = (await handle.get_attribute("role")) or None
        aria_label = (await handle.get_attribute("aria-label")) or None
        inner_text = trim_text(await handle.inner_text())
        label_text = trim_text(aria_label or inner_text)
        description = (
            f"{tag_name or 'element'} \"{label_text}\"" if label_text else f"{tag_name or 'element'} index {i}"
        )
        semantics = compute_semantics(
            tag=tag_name,
            role=role,
            label=label_text or "",
            placeholder="",
            text=inner_text or "",
            input_type=None,
        )
        bbox = await get_bounding_box(handle)
        candidates.append(
            CandidateAction(
                id=f"btn_{i}" if tag_name != "a" else f"link_{i}",
                locator=f"{clickable_selector} >> nth={i}",
                action_type="click",
                description=description,
                tag=tag_name,
                role=role,
                aria_label=aria_label,
                type=None,
                text=inner_text,
                semantics=semantics,
                bounding_box=bbox,
                kind="click",
            )
        )

    # Discover text entry elements.
    type_selector = "input, textarea, [contenteditable], [role='textbox']"
    type_locator = page.locator(type_selector)
    type_count = await type_locator.count()
    for i in range(type_count):
        if len(candidates) >= max_actions:
            return candidates, [c.id for c in candidates if c.action_type == "type"]
        handle = type_locator.nth(i)
        if not await is_visible(handle):
            continue

        tag_name = (await handle.evaluate("(el) => el.tagName.toLowerCase()")) or ""
        if tag_name == "input":
            input_type = ((await handle.get_attribute("type")) or "text").lower()
            if input_type in {"hidden", "checkbox", "radio", "submit", "reset", "button"}:
                continue
        else:
            input_type = (await handle.get_attribute("type")) or None

        role = (await handle.get_attribute("role")) or None
        aria_label = (await handle.get_attribute("aria-label")) or None
        placeholder = trim_text(await handle.get_attribute("placeholder")) or ""
        element_id = (await handle.get_attribute("id")) or ""

        label_text = ""
        if element_id:
            label_locator = page.locator(f"label[for=\"{element_id}\"]")
            if await label_locator.count() > 0:
                label_text = trim_text(await label_locator.first.inner_text()) or ""

        aria_labelledby = (await handle.get_attribute("aria-labelledby")) or ""
        labelledby_text = ""
        if aria_labelledby:
            for ref_id in aria_labelledby.split():
                ref_locator = page.locator(f"#{ref_id}")
                if await ref_locator.count() > 0:
                    labelledby_text = trim_text(await ref_locator.first.inner_text()) or ""
                    if labelledby_text:
                        break

        text_content = trim_text(await handle.inner_text()) or ""
        primary_hint = next(
            (
                hint
                for hint in [label_text, aria_label, labelledby_text, placeholder, text_content, element_id]
                if hint
            ),
            "",
        )

        semantics = compute_semantics(
            tag=tag_name,
            role=role,
            label=primary_hint,
            placeholder=placeholder,
            text=text_content,
            input_type=input_type,
        )

        if tag_name == "textarea":
            base_desc = "multiline text area"
        elif (await handle.get_attribute("contenteditable")) is not None:
            base_desc = "editable text box"
        elif tag_name == "input":
            base_desc = f"text input ({input_type})" if input_type else "text input"
        else:
            base_desc = "text entry"

        desc_hint = primary_hint or placeholder or text_content
        description = (
            f"{base_desc} \"{desc_hint}\"" if desc_hint else base_desc
        )

        bbox = await get_bounding_box(handle)
        candidates.append(
            CandidateAction(
                id=f"input_{i}",
                locator=f"{type_selector} >> nth={i}",
                action_type="type",
                description=description,
                tag=tag_name,
                role=role,
                aria_label=aria_label,
                type=input_type,
                text=primary_hint or text_content,
                semantics=semantics,
                bounding_box=bbox,
                kind="type",
            )
        )

    type_ids = [c.id for c in candidates if c.action_type == "type"]
    return candidates, type_ids
