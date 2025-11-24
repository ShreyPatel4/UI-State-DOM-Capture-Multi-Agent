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
    visible_text: str = ""
    placeholder: Optional[str] = None
    ancestor_text: str = ""
    section_label: Optional[str] = None
    is_primary_cta: bool = False
    is_nav_link: bool = False
    is_form_field: bool = False
    goal_match_score: float = 0.0

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
    page: Page, max_actions: int = 60, goal: Optional[str] = None
) -> tuple[List[CandidateAction], List[str]]:
    """
    Build a HAR-like catalogue of visible interactive elements, identifying both
    click targets and text entry fields using only generic DOM attributes.
    """
    candidates: List[CandidateAction] = []
    goal_tokens: Set[str] = set()
    if goal:
        for token in goal.lower().split():
            cleaned = "".join(ch for ch in token if ch.isalnum() or ch in {"-", "_"})
            if len(cleaned) >= 3:
                goal_tokens.add(cleaned)

    viewport_width = 0.0
    viewport_height = 0.0
    viewport = page.viewport_size
    if viewport:
        viewport_width = float(viewport.get("width", 0.0) or 0.0)
        viewport_height = float(viewport.get("height", 0.0) or 0.0)
    else:
        try:
            metrics = await page.evaluate(
                "() => ({ width: window.innerWidth || document.documentElement.clientWidth || 0, height: window.innerHeight || document.documentElement.clientHeight || 0 })"
            )
            viewport_width = float(metrics.get("width", 0.0) or 0.0)
            viewport_height = float(metrics.get("height", 0.0) or 0.0)
        except Exception:
            viewport_width = viewport_height = 0.0

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

    async def get_element_uid(handle) -> Optional[str]:
        try:
            return await handle.evaluate(
                """
                (el) => {
                    if (!el.__softlight_uid) {
                        el.__softlight_uid = `${Date.now()}_${Math.random().toString(36).slice(2)}`;
                    }
                    return el.__softlight_uid;
                }
                """
            )
        except Exception:
            return None

    async def collect_ancestor_text(handle) -> str:
        try:
            texts = await handle.evaluate(
                """
                (el) => {
                    const parts = [];
                    let node = el.parentElement;
                    let depth = 0;
                    while (node && depth < 3) {
                        const text = (node.innerText || "").trim();
                        if (text) {
                            parts.push(text);
                        }
                        node = node.parentElement;
                        depth += 1;
                    }
                    return parts;
                }
                """
            )
        except Exception:
            return ""
        if not texts:
            return ""
        joined = " | ".join(texts)
        return trim_text(joined, limit=200) or ""

    async def resolve_labelledby_text(handle) -> Optional[str]:
        try:
            aria_labelledby = (await handle.get_attribute("aria-labelledby")) or ""
        except Exception:
            aria_labelledby = ""
        labelledby_text = ""
        if aria_labelledby:
            for ref_id in aria_labelledby.split():
                ref_locator = page.locator(f"#{ref_id}")
                try:
                    if await ref_locator.count() > 0:
                        labelledby_text = trim_text(await ref_locator.first.inner_text(), limit=120) or ""
                        if labelledby_text:
                            break
                except Exception:
                    continue
        return labelledby_text or None

    async def collect_section_chain(handle) -> list[dict]:
        try:
            return await handle.evaluate(
                """
                (el) => {
                    const chain = [];
                    let node = el;
                    let depth = 0;
                    while (node && depth < 4) {
                        const className = typeof node.className === "string" ? node.className : "";
                        chain.push({
                            role: node.getAttribute ? node.getAttribute("role") : null,
                            className,
                            tag: node.tagName ? node.tagName.toLowerCase() : "",
                        });
                        node = node.parentElement;
                        depth += 1;
                    }
                    return chain;
                }
                """
            )
        except Exception:
            return []

    def infer_section_label(section_chain: list[dict], bbox: Optional[Tuple[float, float, float, float]]) -> Optional[str]:
        bbox_center_x = None
        bbox_center_y = None
        if bbox:
            bbox_center_x = bbox[0] + bbox[2] / 2
            bbox_center_y = bbox[1] + bbox[3] / 2

        for entry in section_chain:
            class_lower = (entry.get("className") or "").lower()
            role_lower = (entry.get("role") or "").lower()
            tag_lower = (entry.get("tag") or "").lower()

            if tag_lower == "footer" or "footer" in class_lower:
                return "footer"
            if "sidebar" in class_lower or "side-bar" in class_lower:
                if bbox_center_x is not None and viewport_width:
                    if bbox_center_x < viewport_width * 0.4:
                        return "left_sidebar"
                    if bbox_center_x > viewport_width * 0.6:
                        return "right_sidebar"
                return "left_sidebar"
            if role_lower == "navigation" or "nav" in class_lower or "menu" in class_lower:
                if bbox_center_y is not None and viewport_height:
                    if bbox_center_y < viewport_height * 0.2:
                        return "top_bar"
                    if bbox_center_y > viewport_height * 0.8:
                        return "footer"
                if bbox_center_x is not None and viewport_width:
                    if bbox_center_x < viewport_width * 0.4:
                        return "left_sidebar"
                    if bbox_center_x > viewport_width * 0.6:
                        return "right_sidebar"
                return "main_content"
            if "toolbar" in class_lower or "header" in class_lower or tag_lower == "header":
                return "top_bar"

        if bbox_center_y is not None and viewport_height:
            if bbox_center_y > viewport_height * 0.85:
                return "footer"
            if 0.2 * viewport_height < bbox_center_y < 0.8 * viewport_height and bbox_center_x is not None:
                if viewport_width and 0.2 * viewport_width < bbox_center_x < 0.8 * viewport_width:
                    return "main_content"
        return None

    def compute_flags(
        tag: str,
        role: Optional[str],
        class_name: str,
        section_label: Optional[str],
        is_text_input: bool,
        semantics: Set[str],
    ) -> tuple[bool, bool, bool]:
        class_lower = (class_name or "").lower()
        role_lower = (role or "").lower()
        is_button_like = tag == "button" or role_lower == "button"
        primary_keywords = ["primary", "cta", "submit", "confirm", "save"]
        is_primary_cta = is_button_like and any(keyword in class_lower for keyword in primary_keywords)

        if not is_primary_cta and ("primary" in semantics or "cta" in semantics):
            is_primary_cta = is_button_like or bool(semantics.intersection({"primary", "cta"}))

        is_nav_link = False
        if tag == "a" or role_lower == "link":
            if section_label in {"left_sidebar", "right_sidebar", "top_bar", "footer"}:
                is_nav_link = True
            if any(keyword in class_lower for keyword in ["nav", "menu", "sidebar", "tab"]):
                is_nav_link = True

        is_form_field = is_text_input or role_lower == "textbox"

        return is_primary_cta, is_nav_link, is_form_field

    def compute_goal_score(candidate_text: str) -> float:
        if not goal_tokens:
            return 0.0
        lowered = candidate_text.lower()
        return float(sum(1 for tok in goal_tokens if tok in lowered))

    seen_elements: Set[str] = set()

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
        aria_label_raw = trim_text(await handle.get_attribute("aria-label"), limit=120)
        labelledby_text = await resolve_labelledby_text(handle)
        aria_label = aria_label_raw or labelledby_text
        placeholder = trim_text(await handle.get_attribute("placeholder"), limit=120)
        inner_text = trim_text(await handle.inner_text(), limit=120)
        visible_text = inner_text or aria_label or ""
        ancestor_text = await collect_ancestor_text(handle)
        section_chain = await collect_section_chain(handle)
        bbox = await get_bounding_box(handle)
        section_label = infer_section_label(section_chain, bbox)
        label_text = trim_text(aria_label or visible_text, limit=120)
        description = (
            f"{tag_name or 'element'} \"{label_text}\"" if label_text else f"{tag_name or 'element'} index {i}"
        )
        semantics = compute_semantics(
            tag=tag_name,
            role=role,
            label=label_text or "",
            placeholder=placeholder or "",
            text=inner_text or "",
            input_type=None,
        )
        class_name = section_chain[0].get("className", "") if section_chain else ""
        is_primary_cta, is_nav_link, is_form_field = compute_flags(
            tag_name, role, class_name, section_label, False, semantics
        )
        goal_match_score = compute_goal_score(f"{visible_text} {ancestor_text}")
        element_uid = await get_element_uid(handle)
        if element_uid and element_uid in seen_elements:
            continue
        if element_uid:
            seen_elements.add(element_uid)

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
                text=visible_text or inner_text,
                semantics=semantics,
                bounding_box=bbox,
                kind="click",
                visible_text=visible_text,
                placeholder=placeholder,
                ancestor_text=ancestor_text,
                section_label=section_label,
                is_primary_cta=is_primary_cta,
                is_nav_link=is_nav_link,
                is_form_field=is_form_field,
                goal_match_score=goal_match_score,
            )
        )

    # Discover text entry elements.
    type_selector = "input, textarea, [contenteditable], [role='textbox']"
    type_locator = page.locator(type_selector)
    type_count = await type_locator.count()
    type_index = 0
    allowed_input_types = {"text", "search", "email", "url", "number", "password"}
    for i in range(type_count):
        if len(candidates) >= max_actions:
            return candidates, [c.id for c in candidates if c.action_type == "type"]
        handle = type_locator.nth(i)
        if not await is_visible(handle):
            continue

        tag_name = (await handle.evaluate("(el) => el.tagName.toLowerCase()")) or ""
        contenteditable_attr = await handle.get_attribute("contenteditable")
        if tag_name == "input":
            input_type = ((await handle.get_attribute("type")) or "text").lower()
            if input_type not in allowed_input_types:
                continue
        else:
            input_type = (await handle.get_attribute("type")) or None

        role = (await handle.get_attribute("role")) or None
        if tag_name not in {"input", "textarea"} and not contenteditable_attr and (role or "").lower() != "textbox":
            continue

        aria_label_raw = trim_text(await handle.get_attribute("aria-label"), limit=120)
        placeholder_value = trim_text(await handle.get_attribute("placeholder"), limit=120)
        element_id = (await handle.get_attribute("id")) or ""

        label_text = ""
        if element_id:
            label_locator = page.locator(f"label[for=\"{element_id}\"]")
            if await label_locator.count() > 0:
                label_text = trim_text(await label_locator.first.inner_text()) or ""

        labelledby_text = await resolve_labelledby_text(handle)
        aria_label = aria_label_raw or labelledby_text

        text_content = trim_text(await handle.inner_text(), limit=120) or ""
        primary_hint = next(
            (
                hint
                for hint in [label_text, aria_label, labelledby_text, placeholder_value, text_content, element_id]
                if hint
            ),
            "",
        )

        ancestor_text = await collect_ancestor_text(handle)
        section_chain = await collect_section_chain(handle)
        bbox = await get_bounding_box(handle)
        section_label = infer_section_label(section_chain, bbox)
        visible_text = (
            trim_text(primary_hint, limit=120)
            or text_content
            or aria_label
            or placeholder_value
            or ancestor_text
            or ""
        )

        semantics = compute_semantics(
            tag=tag_name,
            role=role,
            label=primary_hint,
            placeholder=placeholder_value or "",
            text=text_content,
            input_type=input_type,
        )

        class_name = section_chain[0].get("className", "") if section_chain else ""
        is_primary_cta, is_nav_link, is_form_field = compute_flags(
            tag_name, role, class_name, section_label, True, semantics
        )
        goal_match_score = compute_goal_score(f"{visible_text} {ancestor_text}")

        element_uid = await get_element_uid(handle)
        if element_uid and element_uid in seen_elements:
            continue
        if element_uid:
            seen_elements.add(element_uid)

        if tag_name == "textarea":
            base_desc = "multiline text area"
        elif contenteditable_attr is not None:
            base_desc = "editable text box"
        elif tag_name == "input":
            base_desc = f"text input ({input_type})" if input_type else "text input"
        else:
            base_desc = "text entry"

        desc_hint = primary_hint or placeholder_value or text_content
        description = (
            f"{base_desc} \"{desc_hint}\"" if desc_hint else base_desc
        )
        candidates.append(
            CandidateAction(
                id=f"input_{type_index}",
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
                visible_text=visible_text,
                placeholder=placeholder_value,
                ancestor_text=ancestor_text,
                section_label=section_label,
                is_primary_cta=is_primary_cta,
                is_nav_link=is_nav_link,
                is_form_field=is_form_field,
                goal_match_score=goal_match_score,
            )
        )
        type_index += 1

    type_ids = [c.id for c in candidates if c.action_type == "type"]
    return candidates, type_ids
