from __future__ import annotations
"""Generic DOM scanner for interactive elements (no app-specific selectors)."""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Set, Tuple, Union

from playwright.async_api import Frame, Page

from .page_snapshot import AXNode, PageSnapshot, SnapshotNode

ActionType = Literal["click", "type"]

DomContext = Union[Page, Frame]


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
    is_type_target: bool = False
    goal_match_score: float = 0.0
    source_hint: Optional[str] = None
    xpath: Optional[str] = None

    def __post_init__(self) -> None:
        # Preserve backward compatibility by mirroring the action_type in kind and
        # ensuring semantics is always a set instance.
        if self.kind is None:
            self.kind = self.action_type
        if self.semantics is None:
            self.semantics = set()
        elif not isinstance(self.semantics, set):
            self.semantics = set(self.semantics)


def _prepare_goal_tokens(goal: Optional[str]) -> Set[str]:
    goal_tokens: Set[str] = set()
    if goal:
        for token in goal.lower().split():
            cleaned = "".join(ch for ch in token if ch.isalnum() or ch in {"-", "_"})
            if len(cleaned) >= 3:
                goal_tokens.add(cleaned)
    return goal_tokens


def _goal_contains_concrete_name(goal: Optional[str]) -> bool:
    if not goal:
        return False
    if re.search(r"[\"']([^\"']{2,})[\"']", goal):
        return True
    return bool(re.search(r"(named|called|titled|title|name|subject)\s+([\w\-]{3,})", goal, re.IGNORECASE))


def _has_text_field_keyword(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in ["title", "name", "subject", "summary", "heading"])


def _looks_like_invite_field(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    phrases = [
        "email",
        "mail",
        "invite",
        "add people",
        "add person",
        "people",
        "name or email",
        "search or invite",
    ]
    return any(phrase in lowered for phrase in phrases)


def _compute_goal_score(candidate_text: str, goal_tokens: Set[str]) -> float:
    if not goal_tokens:
        return 0.0
    lowered = candidate_text.lower()
    return float(sum(1 for tok in goal_tokens if tok in lowered))


def _snapshot_text_from_dom_indices(dom_indices: List[int], by_dom_index: dict[int, SnapshotNode]) -> str:
    parts: list[str] = []
    for idx in dom_indices:
        node = by_dom_index.get(idx)
        if not node:
            continue
        if node.text_snippet:
            parts.append(node.text_snippet)
        if node.attributes:
            for key in ["aria-label", "placeholder", "alt", "title"]:
                if key in node.attributes and node.attributes[key]:
                    parts.append(node.attributes[key])
    return " ".join(part for part in parts if part).strip()


def _scan_click_candidates_from_snapshot(snapshot: PageSnapshot, goal_tokens: Set[str]) -> List[CandidateAction]:
    candidates: List[CandidateAction] = []
    control_roles = {"button", "link", "menuitem", "tab", "checkbox", "radio", "switch"}
    for i, ax in enumerate(snapshot.ax_nodes):
        role = (ax.role or "").lower()
        if role not in control_roles:
            continue
        label_parts = [ax.name or ""]
        label_parts.append(_snapshot_text_from_dom_indices(ax.dom_node_indices, snapshot.by_dom_index))
        label = next((part.strip() for part in label_parts if part and part.strip()), "")
        locator = f"text={label}" if label else "css=*"
        visible_text = label or role
        goal_match_score = _compute_goal_score(visible_text, goal_tokens)
        is_primary_cta = role == "button" and any(
            keyword in visible_text.lower() for keyword in ["submit", "confirm", "save", "next"]
        )
        is_nav_link = role == "link"
        candidates.append(
            CandidateAction(
                id=f"ax_btn_{i}",
                locator=locator,
                action_type="click",
                description=visible_text or role or "click",
                role=role,
                visible_text=visible_text,
                is_primary_cta=is_primary_cta,
                is_nav_link=is_nav_link,
                goal_match_score=goal_match_score,
                semantics={"ax"},
                tag=None,
                aria_label=ax.name,
                text=visible_text,
                source_hint="ax_snapshot",
            )
        )
    return candidates


def _is_dom_text_input(node: SnapshotNode) -> bool:
    node_name = (node.node_name or "").lower()
    if node_name in {"input", "textarea"}:
        return True
    contenteditable_attr = node.attributes.get("contenteditable")
    if contenteditable_attr is not None and contenteditable_attr.lower() != "false":
        return True
    if node.attributes.get("role", "").lower() == "textbox":
        return True
    return False


def _scan_text_candidates_from_snapshot(
    snapshot: PageSnapshot, goal_tokens: Set[str], goal_has_concrete_name: bool
) -> List[CandidateAction]:
    candidates: List[CandidateAction] = []
    text_roles = {"textbox", "searchbox", "combobox"}

    for i, ax in enumerate(snapshot.ax_nodes):
        role = (ax.role or "").lower()
        if role not in text_roles:
            continue
        label = ax.name or _snapshot_text_from_dom_indices(ax.dom_node_indices, snapshot.by_dom_index)
        label = label.strip() if label else ""
        locator = f"text={label}" if label else "css=input,textarea"
        visible_text = label or role or "input"
        goal_match_score = _compute_goal_score(visible_text, goal_tokens)
        if goal_has_concrete_name and _has_text_field_keyword(label):
            goal_match_score += 1.0
        semantics = {"ax"}
        if _looks_like_invite_field(label):
            semantics.update({"invite_field", "share_email_field"})
        candidates.append(
            CandidateAction(
                id=f"ax_input_{i}",
                locator=locator,
                action_type="type",
                description=visible_text or "text entry",
                role=role,
                visible_text=visible_text,
                placeholder=None,
                is_form_field=True,
                is_type_target=True,
                goal_match_score=goal_match_score,
                semantics=semantics,
                aria_label=ax.name,
                source_hint="ax_snapshot",
            )
        )

    dom_inputs = [node for node in snapshot.dom_nodes if _is_dom_text_input(node)]
    for j, node in enumerate(dom_inputs):
        label = node.attributes.get("aria-label") or node.attributes.get("placeholder") or node.text_snippet or ""
        label = label.strip()
        locator = f"text={label}" if label else "css=input,textarea"
        goal_match_score = _compute_goal_score(label or node.node_name, goal_tokens)
        if goal_has_concrete_name and _has_text_field_keyword(label):
            goal_match_score += 1.0
        semantics = {"dom_snapshot"}
        if _looks_like_invite_field(label):
            semantics.update({"invite_field", "share_email_field"})
        candidates.append(
            CandidateAction(
                id=f"dom_input_{j}",
                locator=locator,
                action_type="type",
                description=label or f"{node.node_name} field",
                tag=node.node_name,
                visible_text=label or node.node_name,
                placeholder=node.attributes.get("placeholder"),
                is_form_field=True,
                is_type_target=True,
                goal_match_score=goal_match_score,
                semantics=semantics,
                source_hint="dom_snapshot",
            )
        )

    return candidates


# This scanner tries to build a generic catalogue of interactive elements on the
# page, similar in spirit to a HAR/trace, but focused on click and text entry
# actions. It must remain generic with no app-specific selectors or workflows.
async def scan_candidate_actions(
    page: Page,
    max_actions: int = 60,
    goal: Optional[str] = None,
    snapshot: PageSnapshot | None = None,
    step_index: int | None = None,
) -> tuple[List[CandidateAction], List[str]]:
    """
    Build a HAR-like catalogue of visible interactive elements, identifying both
    click targets and text entry fields using only generic DOM attributes.
    """
    candidates: List[CandidateAction] = []
    candidate_ids: Set[str] = set()
    candidate_by_id: dict[str, CandidateAction] = {}

    # Allow the scan to collect more than max_actions, then prune later.
    # This avoids starving type targets when pages have lots of buttons.
    soft_limit = max_actions * 2
    goal_tokens: Set[str] = _prepare_goal_tokens(goal)
    goal_has_concrete_name = _goal_contains_concrete_name(goal)

    def add_candidate(cand: CandidateAction) -> None:
        existing = candidate_by_id.get(cand.id)
        if existing:
            existing.is_type_target = existing.is_type_target or cand.is_type_target
            return
        if len(candidates) >= soft_limit:
            return
        candidates.append(cand)
        candidate_ids.add(cand.id)
        candidate_by_id[cand.id] = cand

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

    async def compute_xpath(handle, context: DomContext) -> Optional[str]:
        try:
            return await handle.evaluate(
                """
                (el) => {
                    function getXPath(node) {
                        if (!node || node.nodeType !== Node.ELEMENT_NODE) return "";
                        if (!node.parentElement) {
                            return '/' + (node.tagName || '').toLowerCase();
                        }
                        const parent = node.parentElement;
                        const siblings = Array.from(parent.children).filter(
                            (c) => c.tagName === node.tagName
                        );
                        const index = siblings.indexOf(node) + 1;
                        const tag = (node.tagName || '').toLowerCase();
                        return getXPath(parent) + '/' + tag + '[' + index + ']';
                    }
                    return getXPath(el);
                }
                """
            )
        except Exception as exc:
            logging.debug("compute_xpath: failed %r", exc)
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

    async def resolve_labelledby_text(context, handle) -> Optional[str]:
        try:
            aria_labelledby = (await handle.get_attribute("aria-labelledby")) or ""
        except Exception:
            aria_labelledby = ""
        labelledby_text = ""
        if aria_labelledby:
            for ref_id in aria_labelledby.split():
                ref_locator = context.locator(f"#{ref_id}")
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
        return _compute_goal_score(candidate_text, goal_tokens)

    snapshot_click_candidates: List[CandidateAction] = []
    snapshot_text_candidates: List[CandidateAction] = []
    if snapshot:
        snapshot_click_candidates = _scan_click_candidates_from_snapshot(snapshot, goal_tokens)
        snapshot_text_candidates = _scan_text_candidates_from_snapshot(
            snapshot, goal_tokens, goal_has_concrete_name
        )
        if snapshot_text_candidates:
            examples = [
                {
                    "id": cand.id,
                    "role": cand.role,
                    "visible_text": (cand.visible_text or "")[:60],
                }
                for cand in snapshot_text_candidates[:3]
            ]
            logging.debug(
                "text_candidates_from_snapshot step=%s count=%s examples=%s",
                step_index,
                len(snapshot_text_candidates),
                examples,
            )
        for cand in snapshot_click_candidates:
            add_candidate(cand)

    seen_elements: Set[str] = set()

    contexts: List[DomContext] = [page] + [frame for frame in page.frames if frame != page.main_frame]

    # Discover clickable elements.
    clickable_selector = "button, a[href], [role='button'], [role='link'], [onclick], [tabindex]:not([tabindex='-1'])"
    for context in contexts:
        clickable_locator = context.locator(clickable_selector)
        click_count = await clickable_locator.count()
        for i in range(click_count):
            if len(candidates) >= soft_limit:
                break
            handle = clickable_locator.nth(i)
            if not await is_visible(handle):
                continue

            tag_name = (await handle.evaluate("(el) => el.tagName.toLowerCase()")) or ""
            role_attr = (await handle.get_attribute("role")) or ""
            contenteditable_attr = await handle.get_attribute("contenteditable")

            if tag_name in {"input", "textarea"} or contenteditable_attr or role_attr.lower() == "textbox":
                continue

            role = role_attr or None
            aria_label_raw = trim_text(await handle.get_attribute("aria-label"), limit=120)
            labelledby_text = await resolve_labelledby_text(context, handle)
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

            xpath = await compute_xpath(handle, context)

            add_candidate(
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
                    xpath=xpath,
                )
            )

    type_selector = "input, textarea, [contenteditable], [role='textbox']"
    async def make_type_candidate_from_locator(
        context,
        handle,
        nth_index: int,
        type_index: int,
    ) -> tuple[Optional[str], Optional[CandidateAction]]:
        tag_name = (await handle.evaluate("(el) => el.tagName.toLowerCase()")) or ""
        contenteditable_attr = await handle.get_attribute("contenteditable")
        role = (await handle.get_attribute("role")) or None
        input_type = (await handle.get_attribute("type")) or None

        role_lower = (role or "").lower()
        ce_lower = (contenteditable_attr or "").lower() if contenteditable_attr is not None else None

        if tag_name == "input":
            input_type = (input_type or "text").lower()
            if input_type in {"file", "checkbox", "radio", "submit", "button", "reset", "image"}:
                return None, None
        elif tag_name == "textarea":
            input_type = input_type or None
        else:
            is_contenteditable = contenteditable_attr is not None and ce_lower in {"", "true", "plaintext-only"}
            if not is_contenteditable and role_lower != "textbox":
                return None, None

        aria_label_raw = trim_text(await handle.get_attribute("aria-label"), limit=120)
        placeholder_value = trim_text(await handle.get_attribute("placeholder"), limit=120)
        element_id = (await handle.get_attribute("id")) or ""

        label_text = ""
        if element_id:
            label_locator = context.locator(f"label[for=\"{element_id}\"]")
            if await label_locator.count() > 0:
                label_text = trim_text(await label_locator.first.inner_text()) or ""

        labelledby_text = await resolve_labelledby_text(context, handle)
        invite_hint = " ".join(filter(None, [label_text, placeholder_value, aria_label_raw, labelledby_text]))
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
            trim_text(aria_label_raw, limit=120)
            or trim_text(label_text, limit=120)
            or trim_text(placeholder_value, limit=120)
            or trim_text(text_content, limit=120)
            or trim_text(ancestor_text, limit=120)
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
        if _looks_like_invite_field(invite_hint):
            semantics.update({"invite_field", "share_email_field"})

        class_name = section_chain[0].get("className", "") if section_chain else ""
        is_primary_cta, is_nav_link, is_form_field = compute_flags(
            tag_name, role, class_name, section_label, True, semantics
        )
        is_form_field = True
        goal_match_score = compute_goal_score(f"{visible_text} {ancestor_text}")
        if goal_has_concrete_name and _has_text_field_keyword(primary_hint or visible_text):
            goal_match_score += 1.0

        element_uid = await get_element_uid(handle)

        if tag_name == "textarea":
            base_desc = "multiline text area"
        elif contenteditable_attr is not None:
            base_desc = "editable text box"
        elif tag_name == "input":
            base_desc = f"text input ({input_type})" if input_type else "text input"
        else:
            base_desc = "text entry"

        desc_hint = primary_hint or placeholder_value or text_content
        description = f"{base_desc} \"{desc_hint}\"" if desc_hint else base_desc

        xpath = await compute_xpath(handle, context)

        candidate = CandidateAction(
            id=f"input_{type_index}",
            locator=f"{type_selector} >> nth={nth_index}",
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
            is_type_target=True,
            goal_match_score=goal_match_score,
            source_hint="playwright_scan",
            xpath=xpath,
        )
        return element_uid, candidate

    def next_type_index() -> int:
        max_index = -1
        for cid in candidate_ids:
            if cid.startswith("input_"):
                try:
                    max_index = max(max_index, int(cid.split("_", 1)[1]))
                except ValueError:
                    continue
        return max_index + 1

    async def augment_with_type_candidates_from_playwright(start_index: int) -> None:
        # Scan type like fields in the main page and all child frames.
        contexts = [page] + list(page.frames)
        type_index = start_index

        for ctx in contexts:
            type_locator = ctx.locator(type_selector)
            type_count = await type_locator.count()

            for i in range(type_count):
                if len(candidates) >= soft_limit:
                    return

                handle = type_locator.nth(i)
                if not await is_visible(handle):
                    continue

                element_uid, candidate = await make_type_candidate_from_locator(
                    ctx, handle, i, type_index
                )
                if not candidate:
                    continue

                if element_uid and element_uid in seen_elements:
                    continue

                add_candidate(candidate)
                if element_uid:
                    seen_elements.add(element_uid)

                type_index += 1

    async def augment_with_type_candidates_from_xpath(start_index: int) -> None:
        type_index = start_index
        for ctx_idx, context in enumerate(contexts):
            try:
                nodes = await context.evaluate(
                    """
                    () => {
                        function getXPath(node) {
                            if (node.nodeType !== Node.ELEMENT_NODE) return "";
                            if (node.id) {
                                return 'id("' + node.id + '")';
                            }
                            const parts = [];
                            while (node && node.nodeType === Node.ELEMENT_NODE) {
                                let index = 1;
                                let sibling = node.previousSibling;
                                while (sibling) {
                                    if (sibling.nodeType === Node.ELEMENT_NODE &&
                                        sibling.nodeName === node.nodeName) {
                                        index += 1;
                                    }
                                    sibling = sibling.previousSibling;
                                }
                                const tagName = node.nodeName.toLowerCase();
                                const part = index > 1 ? `${tagName}[${index}]` : tagName;
                                parts.unshift(part);
                                node = node.parentNode;
                            }
                            return "/" + parts.join("/");
                        }

                        const selector = 'input, textarea, [contenteditable], [role="textbox"]';
                        const els = Array.from(document.querySelectorAll(selector));
                        return els.map((el) => {
                            const tag = el.tagName ? el.tagName.toLowerCase() : "";
                            const placeholder = el.getAttribute("placeholder") || "";
                            const aria = el.getAttribute("aria-label") || "";
                            const text = (el.innerText || el.value || "").trim();
                            return {
                                xpath: getXPath(el),
                                tag,
                                placeholder,
                                aria,
                                text,
                            };
                        });
                    }
                    """
                )
            except Exception as exc:
                logging.debug("xpath_type_scan: eval_failed step=%s ctx=%s error=%r", step_index, ctx_idx, exc)
                continue

            if not nodes:
                logging.debug("xpath_type_scan: no_nodes step=%s ctx=%s", step_index, ctx_idx)
                continue

            logging.debug("xpath_type_scan: step=%s ctx=%s count=%s", step_index, ctx_idx, len(nodes))

            for idx, node in enumerate(nodes):
                if len(candidates) >= max_actions:
                    logging.debug(
                        "xpath_type_scan abort: max_actions reached step=%s len=%s",
                        step_index,
                        len(candidates),
                    )
                    return

                xpath = node.get("xpath") or None
                if not xpath:
                    continue

                if any(c.xpath == xpath for c in candidates if c.xpath):
                    continue

                tag = node.get("tag") or ""
                placeholder = node.get("placeholder") or ""
                aria = node.get("aria") or ""
                text = node.get("text") or ""
                visible_text = aria or placeholder or text

                semantics = compute_semantics(
                    tag=tag,
                    role=None,
                    label=visible_text,
                    placeholder=placeholder,
                    text=text,
                    input_type=None,
                )
                if _looks_like_invite_field(visible_text):
                    semantics.update({"invite_field", "share_email_field"})

                ancestor_text = ""
                section_label = None
                is_primary_cta = False
                is_nav_link = False
                is_form_field = True

                goal_match_score = compute_goal_score(visible_text)
                if goal_has_concrete_name and _has_text_field_keyword(visible_text):
                    goal_match_score += 1.0

                description = f"xpath text entry \"{visible_text[:40]}\"" if visible_text else "xpath text entry"

                cand = CandidateAction(
                    id=f"xpath_input_{type_index}",
                    locator=f"xpath={xpath}",
                    action_type="type",
                    description=description,
                    tag=tag,
                    role=None,
                    aria_label=aria,
                    type=None,
                    text=visible_text,
                    semantics=semantics,
                    bounding_box=None,
                    kind="type",
                    visible_text=visible_text,
                    placeholder=placeholder,
                    ancestor_text=ancestor_text,
                    section_label=section_label,
                    is_primary_cta=is_primary_cta,
                    is_nav_link=is_nav_link,
                    is_form_field=is_form_field,
                    is_type_target=True,
                    goal_match_score=goal_match_score,
                    source_hint="xpath_scan",
                    xpath=xpath,
                )

                add_candidate(cand)
                logging.debug(
                    "xpath_type_scan[%s:%s]: added_type_candidate id=%s xpath=%s desc=%s",
                    ctx_idx,
                    idx,
                    cand.id,
                    xpath,
                    description,
                )
                type_index += 1

        logging.debug(
            "xpath_type_scan done step=%s total_xpath_type_candidates=%s",
            step_index,
            len([c for c in candidates if c.is_type_target and c.source_hint == "xpath_scan"]),
        )

    for cand in snapshot_text_candidates:
        add_candidate(cand)

    await augment_with_type_candidates_from_playwright(next_type_index())
    await augment_with_type_candidates_from_xpath(next_type_index())

    live_type_candidates = [
        c for c in candidates if c.is_type_target and c.source_hint == "playwright_scan"
    ]
    sample_entries = [
        {
            "id": cand.id,
            "kind": cand.kind,
            "text": (cand.visible_text or cand.description or "")[:80],
        }
        for cand in live_type_candidates[:5]
    ]
    logging.debug(
        "live_type_targets count=%s sample=%s",
        len(live_type_candidates),
        sample_entries,
    )

    # Hard cap total actions, but never starve type targets.
    if len(candidates) > max_actions:
        type_candidates = [c for c in candidates if c.is_type_target]
        other_candidates = [c for c in candidates if not c.is_type_target]

        # Keep at most half the budget for type fields, but at least one if any exist.
        max_type_keep = min(len(type_candidates), max(1, max_actions // 2)) if type_candidates else 0

        # Prefer fields that look like title or name, then by goal match.
        type_candidates_sorted = sorted(
            type_candidates,
            key=lambda c: (
                1 if "title_field" in (c.semantics or set()) else 0,
                c.goal_match_score,
            ),
            reverse=True,
        )
        kept_types = type_candidates_sorted[:max_type_keep]

        remaining_slots = max_actions - len(kept_types)
        other_candidates_sorted = sorted(
            other_candidates,
            key=lambda c: (
                1 if c.is_primary_cta else 0,
                c.goal_match_score,
            ),
            reverse=True,
        )
        kept_others = other_candidates_sorted[: max(0, remaining_slots)]

        candidates = kept_types + kept_others

    type_ids = [c.id for c in candidates if c.is_type_target]
    return candidates, type_ids
