from src.agent.dom_scanner import scan_candidate_actions
import asyncio
import logging

import pytest

from src.agent.page_snapshot import AXNode, PageSnapshot, SnapshotNode


class FakeLocator:
    async def count(self):
        return 0

    def nth(self, _index):
        return self

    async def is_visible(self):
        return False


class FakeHandle:
    def __init__(self, tag_name: str, attributes: dict[str, str], inner_text: str = "", uid: str = "uid-123"):
        self.tag_name = tag_name
        self.attributes = attributes
        self.inner_text_value = inner_text
        self.uid = uid

    async def is_visible(self):
        return True

    async def evaluate(self, script):
        if "el.tagName" in script:
            return self.tag_name
        if "__softlight_uid" in script:
            return self.uid
        if "const parts" in script:
            return []
        if "const chain" in script:
            return []
        return None

    async def get_attribute(self, name: str):
        return self.attributes.get(name)

    async def inner_text(self):
        return self.inner_text_value

    async def bounding_box(self):
        return {"x": 0, "y": 0, "width": 10, "height": 10}


class FakeLocatorWithHandle:
    def __init__(self, handle: FakeHandle):
        self.handle = handle

    async def count(self):
        return 1

    def nth(self, _index):
        return self.handle

    async def is_visible(self):
        return await self.handle.is_visible()


class FakeLocatorMultiple:
    def __init__(self, handles: list[FakeHandle]):
        self.handles = handles

    async def count(self):
        return len(self.handles)

    def nth(self, index):
        return self.handles[index]


class FakePage:
    viewport_size = {"width": 0, "height": 0}

    def locator(self, _selector):
        return FakeLocator()

    async def evaluate(self, _script):
        return {"width": 0, "height": 0}


class FakePageLive(FakePage):
    def __init__(self, handles: list[FakeHandle]):
        super().__init__()
        self.handles = handles

    def locator(self, selector):
        if selector.startswith("label["):
            return FakeLocator()
        if selector == "input, textarea, [contenteditable], [role='textbox']":
            if len(self.handles) == 1:
                return FakeLocatorWithHandle(self.handles[0])
            return FakeLocatorMultiple(self.handles)
        return FakeLocator()


def test_contenteditable_field_is_type_candidate():
    dom_nodes = [
        SnapshotNode(
            index=0,
            node_name="div",
            attributes={"contenteditable": "true", "placeholder": "Issue Title"},
            text_snippet=None,
        )
    ]
    ax_nodes = [AXNode(node_id="1", role="textbox", name="Issue Title", dom_node_indices=[0])]
    snapshot = PageSnapshot(dom_nodes=dom_nodes, ax_nodes=ax_nodes)

    candidates, type_ids = asyncio.run(
        scan_candidate_actions(
            page=FakePage(),
            snapshot=snapshot,
            goal="How to create issue named 'sftlightis1' in Linear",
        )
    )

    type_candidates = [c for c in candidates if c.action_type == "type" or c.is_form_field]
    assert type_candidates, "expected at least one type candidate"
    assert any(c.id in type_ids for c in type_candidates)
    assert any("title" in (c.placeholder or c.visible_text or "").lower() for c in type_candidates)


def test_live_page_contenteditable_type_candidates():
    handle = FakeHandle(
        tag_name="div",
        attributes={"contenteditable": "true", "aria-label": "Issue title"},
        inner_text="",
    )
    page = FakePageLive([handle])

    candidates, type_ids = asyncio.run(
        scan_candidate_actions(
            page=page,
            snapshot=None,
            goal="How to create issue named 'sftlightis1' in Linear",
        )
    )

    type_candidates = [c for c in candidates if getattr(c, "is_type_target", False)]
    assert type_candidates, "expected live scan to find a type candidate"
    assert any(c.id in type_ids for c in type_candidates), "type_ids should include live candidates"
    assert any("title" in (c.visible_text or c.description or "").lower() for c in type_candidates)


def test_live_page_multiple_inputs_sets_type_ids_and_logs(caplog):
    caplog.set_level(logging.DEBUG)

    contenteditable_handle = FakeHandle(
        tag_name="div",
        attributes={"contenteditable": "plaintext-only", "role": "textbox", "aria-label": "Issue title"},
        inner_text="",
        uid="uid-content",
    )
    text_input_handle = FakeHandle(
        tag_name="input",
        attributes={"type": "text", "aria-label": "Issue description"},
        inner_text="",
        uid="uid-input",
    )
    page = FakePageLive([contenteditable_handle, text_input_handle])

    candidates, type_ids = asyncio.run(
        scan_candidate_actions(
            page=page,
            snapshot=None,
            goal="Create issue",
        )
    )

    type_candidates = [c for c in candidates if getattr(c, "is_type_target", False)]
    assert type_candidates, "expected live scan to find multiple type candidates"
    assert any(c.id in type_ids for c in type_candidates)

    log_messages = [record.message for record in caplog.records if "live_type_targets" in record.message]
    assert any("count=2" in message or "count=1" in message for message in log_messages)
