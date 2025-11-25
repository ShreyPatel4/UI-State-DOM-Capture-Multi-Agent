from src.agent.dom_scanner import scan_candidate_actions
from src.agent.page_snapshot import AXNode, PageSnapshot, SnapshotNode
import asyncio


class FakeLocator:
    async def count(self):
        return 0

    def nth(self, _index):
        return self

    async def is_visible(self):
        return False


class FakePage:
    viewport_size = {"width": 0, "height": 0}

    def locator(self, _selector):
        return FakeLocator()

    async def evaluate(self, _script):
        return {"width": 0, "height": 0}


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
