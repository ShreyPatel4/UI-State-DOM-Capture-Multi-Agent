from src.agent.dom_scanner import (
    _prepare_goal_tokens,
    _goal_contains_concrete_name,
    _scan_click_candidates_from_snapshot,
    _scan_text_candidates_from_snapshot,
)
from src.agent.page_snapshot import AXNode, PageSnapshot, SnapshotNode


def test_snapshot_helpers_produce_candidates():
    dom_nodes = [
        SnapshotNode(index=0, node_name="button", attributes={"aria-label": "Submit"}, text_snippet="Submit"),
        SnapshotNode(index=1, node_name="input", attributes={"placeholder": "Name"}, text_snippet=None),
    ]
    ax_nodes = [
        AXNode(node_id="1", role="button", name="Submit", dom_node_indices=[0]),
        AXNode(node_id="2", role="textbox", name="Name", dom_node_indices=[1]),
    ]
    snapshot = PageSnapshot(dom_nodes=dom_nodes, ax_nodes=ax_nodes)
    goal = "submit the name"
    goal_tokens = _prepare_goal_tokens(goal)

    click_candidates = _scan_click_candidates_from_snapshot(snapshot, goal_tokens)
    text_candidates = _scan_text_candidates_from_snapshot(
        snapshot, goal_tokens, _goal_contains_concrete_name(goal)
    )

    assert any(c.action_type == "click" for c in click_candidates)
    assert any(c.action_type == "type" for c in text_candidates)
    form_candidates = [c for c in text_candidates if getattr(c, "is_form_field", False)]
    assert form_candidates, "expected at least one form field candidate"
    assert form_candidates[0].is_form_field is True
