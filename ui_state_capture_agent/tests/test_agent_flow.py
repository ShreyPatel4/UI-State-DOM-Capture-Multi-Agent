import asyncio
import uuid
from datetime import datetime, timezone
from typing import List

from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from src.agent.agent_loop import run_agent_loop
from src.agent.dom_scanner import CandidateAction
from src.agent.policy import PolicyDecision
from src.models import FlowLog


class DummySession:
    def __init__(self, flow=None):
        self.flow = flow
        self.logs: List[FlowLog] = []

    def add(self, obj):
        if isinstance(obj, FlowLog):
            self.logs.append(obj)
        return None

    def commit(self):
        return None

    def get(self, _model, ident):
        if self.flow and self.flow.id == ident:
            return self.flow
        return None

    def refresh(self, obj, *_args, **_kwargs):
        return obj


class FakeCaptureManager:
    def __init__(self, session: DummySession):
        self.db_session = session
        self.storage = None
        self.steps: list = []

    async def get_dom_snapshot(self, page):
        return await page.content()

    async def capture_step(
        self,
        page,
        flow,
        label,
        dom_html,
        diff_summary,
        diff_score,
        action_description,
        url_changed,
        state_kind,
        description=None,
        step_index=None,
    ):
        idx = step_index or len(self.steps) + 1
        step = {
            "step_index": idx,
            "state_label": label,
            "url": page.url,
            "url_changed": url_changed,
            "state_kind": state_kind,
            "diff_score": diff_score,
            "action_description": action_description,
        }
        self.steps.append(step)
        return step

    def finish_flow(self, flow, status: str):
        flow.status = status
        flow.finished_at = datetime.now(timezone.utc)


class FakeLocator:
    def __init__(self, page, locator: str):
        self.page = page
        self.locator = locator

    async def is_visible(self):
        return True

    async def click(self, timeout: int = 2000):
        if self.locator == "timeout":
            raise PlaywrightTimeoutError("timeout")
        self.page.apply_action(self.locator)

    async def fill(self, text: str, timeout: int | None = None):  # noqa: ARG002
        self.page.apply_action(self.locator, text)

    async def type(self, text: str, timeout: int | None = None):  # noqa: ARG002
        self.page.apply_action(self.locator, text)


class FakePage:
    def __init__(self):
        self.url = "https://example.com/start"
        self.state = 0
        self.typed: list[str] = []

    def locator(self, locator: str):
        return FakeLocator(self, locator)

    async def wait_for_timeout(self, _ms: int):
        return None

    def apply_action(self, locator: str, text: str | None = None):
        if locator == "no_change":
            return
        if locator == "url_change":
            self.url = "https://example.com/next"
        if locator == "url_change2":
            self.url = "https://example.com/final"
        self.state += 1
        if text:
            self.typed.append(text)

    async def content(self):
        filler = "x" * (self.state * 50)
        return f"<html><body><div>{filler}</div></body></html>"

    async def screenshot(self, full_page: bool = True):
        return b""


class FakeBrowserSession:
    def __init__(self, page: FakePage):
        self.page = page

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False

    async def goto(self, url: str, wait_ms: int = 1500):
        self.page.url = url


def make_decision(action_id: str, text: str | None = None) -> PolicyDecision:
    return PolicyDecision(
        action_id=action_id,
        action_type="click" if text is None else "type",
        text_to_type=text,
        done=False,
        capture_before=False,
        capture_after=True,
        label=f"after_action_{action_id}",
        reason="test decision",
        should_capture=True,
    )


def test_respects_max_steps(monkeypatch):
    flow = type("Flow", (), {"id": uuid.uuid4(), "cancel_requested": False, "status": "running", "status_reason": None})()
    session = DummySession(flow)
    capture_manager = FakeCaptureManager(session)
    page = FakePage()
    browser = FakeBrowserSession(page)

    decisions = [make_decision("url_change"), make_decision("url_change")]

    async def fake_scan(_page, max_actions=40, goal=None):
        candidates = [
            CandidateAction(
                id="url_change", action_type="click", locator="url_change", description="button change"
            )
        ]
        return candidates, [c.id for c in candidates if c.action_type == "type"]

    def fake_choose(*_args, **_kwargs):
        return decisions.pop(0)

    monkeypatch.setattr("src.agent.agent_loop.scan_candidate_actions", fake_scan)
    monkeypatch.setattr("src.agent.agent_loop.choose_action_with_llm", fake_choose)

    asyncio.run(
        run_agent_loop(
            task=type("Task", (), {"goal": "", "app_name": "linear"})(),
            flow=flow,
            capture_manager=capture_manager,
            hf_pipeline=None,
            start_url=page.url,
            browser_factory=lambda: browser,
            max_steps=1,
        )
    )

    assert flow.status == "max_steps_reached"
    assert capture_manager.steps[-1]["state_kind"] == "url_change"


def test_bans_repeated_failures(monkeypatch):
    flow = type("Flow", (), {"id": uuid.uuid4(), "cancel_requested": False, "status": "running", "status_reason": None})()
    session = DummySession(flow)
    capture_manager = FakeCaptureManager(session)
    page = FakePage()
    browser = FakeBrowserSession(page)

    decisions = [make_decision("no_change") for _ in range(3)]

    async def fake_scan(_page, max_actions=40, goal=None):
        candidates = [
            CandidateAction(id="no_change", action_type="click", locator="no_change", description="no change")
        ]
        return candidates, [c.id for c in candidates if c.action_type == "type"]

    def fake_choose(*_args, **_kwargs):
        return decisions.pop(0)

    monkeypatch.setattr("src.agent.agent_loop.scan_candidate_actions", fake_scan)
    monkeypatch.setattr("src.agent.agent_loop.choose_action_with_llm", fake_choose)

    asyncio.run(
        run_agent_loop(
            task=type("Task", (), {"goal": "", "app_name": "linear"})(),
            flow=flow,
            capture_manager=capture_manager,
            hf_pipeline=None,
            start_url=page.url,
            browser_factory=lambda: browser,
            max_steps=4,
        )
    )

    assert flow.status == "no_actions"
    assert any("Banned" in log.message for log in session.logs)


def test_cancel_request_stops_loop(monkeypatch):
    flow = type("Flow", (), {"id": uuid.uuid4(), "cancel_requested": True, "status": "running", "status_reason": None})()
    session = DummySession(flow)
    capture_manager = FakeCaptureManager(session)
    page = FakePage()
    browser = FakeBrowserSession(page)

    async def fake_scan(_page, max_actions=40, goal=None):
        candidates = [
            CandidateAction(
                id="url_change", action_type="click", locator="url_change", description="button change"
            )
        ]
        return candidates, [c.id for c in candidates if c.action_type == "type"]

    monkeypatch.setattr("src.agent.agent_loop.scan_candidate_actions", fake_scan)

    asyncio.run(
        run_agent_loop(
            task=type("Task", (), {"goal": "", "app_name": "linear"})(),
            flow=flow,
            capture_manager=capture_manager,
            hf_pipeline=None,
            start_url=page.url,
            browser_factory=lambda: browser,
            max_steps=2,
        )
    )

    assert flow.status == "cancelled"
    assert flow.status_reason == "cancel_requested"


def test_steps_record_url_and_state(monkeypatch):
    flow = type("Flow", (), {"id": uuid.uuid4(), "cancel_requested": False, "status": "running", "status_reason": None})()
    session = DummySession(flow)
    capture_manager = FakeCaptureManager(session)
    page = FakePage()
    browser = FakeBrowserSession(page)

    decisions = [
        make_decision("url_change"),
        PolicyDecision(
            action_id="url_change2",
            action_type="click",
            text_to_type=None,
            done=True,
            capture_before=False,
            capture_after=True,
            label="done",
            reason="finish",
            should_capture=True,
        ),
    ]

    async def fake_scan(_page, max_actions=40, goal=None):
        candidates = [
            CandidateAction(id="url_change", action_type="click", locator="url_change", description="button change"),
            CandidateAction(id="url_change2", action_type="click", locator="url_change2", description="finish"),
        ]
        return candidates, [c.id for c in candidates if c.action_type == "type"]

    def fake_choose(*_args, **_kwargs):
        return decisions.pop(0)

    monkeypatch.setattr("src.agent.agent_loop.scan_candidate_actions", fake_scan)
    monkeypatch.setattr("src.agent.agent_loop.choose_action_with_llm", fake_choose)

    asyncio.run(
        run_agent_loop(
            task=type("Task", (), {"goal": "", "app_name": "linear"})(),
            flow=flow,
            capture_manager=capture_manager,
            hf_pipeline=None,
            start_url=page.url,
            browser_factory=lambda: browser,
            max_steps=3,
        )
    )

    url_change_steps = [s for s in capture_manager.steps if s["state_kind"] == "url_change"]
    assert url_change_steps
    assert url_change_steps[0]["url_changed"] is True


def test_type_action_executes_and_finishes(monkeypatch):
    flow = type("Flow", (), {"id": uuid.uuid4(), "cancel_requested": False, "status": "running", "status_reason": None})()
    session = DummySession(flow)
    capture_manager = FakeCaptureManager(session)
    page = FakePage()
    browser = FakeBrowserSession(page)

    decisions = [
        PolicyDecision(
            action_id="input_0",
            action_type="type",
            text_to_type="Hello world",
            done=True,
            capture_before=False,
            capture_after=True,
            label="typed",
            reason="finish typing",
            should_capture=True,
        )
    ]
    call_count = {"count": 0}

    async def fake_scan(_page, max_actions=40, goal=None):  # noqa: ARG001
        candidates = [
            CandidateAction(
                id="input_0", action_type="type", locator="type_input", description="Text input labeled 'Title'"
            )
        ]
        return candidates, [c.id for c in candidates if c.action_type == "type"]

    def fake_choose(*_args, **_kwargs):  # noqa: ARG001
        call_count["count"] += 1
        return decisions[min(call_count["count"] - 1, len(decisions) - 1)]

    monkeypatch.setattr("src.agent.agent_loop.scan_candidate_actions", fake_scan)
    monkeypatch.setattr("src.agent.agent_loop.choose_action_with_llm", fake_choose)

    asyncio.run(
        run_agent_loop(
            task=type("Task", (), {"goal": "type a title", "app_name": "linear"})(),
            flow=flow,
            capture_manager=capture_manager,
            hf_pipeline=None,
            start_url=page.url,
            browser_factory=lambda: browser,
            max_steps=2,
        )
    )

    assert "Hello world" in page.typed
    assert any("Hello world" in step["action_description"] for step in capture_manager.steps if step["state_label"] != "initial_state")
    assert flow.status_reason == "goal_reached"
    assert any("typed value='Hello world'" in log.message for log in session.logs)
    assert call_count["count"] == 1


def test_done_without_change_marks_uncertain(monkeypatch):
    flow = type("Flow", (), {"id": uuid.uuid4(), "cancel_requested": False, "status": "running", "status_reason": None})()
    session = DummySession(flow)
    capture_manager = FakeCaptureManager(session)
    page = FakePage()
    browser = FakeBrowserSession(page)

    decisions = [
        PolicyDecision(
            action_id="no_change",
            action_type="click",
            text_to_type=None,
            done=True,
            capture_before=False,
            capture_after=True,
            label="done",
            reason="finish",
            should_capture=True,
        )
    ]

    async def fake_scan(_page, max_actions=40, goal=None):
        candidates = [CandidateAction(id="no_change", action_type="click", locator="no_change", description="noop")]
        return candidates, [c.id for c in candidates if c.action_type == "type"]

    def fake_choose(*_args, **_kwargs):
        return decisions.pop(0)

    monkeypatch.setattr("src.agent.agent_loop.scan_candidate_actions", fake_scan)
    monkeypatch.setattr("src.agent.agent_loop.choose_action_with_llm", fake_choose)

    asyncio.run(
        run_agent_loop(
            task=type("Task", (), {"goal": "create issue named test", "app_name": "linear"})(),
            flow=flow,
            capture_manager=capture_manager,
            hf_pipeline=None,
            start_url=page.url,
            browser_factory=lambda: browser,
            max_steps=2,
        )
    )

    assert flow.status_reason == "uncertain_goal"
    assert any(step["state_label"] != "initial_state" for step in capture_manager.steps)


def test_capture_goal_can_finish_after_initial(monkeypatch):
    flow = type("Flow", (), {"id": uuid.uuid4(), "cancel_requested": False, "status": "running", "status_reason": None})()
    session = DummySession(flow)
    capture_manager = FakeCaptureManager(session)
    page = FakePage()
    browser = FakeBrowserSession(page)

    decisions = [
        PolicyDecision(
            action_id="no_change",
            action_type="click",
            text_to_type=None,
            done=True,
            capture_before=False,
            capture_after=True,
            label="capture_done",
            reason="capture",
            should_capture=True,
        )
    ]

    async def fake_scan(_page, max_actions=40, goal=None):
        candidates = [CandidateAction(id="no_change", action_type="click", locator="no_change", description="noop")]
        return candidates, [c.id for c in candidates if c.action_type == "type"]

    def fake_choose(*_args, **_kwargs):
        return decisions.pop(0)

    monkeypatch.setattr("src.agent.agent_loop.scan_candidate_actions", fake_scan)
    monkeypatch.setattr("src.agent.agent_loop.choose_action_with_llm", fake_choose)

    asyncio.run(
        run_agent_loop(
            task=type("Task", (), {"goal": "open home page and capture one screenshot", "app_name": "linear"})(),
            flow=flow,
            capture_manager=capture_manager,
            hf_pipeline=None,
            start_url=page.url,
            browser_factory=lambda: browser,
            max_steps=1,
        )
    )

    assert flow.status_reason == "goal_reached"
    assert len(capture_manager.steps) >= 2


def test_llm_fallback_still_captures(monkeypatch):
    flow = type("Flow", (), {"id": uuid.uuid4(), "cancel_requested": False, "status": "running", "status_reason": None})()
    session = DummySession(flow)
    capture_manager = FakeCaptureManager(session)
    page = FakePage()
    browser = FakeBrowserSession(page)

    decisions = [
        PolicyDecision(
            action_id=None,
            action_type="click",
            text_to_type=None,
            done=True,
            capture_before=False,
            capture_after=True,
            label="fallback",
            reason="bad json",
            should_capture=True,
        )
    ]

    async def fake_scan(_page, max_actions=40, goal=None):
        candidates = [
            CandidateAction(
                id="url_change", action_type="click", locator="url_change", description="button change"
            )
        ]
        return candidates, [c.id for c in candidates if c.action_type == "type"]

    def fake_choose(*_args, **_kwargs):
        return decisions.pop(0)

    monkeypatch.setattr("src.agent.agent_loop.scan_candidate_actions", fake_scan)
    monkeypatch.setattr("src.agent.agent_loop.choose_action_with_llm", fake_choose)

    asyncio.run(
        run_agent_loop(
            task=type("Task", (), {"goal": "", "app_name": "linear"})(),
            flow=flow,
            capture_manager=capture_manager,
            hf_pipeline=None,
            start_url=page.url,
            browser_factory=lambda: browser,
            max_steps=1,
        )
    )

    assert flow.status_reason == "llm_fallback"
    assert len(capture_manager.steps) >= 2
