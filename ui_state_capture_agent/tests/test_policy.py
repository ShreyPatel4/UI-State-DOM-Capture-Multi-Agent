import uuid

from src.agent.dom_scanner import CandidateAction
from src.agent.policy import (
    PolicyDecision,
    build_policy_prompt,
    _extract_json,
    _validate_and_normalize_decision,
    choose_action_with_llm,
)
from src.agent.task_spec import TaskSpec
from src.models import FlowLog


class DummyLLM:
    def __init__(self, output: str, should_raise: bool = False):
        self.output = output
        self.should_raise = should_raise

    def generate_text(self, prompt: str) -> str:  # noqa: ARG002
        if self.should_raise:
            raise RuntimeError("boom")
        return self.output


class DummySession:
    def __init__(self, flow=None):
        self.logs = []
        self.flow = flow

    def add(self, obj):
        if isinstance(obj, FlowLog):
            self.logs.append(obj)

    def commit(self):
        return None

    def refresh(self, obj, *_args, **_kwargs):
        return obj

    def get(self, *_args, **_kwargs):
        return self.flow


def test_extract_json_variants():
    assert _extract_json('{"a": 1}')[0] == {"a": 1}
    fenced = """```json\n{\n  \"a\": 2\n}\n```"""
    assert _extract_json(fenced)[0] == {"a": 2}
    noisy = "Here is the result: {\"a\":3} and some trailing text"
    assert _extract_json(noisy)[0] == {"a": 3}
    empty, reason = _extract_json("")
    assert empty is None and reason == "empty_output"


def test_choose_action_valid_json():
    candidates = [CandidateAction(id="btn_0", action_type="click", locator="locator", description="button A")]
    llm = DummyLLM('{"action_id": "btn_0", "action_type": "click", "done": false}')
    task = TaskSpec(original_query="", app_name="linear", goal="go", start_url="http://example.com")

    decision = choose_action_with_llm(
        llm,
        task,
        task.app_name,
        "http://example.com",
        "",
        candidates,
    )

    assert isinstance(decision, PolicyDecision)
    assert decision.action_id == "btn_0"
    assert decision.text_to_type is None


def test_choose_action_with_text_to_type():
    candidates = [CandidateAction(id="input_1", action_type="type", locator="locator", description="Text input labeled 'Title'")]
    llm = DummyLLM('{"action_id": "input_1", "action_type": "type", "text_to_type": "Some value", "done": false}')
    task = TaskSpec(original_query="", app_name="linear", goal="go", start_url="http://example.com")

    decision = choose_action_with_llm(
        llm,
        task,
        task.app_name,
        "http://example.com",
        "",
        candidates,
    )

    assert decision.text_to_type == "Some value"


def test_choose_action_logs_decision_with_text():
    candidates = [CandidateAction(id="input_1", action_type="type", locator="locator", description="type issue title")]
    llm = DummyLLM('{"action_id": "input_1", "action_type": "type", "text_to_type": "Title here", "done": false}')
    task = TaskSpec(original_query="", app_name="linear", goal="create issue named 'Title here'", start_url="http://example.com")
    flow = type("Flow", (), {"id": uuid.uuid4()})()
    session = DummySession(flow)

    decision = choose_action_with_llm(
        llm,
        task,
        task.app_name,
        "http://example.com",
        "",
        candidates,
        session=session,
        flow=flow,
        step_index=2,
    )

    assert decision.text_to_type == "Title here"
    assert any("policy_decision" in log.message for log in session.logs)


def test_choose_action_fallback_logs_warning():
    candidates = [CandidateAction(id="btn_0", action_type="click", locator="locator", description="button A")]
    llm = DummyLLM("nonsense without json")
    task = TaskSpec(original_query="", app_name="linear", goal="go", start_url="http://example.com")
    flow = type("Flow", (), {"id": uuid.uuid4()})()
    session = DummySession(flow)

    decision = choose_action_with_llm(
        llm,
        task,
        task.app_name,
        "http://example.com",
        "",
        candidates,
        session=session,
        flow=flow,
    )

    assert decision.action_id is None
    assert decision.done is True
    assert session.logs
    assert any("policy_parse_failure" in log.message for log in session.logs)


def test_choose_action_invalid_json_sets_text_none():
    candidates = [CandidateAction(id="btn_0", action_type="click", locator="locator", description="button A")]
    llm = DummyLLM("{not json}")
    task = TaskSpec(original_query="", app_name="linear", goal="go", start_url="http://example.com")

    decision = choose_action_with_llm(
        llm,
        task,
        task.app_name,
        "http://example.com",
        "",
        candidates,
    )

    assert decision.text_to_type is None


def test_build_policy_prompt_highlights_form_fields():
    candidates = [
        CandidateAction(
            id="input_1",
            action_type="type",
            locator="input",
            description="title field",
            visible_text="Title",
            is_form_field=True,
            goal_match_score=2.0,
        ),
        CandidateAction(
            id="btn_0",
            action_type="click",
            locator="btn",
            description="Create",
            is_primary_cta=True,
            goal_match_score=1.0,
        ),
    ]
    task = TaskSpec(
        original_query="",
        app_name="linear",
        goal='create issue named "Example title"',
        start_url="http://example.com",
    )

    prompt = build_policy_prompt(task, task.app_name, task.start_url, "", candidates)

    assert "form_field" in prompt
    assert "action_type='type'" in prompt
    assert "title" in prompt.lower()
    assert "goal_match_score=2.00" in prompt


def test_validate_alias_id_preserved():
    candidates = [
        CandidateAction(id="btn_0", action_type="click", locator="loc", description="button"),
        CandidateAction(id="btn_2", action_type="click", locator="loc2", description="create"),
    ]
    decision = _validate_and_normalize_decision(
        obj={"id": "btn_2", "action_type": "click"},
        candidates=candidates,
        flow=None,
        db_session=None,
        step_index=1,
    )
    assert decision.action_id == "btn_2"
    assert decision.action_type == "click"


def test_validate_type_action_keeps_text():
    candidates = [CandidateAction(id="input_0", action_type="type", locator="loc", description="type title")]
    decision = _validate_and_normalize_decision(
        obj={"action_id": "input_0", "action_type": "type", "text_to_type": "Hello"},
        candidates=candidates,
        flow=None,
        db_session=None,
        step_index=1,
    )
    assert decision.action_type == "type"
    assert decision.text_to_type == "Hello"


def test_invalid_action_id_returns_fallback():
    candidates = [CandidateAction(id="btn_0", action_type="click", locator="loc", description="button")]
    decision = _validate_and_normalize_decision(
        obj={"action_id": "missing", "action_type": "click", "done": False},
        candidates=candidates,
        flow=None,
        db_session=None,
        step_index=1,
    )
    assert decision.action_id is None
    assert decision.done is True
    assert decision.notes == "fallback_invalid_action_id"


def test_null_action_id_not_done_returns_fallback():
    candidates = [CandidateAction(id="btn_0", action_type="click", locator="loc", description="button")]
    decision = _validate_and_normalize_decision(
        obj={"action_id": None, "action_type": "click", "done": False},
        candidates=candidates,
        flow=None,
        db_session=None,
        step_index=2,
    )
    assert decision.action_id is None
    assert decision.done is True
    assert decision.notes == "fallback_null_action_id_not_done"


def test_type_missing_text_returns_fallback():
    candidates = [CandidateAction(id="input_0", action_type="type", locator="loc", description="type title")]
    decision = _validate_and_normalize_decision(
        obj={"action_id": "input_0", "action_type": "type", "text_to_type": "   "},
        candidates=candidates,
        flow=None,
        db_session=None,
        step_index=3,
    )
    assert decision.action_id is None
    assert decision.done is True
    assert decision.notes == "fallback_type_missing_text"


def test_llm_exception_falls_back():
    candidates = [CandidateAction(id="btn_0", action_type="click", locator="locator", description="button A")]
    llm = DummyLLM("", should_raise=True)
    task = TaskSpec(original_query="", app_name="linear", goal="go", start_url="http://example.com")

    decision = choose_action_with_llm(
        llm,
        task,
        task.app_name,
        "http://example.com",
        "",
        candidates,
    )

    assert decision.action_id is None
    assert decision.done is True
