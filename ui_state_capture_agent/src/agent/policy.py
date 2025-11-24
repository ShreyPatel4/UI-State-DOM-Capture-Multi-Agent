import json
import re
from dataclasses import dataclass
from typing import Any, Literal, Optional, Sequence

from sqlalchemy.orm import Session
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from ..config import settings
from ..models import Flow, log_flow_event
from .dom_scanner import CandidateAction
from .task_spec import TaskSpec


@dataclass
class PolicyDecision:
    action_id: Optional[str]
    action_type: Literal["click", "type"]
    text_to_type: Optional[str]
    capture: bool = True
    done: bool = False
    notes: str = ""
    capture_before: bool = False
    capture_after: bool = True
    label: str = ""
    reason: str = ""
    should_capture: bool = True


POLICY_SYSTEM_PROMPT = """
You are a deterministic UI policy that selects exactly one action for the next step.
Return exactly one JSON object and nothing else. No natural language. No markdown. No code fences.

JSON schema:
{
  "action_id": "<one of the provided candidate ids or null>",
  "action_type": "click" or "type",
  "text_to_type": "<text to type>" or null,
  "capture": true or false,
  "done": true or false,
  "notes": "<short free text or empty string>"
}
"""


def build_policy_prompt(
    task: TaskSpec,
    app_name: str,
    url: str,
    history_summary: str,
    candidates: Sequence[CandidateAction],
) -> str:
    lines: list[str] = []
    lines.append(POLICY_SYSTEM_PROMPT.strip())
    lines.append("")
    lines.append("User goal:")
    lines.append(task.goal)
    lines.append("")
    lines.append(f"App name: {app_name}")
    lines.append(f"Current URL: {url}")
    lines.append("")
    lines.append("History summary:")
    lines.append(history_summary if history_summary else "(no previous actions)")
    lines.append("")
    lines.append("Candidate actions (choose one id):")
    for cand in candidates:
        lines.append(
            f"  - id={cand.id} type={cand.action_type} description={cand.description}"
        )
    lines.append("")
    lines.append(
        "Respond with exactly one JSON object and nothing else. Do not use markdown or code fences."
    )
    return "\n".join(lines)


def _extract_json(text: str) -> tuple[dict | None, str | None]:
    """Robustly extract a JSON object from LLM chatter without raising."""

    if text is None or not str(text).strip():
        return None, "empty_output"

    cleaned = str(text).strip()

    fence_pattern = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)
    match = fence_pattern.match(cleaned)
    if match:
        cleaned = match.group(1).strip()

    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj, None
        return None, "json_not_object"
    except json.JSONDecodeError:
        pass

    last_open = cleaned.rfind("{")
    last_close = cleaned.rfind("}")
    if last_open < 0 or last_close < 0 or last_close < last_open:
        return None, "no_brace_block_found"

    snippet = cleaned[last_open : last_close + 1]
    try:
        obj = json.loads(snippet)
        if isinstance(obj, dict):
            return obj, None
        return None, "json_not_object"
    except json.JSONDecodeError as exc:  # noqa: BLE001
        return None, f"json_decode_error:{exc.msg}"
    except Exception:
        return None, "json_parse_exception"


def choose_fallback_action(goal: str, candidates: Sequence[CandidateAction]) -> CandidateAction:
    def score(cand: CandidateAction) -> int:
        g = set(goal.lower().split())
        d = set(cand.description.lower().split())
        return len(g & d)

    return max(candidates, key=score)


def create_policy_hf_pipeline(model_name: str | None = None) -> Any:
    model_name = model_name or settings.hf_model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device="cpu",
        max_new_tokens=128,
        do_sample=False,
    )


class PolicyLLMClient:
    def __init__(self, hf_pipeline: Any) -> None:
        self.hf_pipeline = hf_pipeline

    def generate(self, prompt: str) -> str:
        return self.hf_pipeline(
            prompt, num_return_sequences=1, return_full_text=False
        )[0]["generated_text"]

    def complete(self, prompt: str) -> str:
        return self.generate(prompt)


def _validate_and_normalize_decision(
    data: dict,
    candidates: Sequence[CandidateAction],
    *,
    session: Session | None = None,
    flow: Flow | None = None,
    step_index: int | None = None,
) -> PolicyDecision:
    candidate_ids = {c.id for c in candidates}
    type_candidates = [c for c in candidates if c.action_type == "type"]
    type_candidate_ids = {c.id for c in type_candidates}

    action_id = data.get("action_id")
    action_type = (data.get("action_type") or "click").lower()
    text_to_type = data.get("text_to_type")
    capture = bool(data.get("capture", True))
    done = bool(data.get("done", False))
    notes = str(data.get("notes") or "").strip()

    def log(level: str, message: str) -> None:
        if session and flow:
            log_flow_event(
                session,
                flow,
                level,
                f"{message} step={step_index if step_index is not None else '?'}",
            )

    if action_id is not None and action_id not in candidate_ids:
        log("warning", f"policy_invalid_action_id id={action_id}")
        action_id = None

    if action_type == "type":
        if not (isinstance(text_to_type, str) and text_to_type.strip()):
            log("warning", "policy_type_without_text")
            action_type = "click"
            text_to_type = None
        elif not type_candidates:
            log("warning", "policy_requested_type_but_no_type_candidates")
            return PolicyDecision(
                action_id=None,
                action_type="click",
                text_to_type=None,
                capture=True,
                done=True,
                notes="fallback: no type candidates",
            )
        elif action_id is not None and action_id not in type_candidate_ids:
            log("warning", "policy_type_action_on_non_type_candidate")
            action_id = type_candidates[0].id
    elif action_type != "click":
        log("warning", f"policy_invalid_action_type type={action_type}")
        action_type = "click"
        text_to_type = None

    if action_id is None and not done and candidates:
        action_id = candidates[0].id
        log("info", f"policy_missing_action_id_using_first_candidate id={action_id}")

    return PolicyDecision(
        action_id=action_id,
        action_type=action_type if action_type in {"click", "type"} else "click",
        text_to_type=text_to_type if isinstance(text_to_type, str) else None,
        capture=capture,
        done=done,
        notes=notes,
    )


def choose_action_with_llm(
    llm: PolicyLLMClient,
    task: TaskSpec,
    app_name: str,
    url: str,
    history_summary: str,
    candidates: Sequence[CandidateAction],
    session: Session | None = None,
    flow: Flow | None = None,
    step_index: int | None = None,
) -> PolicyDecision:
    prompt = build_policy_prompt(task, app_name, url, history_summary, candidates)
    raw = llm.complete(prompt)

    if session and flow:
        raw_excerpt = (raw or "")[:500].replace("\n", " ")
        log_flow_event(
            session,
            flow,
            "debug",
            f"policy_raw_output step={step_index} text={raw_excerpt}",
        )

    parsed, reason = _extract_json(raw)
    if parsed is None:
        if session and flow:
            raw_head = (raw or "")[:120].replace("\n", " ")
            log_flow_event(
                session,
                flow,
                "warning",
                f"policy_parse_failure step={step_index} reason={reason} head={raw_head}",
            )
        return PolicyDecision(
            action_id=None,
            action_type="click",
            text_to_type=None,
            capture=True,
            done=True,
            notes=f"fallback_due_to_parse_failure:{reason}",
        )

    decision = _validate_and_normalize_decision(
        parsed,
        candidates,
        session=session,
        flow=flow,
        step_index=step_index,
    )

    if session and flow:
        log_flow_event(
            session,
            flow,
            "info",
            "policy_decision step={step} action_id={aid} type={atype} capture={cap} done={done}"
            .format(
                step=step_index,
                aid=decision.action_id,
                atype=decision.action_type,
                cap=decision.capture,
                done=decision.done,
            ),
        )

    return decision


class Policy:
    """LLM backed policy that chooses the next UI action."""

    def __init__(self, model_name: str | None = None) -> None:
        model_name = model_name or settings.hf_model_name
        self.generator = create_policy_hf_pipeline(model_name)

    def _run_hf(self, prompt: str) -> str:
        out = self.generator(
            prompt,
            num_return_sequences=1,
            return_full_text=False,
        )[0]["generated_text"]
        return out

    async def choose_action(
        self,
        task: TaskSpec,
        candidates: Sequence[CandidateAction],
        history_summary: str,
        url: str,
    ) -> PolicyDecision:
        prompt = build_policy_prompt(
            task=task,
            app_name=task.app_name,
            url=url,
            history_summary=history_summary,
            candidates=candidates,
        )
        raw = self._run_hf(prompt)
        parsed, reason = _extract_json(raw)
        if parsed is None:
            fallback = choose_fallback_action(task.goal, candidates)
            return PolicyDecision(
                action_id=None,
                action_type=fallback.action_type,
                text_to_type=None,
                capture=True,
                done=True,
                notes=f"fallback_due_to_parse_failure:{reason}",
            )

        return _validate_and_normalize_decision(parsed, candidates)
