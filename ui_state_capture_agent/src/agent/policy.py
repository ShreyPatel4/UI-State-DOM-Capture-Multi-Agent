"""Policy module remains generic with no app-specific selectors or workflows."""

import json
import re
from dataclasses import dataclass
from typing import Any, Literal, Optional, Sequence

from sqlalchemy.orm import Session
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from ..config import settings
from ..models import Flow, log_flow_event
from .dom_scanner import CandidateAction
from .llm_client import PolicyLLMClient
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
Return only a single JSON object. No prose. No lead-in sentences. No markdown. No code fences.

You will receive two groups of candidate UI elements:
- clickable_ui: elements meant for clicks (navigation, confirmation, opening dialogs)
- text_entry_ui: elements meant for typing (form fields, text editors)

Each candidate includes id, tag, role, description, semantics, and other DOM hints. You must base all decisions only on these candidates and the user goal. Do not assume any app-specific behavior.

Rules:
- Use clickable_ui to navigate, open modals, confirm actions, and similar behaviors.
- Use text_entry_ui for typing into title, name, description, or rich-text fields.
- If the goal contains quoted phrases (e.g., "example text") and a text entry candidate has semantics such as "title_field" or its description suggests a name/title field, choose action_type="type" on that candidate with text_to_type exactly set to the quoted phrase.
- action_id must be one of the provided candidate ids or null.
- If action_type="type" is chosen, text_to_type must be a non-empty string.

JSON schema:
{
  "action_id": "<one of the provided candidate ids or null>",
  "action_type": "click" or "type",
  "text_to_type": "<text to type>" or null,
  "capture": true or false,
  "done": true or false,
  "notes": "<short explanation or empty string>"
}
"""


def build_policy_prompt(
    task: TaskSpec,
    app_name: str,
    url: str,
    history_summary: str,
    candidates: Sequence[CandidateAction],
) -> str:
    click_candidates = [c for c in candidates if c.action_type == "click"]
    type_candidates = [c for c in candidates if c.action_type == "type"]

    def fmt(c: CandidateAction) -> str:
        semantics = f"[{', '.join(sorted(c.semantics))}]" if c.semantics else "[]"
        return (
            f"id: {c.id}, tag: {c.tag or '-'}, role: {c.role or '-'}, "
            f"semantics: {semantics}, description: \"{c.description}\""
        )

    quoted_phrases = re.findall(r'"([^"]+)"', task.goal)

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
    lines.append("Clickable UI elements:")
    if click_candidates:
        for cand in click_candidates:
            lines.append(f"  - {fmt(cand)}")
    else:
        lines.append("  - (none)")
    lines.append("")
    lines.append("Text entry UI elements:")
    if type_candidates:
        for cand in type_candidates:
            lines.append(f"  - {fmt(cand)}")
    else:
        lines.append("  - (none)")
    lines.append("")
    lines.append(f"Quoted phrases in goal: {quoted_phrases if quoted_phrases else '[]'}")
    lines.append("")
    lines.append("You must pick action_id from the candidate ids listed above or null.")
    lines.append("If you choose action_type='type', you must also set text_to_type to a non-empty string.")
    lines.append(
        "Respond with exactly one JSON object and nothing else. Do not use markdown or code fences."
    )
    return "\n".join(lines)


def _extract_json(raw_text: str) -> tuple[dict | None, str | None]:
    """Robustly extract a JSON object from LLM chatter without raising."""

    if raw_text is None or not str(raw_text).strip():
        return None, "empty_output"

    cleaned = str(raw_text).strip()
    decode_error = ""

    fence_pattern = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)
    fence_match = fence_pattern.match(cleaned)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj, None
        return None, "json_not_object"
    except json.JSONDecodeError as exc:  # noqa: BLE001
        decode_error = exc.msg
    else:
        return None, "json_not_object"

    last_open = cleaned.rfind("{")
    last_close = cleaned.rfind("}")
    if last_open < 0 or last_close < 0 or last_close < last_open:
        return None, "no_brace_block_found"

    candidate = cleaned[last_open : last_close + 1].strip()
    candidate_match = fence_pattern.match(candidate)
    if candidate_match:
        candidate = candidate_match.group(1).strip()

    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj, None
        return None, "json_not_object"
    except json.JSONDecodeError as exc:  # noqa: BLE001
        return None, f"json_decode_error:{exc.msg}"
    except Exception:
        return None, f"json_decode_error:{decode_error}"


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


def _validate_and_normalize_decision(
    *,
    obj,
    candidates: Sequence[CandidateAction],
    flow: Flow | None,
    db_session: Session | None,
    step_index: int | None,
) -> PolicyDecision:
    candidate_ids = {c.id for c in candidates}
    type_candidates = [c for c in candidates if c.action_type == "type"]
    type_ids = {c.id for c in type_candidates}

    action_id = obj.get("action_id")
    if action_id is None and "id" in obj:
        action_id = obj.get("id")

    action_type = (obj.get("action_type") or "click").lower()
    text_to_type = obj.get("text_to_type")
    capture = bool(obj.get("capture", True))
    done = bool(obj.get("done", False))
    notes = obj.get("notes") or ""

    def log(level: str, message: str) -> None:
        if db_session and flow:
            log_flow_event(
                db_session,
                flow,
                level,
                f"{message} step={step_index if step_index is not None else '?'}",
            )

    if action_id is not None and action_id not in candidate_ids:
        log("warning", f"policy_invalid_action_id step={step_index} action_id={action_id}")
        action_id = None

    if action_type not in ("click", "type"):
        log("warning", f"policy_invalid_action_type step={step_index} type={action_type}")
        action_type = "click"

    if action_type == "type":
        if not text_to_type or str(text_to_type).strip() == "":
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
                notes="fallback_due_to_type_without_candidates",
            )
        elif action_id is not None and action_id not in type_ids:
            log("warning", "policy_type_action_on_non_type_candidate")
            action_id = next(iter(type_ids))

    if action_id is None and not done and candidates:
        fallback = candidates[0]
        log(
            "info",
            f"policy_missing_action_id_using_first_candidate id={fallback.id}",
        )
        action_id = fallback.id

    return PolicyDecision(
        action_id=action_id,
        action_type=action_type,
        text_to_type=text_to_type if isinstance(text_to_type, str) else None,
        capture=capture,
        done=done,
        notes=str(notes),
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
    try:
        raw = llm.generate_text(prompt)
    except Exception as exc:  # noqa: BLE001
        if session and flow:
            log_flow_event(
                session,
                flow,
                "error",
                f"policy_llm_exception step={step_index} msg={repr(exc)}",
            )
        return PolicyDecision(
            action_id=None,
            action_type="click",
            text_to_type=None,
            capture=True,
            done=True,
            notes=f"fallback_due_to_llm_exception:{exc}",
        )

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
        obj=parsed,
        candidates=candidates,
        flow=flow,
        db_session=session,
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

        return _validate_and_normalize_decision(
            obj=parsed,
            candidates=candidates,
            flow=None,
            db_session=None,
            step_index=None,
        )
