"""Policy module remains generic with no app-specific selectors or workflows."""

import json
import re
from dataclasses import dataclass
from typing import Any, Literal, Optional, Sequence

from sqlalchemy.orm import Session
from .llm_client import PolicyLLMClient, create_text_generation_pipeline
from ..config import settings
from ..models import Flow, log_flow_event
from .dom_scanner import CandidateAction
from .task_spec import TaskSpec


@dataclass
class PolicyDecision:
    action_id: Optional[str]
    action_type: Literal["click", "type", "none"]
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
You are a deterministic UI policy that selects exactly one action for the next step on the current page.

Schema (all fields are required):
{
  "action_id": "<one of the provided candidate ids or null>",
  "action_type": "click" | "type" | "none",
  "text_to_type": "<text to type>" | null,
  "capture": true | false,
  "done": true | false,
  "notes": "<short explanation or empty string>"
}

Rules:
- action_id MUST be either one of the candidate ids from the list OR null.
- Choose action_type "type" ONLY when the provided action_id is present in type_ids.
- If action_type == "type", you MUST set text_to_type to a non empty string derived from the goal.
- Base every decision strictly on the provided candidates and the user goal. Do not assume any app-specific behaviors.

Decision rules for forms:

1. If there is a primary call to action button (for example is_primary_cta is true or semantics contains "primary" or "cta") whose visible text contains a verb like "create", "new", "add", "save", "submit", "done", "finish" or "update", and the user goal is about creating, adding, saving or finishing something, you should strongly prefer clicking that button once the key text fields are filled.

2. A key text field is one whose label or placeholder matches the goal, for example "Project name", "Title", "Issue title", "Page name" or similar. Once you have typed the requested name or title from the goal into such a field, you should NOT type into that same field again.

3. After you have filled the most relevant name or title field, your next step should usually be a primary call to action such as "Create project", "Create issue", "Save", or similar that completes the task, rather than typing the same text into more and more fields.

4. Avoid repeating the same type action that does not produce a new effect. If the last action already set the correct text and you see the same state again, choose a different action, usually a primary call to action that moves the flow forward.

- If done == true, action_id MUST be null.
- If action_id is null and done == false, that means "no suitable action" and the controller will stop.

When deciding between click and type actions:
- If the goal mentions creating, naming, or titling something (for example "create issue named X" or "new project titled Y") and there is a candidate that is a text input or form field whose label or nearby text suggests it captures that name/title, you should select that candidate with action_type="type" and set text_to_type to the relevant text from the goal.
- When you choose action_type="type", text_to_type MUST be a non-empty string taken from the user goal.
- Consider clicking any obvious "create" or "new" buttons first if required to reveal a form, then type into the appropriate text field.

Output format requirements:
- You MUST respond with a single JSON object that matches this schema.
- Do NOT include any explanation, commentary, or Markdown.
- Do NOT wrap the JSON in ```json``` fences.
- Your entire response MUST be just the JSON object.
"""


def build_policy_prompt(
    task: TaskSpec,
    app_name: str,
    url: str,
    history_summary: str,
    candidates: Sequence[CandidateAction],
    type_ids: Sequence[str] | None = None,
) -> str:
    type_ids = list(type_ids) if type_ids is not None else [
        c.id for c in candidates if c.is_form_field or c.action_type == "type"
    ]
    def fmt(c: CandidateAction) -> str:
        kind = (
            "primary_cta"
            if c.is_primary_cta
            else "nav_link"
            if c.is_nav_link
            else "form_field"
            if c.is_form_field
            else (c.kind or c.tag or "-")
        )
        visible_text = (c.visible_text or c.text or "").strip()
        section = c.section_label or "-"
        return (
            f"id={c.id} | action_type={c.action_type} | kind={kind} | tag={c.tag or '-'} | role={c.role or '-'} | "
            f"text=\"{visible_text}\" | section=\"{section}\" | primary={c.is_primary_cta} | nav={c.is_nav_link} | "
            f"form_field={c.is_form_field} | goal_match_score={c.goal_match_score:.2f}"
        )

    quoted_phrases = re.findall(r'"([^"]+)"', task.goal)

    lines: list[str] = []
    lines.append(POLICY_SYSTEM_PROMPT.strip())
    lines.append("")
    lines.append("User goal:")
    lines.append(task.goal)
    lines.append("")
    lines.append(f"App/context name (do not assume behaviors): {app_name}")
    lines.append(f"Current page URL: {url}")
    lines.append("")
    lines.append(
        "Valid type targets (type_ids): {ids}. Use action_type='type' ONLY when action_id is in this list. "
        "If type_ids is empty, choose a click or action_id null instead."
        .format(ids=type_ids if type_ids else [])
    )
    lines.append("")
    lines.append("History summary:")
    lines.append(history_summary if history_summary else "(no previous actions)")
    lines.append("")
    lines.append("Candidate actions (id, kind, text, section, scoring):")
    if candidates:
        for cand in candidates:
            lines.append(f"  - {fmt(cand)}")
    else:
        lines.append("  - (none)")
    lines.append("")
    lines.append(f"Quoted phrases in goal: {quoted_phrases if quoted_phrases else '[]'}")
    lines.append("")
    lines.append(
        "If a form_field/text input matches a title or name in the goal, choose action_type='type' and set text_to_type accordingly."
    )
    lines.append("Follow the schema and output requirements exactly. Return only the JSON object.")
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


def _best_click_candidate(candidates: Sequence[CandidateAction]) -> CandidateAction | None:
    click_candidates = [
        cand
        for cand in candidates
        if (cand.action_type == "click") or ((cand.kind or "").lower() == "click")
    ]
    if not click_candidates:
        return None

    def score(cand: CandidateAction) -> tuple[float, int, int]:
        return (
            float(cand.goal_match_score or 0.0),
            1 if cand.is_primary_cta else 0,
            0 if cand.is_nav_link else 1,
        )

    return sorted(click_candidates, key=score, reverse=True)[0]


def create_policy_hf_pipeline(model_name: str | None = None) -> Any:
    return create_text_generation_pipeline(model_name=model_name, max_new_tokens=128)

def _validate_and_normalize_decision(
    *,
    obj,
    candidates: Sequence[CandidateAction],
    flow: Flow | None,
    db_session: Session | None,
    step_index: int | None,
) -> PolicyDecision:
    candidate_ids = {c.id for c in candidates}
    candidates_by_id = {c.id: c for c in candidates}

    action_id = obj.get("action_id")
    if action_id is None and "id" in obj:
        action_id = obj.get("id")

    raw_action_type = obj.get("action_type") or "click"
    action_type = raw_action_type.lower() if isinstance(raw_action_type, str) else "click"
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
        log(
            "warning",
            f"policy_invalid_action_id step={step_index} action_id={action_id} valid_ids={len(candidate_ids)}",
        )
        return PolicyDecision(
            action_id=None,
            action_type="click",
            text_to_type=None,
            capture=True,
            done=True,
            notes="fallback_invalid_action_id",
        )

    if action_type not in {"click", "type", "none"}:
        log("warning", f"policy_invalid_action_type step={step_index} type={action_type}")
        action_type = "click"

    if done and action_id is not None:
        log("warning", f"policy_done_with_action_id step={step_index} action_id={action_id}")
        action_id = None

    if not done and action_id is None:
        log("warning", f"policy_null_action_id_not_done step={step_index}")
        return PolicyDecision(
            action_id=None,
            action_type="click",
            text_to_type=None,
            capture=True,
            done=True,
            notes="fallback_null_action_id_not_done",
        )

    if action_type == "type":
        if not text_to_type or str(text_to_type).strip() == "":
            log("warning", f"policy_invalid_type_text step={step_index} reason=empty_text")
            return PolicyDecision(
                action_id=None,
                action_type="click",
                text_to_type=None,
                capture=True,
                done=True,
                notes="fallback_type_missing_text",
            )

        candidate = candidates_by_id.get(action_id)
        supports_type = bool(
            candidate
            and (
                candidate.is_form_field
                or ((candidate.kind or "").lower() == "type" and candidate.is_form_field)
                or (candidate.action_type == "type" and candidate.is_form_field)
            )
        )
        if candidate is None or not supports_type:
            reason = "unknown_action_id" if candidate is None else "target_not_form_field"
            log(
                "warning",
                f"policy_invalid_type_target step={step_index} reason={reason} id={action_id}",
            )
            fallback_click = _best_click_candidate(candidates)
            if fallback_click:
                return PolicyDecision(
                    action_id=fallback_click.id,
                    action_type="click",
                    text_to_type=None,
                    capture=True,
                    done=False,
                    notes="Fallback click after invalid type target",
                )
            return PolicyDecision(
                action_id=None,
                action_type="click",
                text_to_type=None,
                capture=True,
                done=True,
                notes="fallback_type_invalid_target",
            )

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
    type_ids: Sequence[str] | None = None,
    session: Session | None = None,
    flow: Flow | None = None,
    step_index: int | None = None,
) -> PolicyDecision:
    prompt = build_policy_prompt(task, app_name, url, history_summary, candidates, type_ids)
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
        default_model = model_name or settings.openai_model
        self.generator = create_policy_hf_pipeline(default_model)

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
        type_ids: Sequence[str] | None = None,
    ) -> PolicyDecision:
        prompt = build_policy_prompt(
            task=task,
            app_name=task.app_name,
            url=url,
            history_summary=history_summary,
            candidates=candidates,
            type_ids=type_ids,
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
