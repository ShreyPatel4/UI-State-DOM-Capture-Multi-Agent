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
- If action_type == "type", you MUST set text_to_type to a non empty string.
- If done == true, action_id MUST be null.
- If action_id is null and done == false, that means "no suitable action" and the controller will stop.
- Base every decision strictly on the provided candidates and the user goal. Do not assume any app-specific behaviors.

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
) -> str:
    def fmt(c: CandidateAction) -> str:
        kind = "primary_cta" if c.is_primary_cta else "nav_link" if c.is_nav_link else "form_field" if c.is_form_field else (
            c.kind or c.tag or "-"
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
            log("warning", f"policy_type_missing_text step={step_index}")
            return PolicyDecision(
                action_id=None,
                action_type="click",
                text_to_type=None,
                capture=True,
                done=True,
                notes="fallback_type_missing_text",
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
