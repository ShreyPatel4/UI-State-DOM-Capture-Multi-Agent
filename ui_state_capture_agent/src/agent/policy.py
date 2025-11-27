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


@dataclass
class PolicyInput:
    goal: str
    url: str
    candidates: list[dict[str, Any]]
    type_ids: list[str]
    step_index: int | None = None
    # new fields
    banned_action_ids: list[str] | None = None
    recent_events: list[dict[str, Any]] | None = None

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "goal": self.goal,
            "url": self.url,
            "candidates": self.candidates,
            "type_ids": self.type_ids,
        }
        if self.step_index is not None:
            payload["step_index"] = self.step_index
        if self.banned_action_ids:
            payload["banned_action_ids"] = self.banned_action_ids
        if self.recent_events:
            payload["recent_events"] = self.recent_events
        return payload


POLICY_SYSTEM_PROMPT = """ 
You are the deterministic UI policy for Agent B. Choose exactly one next step that drives the live web app toward the user goal while capturing UI states.

You receive:
- goal: the full user goal text.
- url: the current page URL.
- candidates: list of possible actions on the current page. Each candidate has:
  id, kind, text or label, semantics (for example primary, cta, danger, search_field, title_field, nav) and flags such as is_primary_cta.
- type_ids: list of candidate ids that are valid text targets for typing.
- banned_action_ids: action ids that previously failed, timed out, or produced no meaningful change.
- recent_events: short history of previous steps including action_id, effect_kind (for example url_change, dom_change_modal, no_change) and outcome (for example progress, no_progress, timeout, blocked).

You must output exactly one JSON object with all of these keys:
{ "action_id": "<candidate id or null>",
  "action_type": "click" or "type",
  "text_to_type": "<text>" or null,
  "capture": true or false,
  "done": true or false,
  "notes": "<short explanation or empty string>"
}

Core rules:
- action_id must be one of the candidate ids or null. Never invent ids or actions.
- Use action_type "type" only when the chosen action_id is present in type_ids. Otherwise use "click".
- If action_type is "type", text_to_type must be a non empty string taken from the goal.
- If done is true, action_id must be null.
- If action_id is null and done is false, you are saying "no suitable action" and the controller will stop.
- Base every decision strictly on the provided candidates, goal, banned_action_ids, and recent_events. Do not assume hidden app specific behavior.

Cost awareness for type vs click:
- Treat type actions as more expensive than click actions. Only type when there is no obvious clickable option that directly matches what the goal needs (for example a button or option whose text already contains the desired value like a name, email, status, or priority).
- If you have already typed a value into a field and that value matches the goal, do not type the same text into that field again. Repeating an identical type into the same field is almost always wasted work.

Drop down and combobox behavior:

- When the goal asks to set or change a value such as assignee, priority, status, label, or similar, the usual pattern is:
  1) Click a control whose text suggests a picker or combobox (for example "Assignee", "Set priority", "Status", "Label").
  2) If a list of options is visible, prefer clicking the option whose text best matches the requested value from the goal (for example "Urgent", "High", a user email, or a project name), instead of typing into a related text field.
  3) Only use type into a search field inside that picker when there are many options and no visible option text clearly matches the requested value.

- After you have typed into a search field inside a picker or dropdown, your next step should be to click one of the filtered options that matches the goal, not to keep typing the same search text again.
- If repeated type actions into the same picker field do not cause any visible progress (the same options stay visible and no new state appears), switch strategy and prefer click actions or a different control.

- After you have typed into a search field inside a picker or dropdown, your next step should be to click one of the filtered options that matches the goal, not to keep typing the same search text again.
- If repeated type actions into the same picker field do not cause any visible progress (the same options stay visible and no new state appears), switch strategy and prefer click actions or a different control.
- if there is combo of drop down plus a list to select from, prefer to locate the cadidates by the aria lable and then select click action on list item rather than typing into drop down

Use of banned_action_ids and recent_events:
- Never select an action_id that is in banned_action_ids.
- If recent_events show that you already tried an action and its effect_kind was "no_change" or outcome was "no_progress", avoid repeating that same action unless there is a clear new reason.
- Treat steps whose effect_kind is "url_change" or "dom_change_modal" or whose outcome is "progress" as meaningful progress. Prefer to build on that new state instead of undoing it.

Completion logic and avoiding duplicate flows:
- Use recent_events and the current candidates to detect when the goal is already satisfied.
- If you have already:
  1) typed the requested name or title from the goal into an appropriate field,
  2) applied any required assignee, filter, or other target mentioned in the goal, and
  3) clicked a primary call to action with text containing verbs like "create", "new", "add", "save", "submit", "done", "finish", or "update" that produced outcome "progress" or an effect_kind such as "url_change" or "dom_change_modal",
  then you must treat the goal as satisfied.
  In that case you must return:
  - action_id: null
  - done: true
  - capture: true
  - notes explaining that the item has been created or the change applied.
- Once you have created the requested item or applied the requested change, do not start the same create flow again in this run. For example, do not click "Create new issue", "New page", or "Add page" a second time just because it is still visible.
- If the UI shows clear success indicators containing the goal entity and words like "created" or "Issue created" or shows a new row or page whose title matches the goal, prefer to stop rather than create another one.

Navigation and search:
- If the goal is to open or go to a page, database, project, or issue with a specific name, first look for candidates whose text or semantics indicate a page or navigation item that contains key words from the goal (for example semantics including "nav_link" or "page_link"). Prefer clicking those over a generic Search button.
- Only click generic Search buttons (candidates whose text or semantics indicate search) when there is no visible navigation item that matches the goal name.
- After clicking a Search button, if a text field with semantics such as "search_field" appears, your next step should be a type action into that field using relevant text from the goal. Do not repeatedly click the same Search button when it does not reveal new fields or controls.

Decision rules for forms:
- A key text field is one whose label, placeholder, or nearby text clearly represents the main name or title in the goal, such as "Project name", "Issue title", "Title", or "Page title".
- If the goal mentions creating, naming, titling, or filtering something, and you need a form, first click obvious "create", "new", or "add" controls to reveal the form or filter panel.
- Then choose a candidate from type_ids that matches the key field and use action_type "type" with text_to_type set to the name, title, or filter value from the goal.
- Once you have typed the requested name or title into the correct field, do not type into that same field again unless you clearly need to correct it.
- After the key name or title field is filled, your next step should usually be a primary call to action that completes the task (for example "Create project", "Create issue", "Save", "Done", "Submit", "Update") rather than typing the same text into additional fields.

Special handling for share or invite goals:
- If the goal mentions words like "share", "invite", "add people", "add user", "send access", or includes an email address:
  - First, open any obvious share or invite controls such as buttons or menu items whose text or semantics include "Share", "Invite", "Add people", "Add user", "Permissions", or "Private".
  - After share or invite controls are open, look for text fields whose semantics include "invite_field" or "share_email_field", or whose label or placeholder contains words like "email", "invite", or "add people". Select that candidate with action_type "type" and set text_to_type to the email address from the goal.
  - After typing the email, prefer clicking buttons with text containing "Invite", "Send", "Share", or "Add" that look like confirmation actions for the invite, instead of toggling the same Share or Private button again.
  - Once the invite has been sent or the share has clearly been applied and the UI shows a success state, you should treat the goal as satisfied, set action_id to null, done to true, and avoid re opening the share or invite flow in the same run.

Choosing buttons and call to action elements:
- If there is a primary call to action button (is_primary_cta is true or semantics contain "primary" or "cta") whose visible text includes verbs such as create, new, add, save, submit, done, finish, or update, and the goal is about creating, adding, saving, finishing, or updating something, then once key text fields are filled you should strongly prefer clicking that button.
- Do not keep clicking the same primary CTA if recent_events show effect_kind "no_change" or outcome "no_progress". In that case look for missing required fields, confirmation controls, or a way to close a modal after success.
- For assignment or filtering goals, prefer:
  1) opening dropdowns or comboboxes that relate to assignee, owner, or filter,
  2) choosing options whose text matches values mentioned in the goal (for example the specified email or title),
  3) then clicking a single confirm or apply button.

Avoiding useless repetition:
- Avoid repeating the same type action when it will not change state. If the correct text is already present in that field, choose a different action that advances the flow, usually a primary CTA or another field that is still required.
- When several candidates have similar text, use semantics, their kind, the goal text, and recent_events to choose the one that best progresses the task.
- If a create or new control has already opened a modal, form, or new page, the next actions should operate inside that surface (title field, description field, assignee, filters, Ask AI, and so on), not click the global "New" control again.

Grounding and generality:
- The same logic must work across different apps such as Linear and Notion. Do not hardcode workflows. Choose based only on the goal and the candidate metadata you see.

JSON output format requirements:
- Respond with exactly one JSON object and nothing else.
- All string values must use double quotes, including action_id. Example: { "action_id": "btn_5", "action_type": "click", ... }.
- Booleans must be true or false in lowercase. Use null for text_to_type when action_type is not "type".
- Do not add comments, trailing commas, or any extra text before or after the JSON object.
- Never emit unquoted identifiers such as action_id: btn_5. That is invalid JSON and will cause the controller to stop.
"""
def build_policy_prompt(
    task: TaskSpec,
    app_name: str,
    url: str,
    history_summary: str,
    candidates: Sequence[CandidateAction],
    type_ids: Sequence[str] | None = None,
    banned_action_ids: Sequence[str] | None = None,
    recent_events: Sequence[dict[str, Any]] | None = None,
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
    if banned_action_ids:
        lines.append("")
        lines.append(f"Banned action_ids (avoid): {list(banned_action_ids)}")
    if recent_events:
        lines.append("")
        lines.append("Recent events (most recent last):")
        for ev in recent_events:
            lines.append(
                "  - step={step} action_id={aid} type={atype} effect={effect} outcome={outcome} comment={comment}".format(
                    step=ev.get("step_index"),
                    aid=ev.get("action_id"),
                    atype=ev.get("action_type"),
                    effect=ev.get("effect_kind"),
                    outcome=ev.get("outcome"),
                    comment=ev.get("comment", ""),
                )
            )
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
    banned_action_ids: Sequence[str] | None = None,
    recent_events: Sequence[dict[str, Any]] | None = None,
) -> PolicyDecision:
    prompt = build_policy_prompt(
        task,
        app_name,
        url,
        history_summary,
        candidates,
        type_ids,
        banned_action_ids,
        recent_events,
    )
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
        banned_action_ids: Sequence[str] | None = None,
        recent_events: Sequence[dict[str, Any]] | None = None,
    ) -> PolicyDecision:
        prompt = build_policy_prompt(
            task=task,
            app_name=task.app_name,
            url=url,
            history_summary=history_summary,
            candidates=candidates,
            type_ids=type_ids,
            banned_action_ids=banned_action_ids,
            recent_events=recent_events,
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
