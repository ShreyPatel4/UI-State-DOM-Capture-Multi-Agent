import json
from dataclasses import dataclass
from typing import Any, Literal, Optional, Sequence

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from ..config import settings
from .dom_scanner import CandidateAction
from .task_spec import TaskSpec


def _extract_json(text: str) -> dict | None:
    """
    Try to extract and parse the first JSON object from an LLM response.
    Returns a dict or None if parsing fails.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


@dataclass
class PolicyDecision:
    action_id: str
    action_type: Literal["click", "type"]
    text: Optional[str]
    done: bool
    capture_before: bool
    capture_after: bool
    label: Optional[str]
    reason: Optional[str]


POLICY_SYSTEM_PROMPT = """
You are a UI navigation agent inside a multi agent system.

You do not see raw HTML. You only see:
  • The user goal for this run.
  • The current app name.
  • The current page URL.
  • A short natural language summary of what has already happened.
  • A list of candidate actions extracted from the DOM.

Each candidate action has:
  • id: an identifier such as "btn_0", "link_3", "input_2".
  • action_type: either "click" or "type".
  • description: a short human description such as "button with text 'Create issue'" or "input with placeholder 'Title'".

Your job is to pick the single best next action to move toward the goal.

Very important:
  • You must always choose exactly one of the provided candidates. Never invent new ids.
  • If the goal requires typing text (for example "create issue named Softlight test"), choose a candidate with action_type "type" and provide the text field.
  • If the goal requires clicking navigation or confirmation controls, choose a candidate with action_type "click".
  • You should think step by step about the goal and what state we are currently in before choosing.

Captures and state labels:
  • The engine has already captured the initial page after navigation, so you do not need to handle the very first screenshot.
  • Use capture_before=true when this action will show an interesting "before" state, for example before opening a modal or before submitting a form.
  • Use capture_after=true when the action is likely to visibly change the UI, for example:
      • opening a modal or dropdown
      • moving to a different page or tab
      • applying a filter or creating a new entity
  • Set label to a short snake_case description of the state after the action, for example:
      • "issue_form_open"
      • "issue_created"
      • "pricing_page_open"

Done flag:
  • If you believe the user goal will be satisfied after this action and the resulting state is visible, set done=true.
  • Examples:
      • After clicking a "Create issue" button that submits a filled form and the goal is "create issue named X".
      • After clicking a "Pricing" link when the goal is "open the pricing page and capture one screenshot".
  • Otherwise, set done=false and the engine will call you again with an updated history.

Output requirements:
  • You must return exactly one JSON object.
  • The JSON must match this schema exactly:
       {
         "action_id": "<one of the candidate ids>",
         "action_type": "click" or "type",
         "text": null or "<text to type>",
         "done": true or false,
         "capture_before": true or false,
         "capture_after": true or false,
         "label": null or "<short_snake_case_state_label>",
         "reason": "<short natural language reason>"
       }
  • Return only JSON. Do not include any explanation outside of the JSON object.
"""


def build_policy_prompt(
    task: TaskSpec,
    app_name: str,
    url: str,
    history_summary: str,
    candidates: Sequence[CandidateAction],
) -> str:
    """
    Build the text prompt for Qwen.
    """
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
    lines.append("Candidate actions:")
    for cand in candidates:
        lines.append(
            f"  - id={cand.id}  type={cand.action_type}  description={cand.description}"
        )
    lines.append("")
    lines.append(
        "Return a single JSON object that follows the schema exactly. "
        "Do not include any text before or after the JSON."
    )
    return "\n".join(lines)


def choose_fallback_action(
    goal: str, candidates: Sequence[CandidateAction]
) -> CandidateAction:
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
        return self.hf_pipeline(prompt, num_return_sequences=1)[0]["generated_text"]

    def complete(self, prompt: str) -> str:
        return self.generate(prompt)


def choose_action_with_llm(
    llm: PolicyLLMClient,
    task: TaskSpec,
    app_name: str,
    url: str,
    history_summary: str,
    candidates: Sequence[CandidateAction],
) -> PolicyDecision:
    prompt = build_policy_prompt(task, app_name, url, history_summary, candidates)
    raw = llm.complete(prompt).strip()

    data = _extract_json(raw)
    if data is None:
        fallback = choose_fallback_action(task.goal, candidates)
        return PolicyDecision(
            action_id=fallback.id,
            action_type=fallback.action_type,
            text=None,
            done=False,
            capture_before=True,
            capture_after=True,
            label=f"after_{fallback.id}",
            reason="Fallback decision because model output was not valid JSON",
        )

    action_id = data.get("action_id")
    cand_map = {c.id: c for c in candidates}
    if action_id not in cand_map:
        fallback = choose_fallback_action(task.goal, candidates)
        return PolicyDecision(
            action_id=fallback.id,
            action_type=fallback.action_type,
            text=None,
            done=False,
            capture_before=True,
            capture_after=True,
            label=f"after_{fallback.id}",
            reason="Fallback decision because model output did not match a candidate id",
        )

    cand = cand_map[action_id]
    action_type = data.get("action_type") or cand.action_type

    return PolicyDecision(
        action_id=action_id,
        action_type=action_type,
        text=data.get("text"),
        done=bool(data.get("done", False)),
        capture_before=bool(data.get("capture_before", True)),
        capture_after=bool(data.get("capture_after", True)),
        label=data.get("label"),
        reason=data.get("reason"),
    )


class Policy:
    """
    LLM backed policy that chooses the next UI action using a Hugging Face model.
    """

    def __init__(self, model_name: str | None = None) -> None:
        model_name = model_name or settings.hf_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.generator = create_policy_hf_pipeline(model_name)

    def _run_hf(self, prompt: str) -> str:
        out = self.generator(
            prompt,
            num_return_sequences=1,
        )[0]["generated_text"]
        return out

    def _extract_json(self, raw: str) -> dict:
        data = _extract_json(raw.strip())
        if data is None:
            raise ValueError("bad_json")
        return data

    async def choose_action(
        self,
        task: TaskSpec,
        candidates: Sequence[CandidateAction],
        history_summary: str,
        url: str,
    ) -> PolicyDecision:
        """
        Decide the next action. Called from the agent loop.
        """
        prompt = build_policy_prompt(
            task=task,
            app_name=task.app_name,
            url=url,
            history_summary=history_summary,
            candidates=candidates,
        )
        raw = self._run_hf(prompt)

        try:
            data = self._extract_json(raw)
        except ValueError:
            fallback = choose_fallback_action(task.goal, candidates)
            return PolicyDecision(
                action_id=fallback.id,
                action_type=fallback.action_type,
                text=None,
                done=False,
                capture_before=True,
                capture_after=True,
                label=f"after_{fallback.id}",
                reason="Fallback decision because model output was not valid JSON",
            )

        # Validate action_id
        valid_ids = {c.id for c in candidates}
        if data.get("action_id") not in valid_ids:
            first = candidates[0]
            data["action_id"] = first.id
            data["action_type"] = first.action_type

        # Ensure required keys exist with sane defaults
        data.setdefault("text", None)
        data.setdefault("capture_before", True)
        data.setdefault("capture_after", True)
        data.setdefault("label", f"after_{data['action_id']}")
        data.setdefault("done", False)
        data.setdefault("reason", "Model did not provide a reason")

        decision = PolicyDecision(
            action_id=data.get("action_id"),
            action_type=data.get("action_type", "click"),
            text=data.get("text"),
            done=bool(data.get("done")),
            capture_before=bool(data.get("capture_before")),
            capture_after=bool(data.get("capture_after")),
            label=data.get("label"),
            reason=data.get("reason"),
        )

        print(
            "[policy] decision:",
            json.dumps(
                {
                    "app": task.app_name,
                    "goal": task.goal,
                    "action_id": decision.action_id,
                    "action_type": decision.action_type,
                    "capture_before": decision.capture_before,
                    "capture_after": decision.capture_after,
                    "label": decision.label,
                    "done": decision.done,
                },
                ensure_ascii=False,
            ),
        )

        return decision
