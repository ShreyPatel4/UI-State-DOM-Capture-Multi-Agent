import json
from typing import List, Dict

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from ..config import settings
from .task_spec import TaskSpec
from .dom_scanner import CandidateAction


class Policy:
    """
    LLM backed policy that chooses the next UI action using a Hugging Face model.
    """

    def __init__(self, model_name: str | None = None) -> None:
        model_name = model_name or settings.hf_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        self.generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            device="cpu",
            max_new_tokens=128,
            do_sample=False,
        )

    def _build_prompt(
        self,
        task: TaskSpec,
        candidates: List[CandidateAction],
        history_summary: str,
    ) -> str:
        actions_text = "\n".join(
            f"{idx + 1}. id={c.id} action_type={c.action_type} description={c.description}"
            for idx, c in enumerate(candidates)
        )

        prompt = f"""
You are a UI agent controlling a web browser.

Your goal is to complete the user's task by choosing the next best UI action.

Task:
  app: {task.app_name}
  goal: {task.goal}
  object_type: {task.object_type}

Recent history (previous steps):
{history_summary or "none"}

You have these candidate actions on the current page:
{actions_text}

Choose ONE best next action.

Respond ONLY with valid JSON using this schema:

{{
  "chosen_action_id": "act_3",
  "action_type": "click",
  "input_text": null,
  "capture_before": true,
  "capture_after": true,
  "state_label_after": "some_state_label",
  "done": false,
  "reason": "short explanation of your choice"
}}

Do NOT include any extra commentary outside the JSON.
"""
        return prompt.strip()

    def _run_hf(self, prompt: str) -> str:
        out = self.generator(
            prompt,
            num_return_sequences=1,
        )[0]["generated_text"]
        return out

    def _extract_json(self, raw: str) -> Dict:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("No JSON object found in model output")
        json_str = raw[start : end + 1]
        return json.loads(json_str)

    async def choose_action(
        self,
        task: TaskSpec,
        candidates: List[CandidateAction],
        history_summary: str,
    ) -> Dict:
        """
        Decide the next action. Called from the agent loop.
        """
        prompt = self._build_prompt(task, candidates, history_summary)
        raw = self._run_hf(prompt)

        try:
            data = self._extract_json(raw)
        except Exception:
            # Fallback to first candidate if parsing fails
            first = candidates[0]
            return {
                "chosen_action_id": first.id,
                "action_type": first.action_type,
                "input_text": None,
                "capture_before": True,
                "capture_after": True,
                "state_label_after": f"after_{first.id}",
                "done": False,
                "reason": "Fallback decision because model output was not valid JSON",
            }

        # Validate chosen_action_id
        valid_ids = {c.id for c in candidates}
        if data.get("chosen_action_id") not in valid_ids:
            first = candidates[0]
            data["chosen_action_id"] = first.id
            data["action_type"] = first.action_type

        # Ensure required keys exist with sane defaults
        data.setdefault("input_text", None)
        data.setdefault("capture_before", True)
        data.setdefault("capture_after", True)
        data.setdefault("state_label_after", f"after_{data['chosen_action_id']}")
        data.setdefault("done", False)
        data.setdefault("reason", "Model did not provide a reason")

        print(
            "[policy] decision:",
            json.dumps(
                {
                    "app": task.app_name,
                    "goal": task.goal,
                    "chosen_action_id": data.get("chosen_action_id"),
                    "action_type": data.get("action_type"),
                    "capture_before": data.get("capture_before"),
                    "capture_after": data.get("capture_after"),
                    "state_label_after": data.get("state_label_after"),
                    "done": data.get("done"),
                },
                ensure_ascii=False,
            ),
        )

        return data
