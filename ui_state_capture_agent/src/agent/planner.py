from __future__ import annotations

import logging
import re

from openai import OpenAI

from src.config import settings
from .task_spec import TaskSpec

logger = logging.getLogger(__name__)


class Planner:
    def __init__(self, client: OpenAI | None = None) -> None:
        self.client = client or OpenAI(api_key=settings.openai_api_key)

    @staticmethod
    def _build_user_message(task: TaskSpec) -> str:
        lines: list[str] = [
            f"App: {task.app_name}",
            f"Goal: {task.goal}",
            f"Object type: {task.object_type}",
        ]

        if task.constraints:
            lines.append("Constraints:")
            lines.extend(f"- {key}: {value}" for key, value in task.constraints.items())

        return "\n".join(lines)

    @staticmethod
    def _parse_steps(text: str) -> list[str]:
        pattern = re.compile(r"^\s*(\d+)\.\s*(.+)$")
        steps: list[str] = []
        for line in text.splitlines():
            match = pattern.match(line)
            if match:
                steps.append(match.group(2).strip())

        if not steps:
            remaining = text.strip()
            if remaining:
                steps.append(remaining)

        return steps or ["Review app and determine next steps."]

    def plan(self, task: TaskSpec) -> list[str]:
        system_prompt = (
            "You are a UI task planner. Given an app, goal, and object type, "
            "output a numbered list of high level UI steps to achieve the goal."
        )
        user_content = self._build_user_message(task)

        try:
            completion = self.client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            )
            content = completion.choices[0].message.content if completion.choices else None
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to generate plan: %s", exc)
            return ["Review app and determine next steps."]

        if not content:
            return ["Review app and determine next steps."]

        return self._parse_steps(content)
