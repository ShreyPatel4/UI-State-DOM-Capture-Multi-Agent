from typing import List

from .task_spec import TaskSpec


class Planner:
    """
    Minimal planner stub.
    For now, the main behavior is driven by the Policy and agent loop.
    """

    def plan(self, task: TaskSpec) -> List[str]:
        return [f"High level plan: {task.goal}"]
