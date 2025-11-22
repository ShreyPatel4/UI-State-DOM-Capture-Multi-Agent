from dataclasses import dataclass
from typing import Dict


@dataclass
class AppResolution:
    app_name: str
    start_url: str
    normalized_goal: str


class AppResolver:
    """
    Agent A light: infer app name and starting URL from a natural language query.
    """

    def __init__(self) -> None:
        self.known_apps: Dict[str, str] = {
            "linear": "https://linear.app",
            "notion": "https://www.notion.so",
            "outlook": "https://outlook.live.com",
            "linkedin": "https://www.linkedin.com",
        }

    def resolve(self, raw_query: str) -> AppResolution:
        text = raw_query.strip()

        prefix, sep, rest = text.partition(":")
        if sep:
            candidate_app = prefix.strip().lower()
            goal = rest.strip()
        else:
            candidate_app = ""
            goal = text

        app = None
        lower = text.lower()
        if candidate_app:
            app = candidate_app

        if not app:
            for name in self.known_apps:
                if name in lower:
                    app = name
                    break

        if not app:
            tokens = goal.split()
            app = tokens[0].lower() if tokens else ""

        app = app.lower()
        start_url = self.known_apps.get(app) if app else ""
        if not start_url and app:
            start_url = f"https://{app}.com"

        return AppResolution(
            app_name=app,
            start_url=start_url,
            normalized_goal=goal or text,
        )
