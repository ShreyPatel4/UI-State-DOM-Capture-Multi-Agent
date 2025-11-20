import argparse
import asyncio
import re

from src.agent.browser import BrowserSession
from src.agent.capture import CaptureManager
from src.agent.planner import Planner
from src.agent.task_spec import TaskSpec
from src.models import SessionLocal
from src.storage.minio_store import get_storage


APP_URLS = {
    "linear": "https://linear.app",
    "notion": "https://www.notion.so",
    "outlook": "https://outlook.office.com/mail/",
}


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", text).strip("-")
    return slug or "task"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a planned UI task")
    parser.add_argument("--query", required=True, help="Task query, e.g. 'linear: create task'")
    parser.add_argument("--app", required=False, help="Optional app override (linear, notion, outlook)")
    return parser.parse_args()


async def run_task(query: str, app_override: str | None = None) -> None:
    task_spec = TaskSpec.from_query(query)
    if app_override:
        task_spec.app_name = app_override

    planner = Planner()
    steps = planner.plan(task_spec)

    db_session = SessionLocal()
    storage = get_storage()
    capture_manager = CaptureManager(db_session, storage)

    task_id = _slugify(task_spec.goal or task_spec.raw_query)
    task_title = task_spec.goal or task_spec.raw_query
    flow = capture_manager.start_flow(
        app_name=task_spec.app_name,
        task_id=task_id,
        task_title=task_title,
        task_blurb=task_spec.raw_query,
    )

    app_url = APP_URLS.get(task_spec.app_name.lower())
    if not app_url:
        raise ValueError(f"Unsupported app: {task_spec.app_name}")

    try:
        async with BrowserSession() as browser:
            await browser.goto(app_url)

            for index, step_description in enumerate(steps, start=1):
                if not browser.page:
                    raise RuntimeError("Browser page is not initialized")

                screenshot_bytes = await browser.page.screenshot(full_page=True)
                dom_html = await browser.get_dom()
                page_url = browser.page.url

                capture_manager.capture_step(
                    flow=flow,
                    step_index=index,
                    state_label=f"planned_step_{index}",
                    description=step_description,
                    page_url=page_url,
                    screenshot_bytes=screenshot_bytes,
                    dom_html=dom_html,
                )
    finally:
        capture_manager.finish_flow(flow, "success")
        db_session.close()


def main() -> None:
    args = parse_args()
    asyncio.run(run_task(query=args.query, app_override=args.app))


if __name__ == "__main__":
    main()
