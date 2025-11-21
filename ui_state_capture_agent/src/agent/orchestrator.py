import asyncio
from typing import Final

from ..models import SessionLocal, Flow
from .task_spec import TaskSpec
from .capture import CaptureManager
from .policy import Policy
from .agent_loop import run_agent_loop
from ..storage.minio_store import get_storage

APP_START_URLS: Final = {
    "linear": "https://linear.app",
    "notion": "https://www.notion.so",
    "outlook": "https://outlook.office.com/mail/",
}


_run_lock: asyncio.Lock | None = None
_run_lock_loop: asyncio.AbstractEventLoop | None = None


def _get_run_lock() -> asyncio.Lock:
    """Ensure only one agent run executes per event loop.

    Playwright does not always behave well when multiple browser sessions are
    created concurrently in the same process. A module-level asyncio.Lock keeps
    requests serialized so that FastAPI only runs one long-lived agent at a
    time. The lock is recreated if a new event loop is used (e.g., when calling
    from the CLI via asyncio.run).
    """

    global _run_lock, _run_lock_loop

    loop = asyncio.get_running_loop()
    if _run_lock is None or _run_lock_loop is not loop:
        _run_lock = asyncio.Lock()
        _run_lock_loop = loop

    return _run_lock


async def run_task_query_async(raw_query: str) -> Flow:
    """Orchestrate a full agent run for a single natural language query."""

    task = TaskSpec.from_query(raw_query)
    if task.app_name not in APP_START_URLS:
        raise ValueError(f"Unsupported app_name: {task.app_name}")

    start_url = APP_START_URLS[task.app_name]

    run_lock = _get_run_lock()

    async with run_lock:
        db = SessionLocal()
        try:
            storage = get_storage()
            capture_manager = CaptureManager(db_session=db, storage=storage)

            flow = capture_manager.start_flow(
                app_name=task.app_name,
                task_id=task.object_type or "generic_task",
                task_title=task.goal,
                task_blurb=f"Auto run for query: {raw_query}",
            )

            policy = Policy()

            await run_agent_loop(
                task=task,
                flow=flow,
                capture_manager=capture_manager,
                policy=policy,
                start_url=start_url,
                max_steps=15,
            )
            db.refresh(flow)
            return flow
        finally:
            db.close()


def run_task_query_blocking(raw_query: str) -> Flow:
    """Synchronous wrapper for CLI usage."""

    return asyncio.run(run_task_query_async(raw_query))
