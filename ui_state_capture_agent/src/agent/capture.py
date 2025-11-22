from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from playwright.async_api import Page
from sqlalchemy.orm import Session

from ..config import settings
from ..models import Flow, Step
from ..storage.base import StorageBackend


class CaptureManager:
    def __init__(self, db_session: Session, storage: StorageBackend) -> None:
        self.db_session = db_session
        self.storage = storage

    async def get_dom_snapshot(self, page: Page) -> str:
        return await page.content()

    def start_flow(self, app_name: str, task_id: str, task_title: str, task_blurb: str) -> Flow:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        prefix = f"{app_name}/{task_id}/{run_id}"

        flow = Flow(
            app_name=app_name,
            task_id=task_id,
            task_title=task_title,
            task_blurb=task_blurb,
            run_id=run_id,
            status="running",
            started_at=datetime.now(timezone.utc),
            bucket=settings.minio_bucket,
            prefix=prefix,
        )
        self.db_session.add(flow)
        self.db_session.commit()
        self.db_session.refresh(flow)
        return flow

    async def capture_step(
        self,
        flow: Flow,
        page: Page,
        label: str,
        dom_html: str,
        description: str | None = None,
        diff_summary: Optional[str] = None,
        diff_score: Optional[float] = None,
<<<<<<< ours
        action_description: str | None = None,
=======
        url_changed: Optional[bool] = None,
        state_kind: Optional[str] = None,
>>>>>>> theirs
        step_index: Optional[int] = None,
    ) -> Step:
        if step_index is None:
            latest_index = (
                self.db_session.query(Step.step_index)
                .filter(Step.flow_id == flow.id)
                .order_by(Step.step_index.desc())
                .first()
            )
            step_index = (latest_index[0] if latest_index else 0) + 1

        screenshot_key = f"{flow.prefix}/step_{step_index}_screenshot.png"
        dom_key = f"{flow.prefix}/step_{step_index}_dom.html"

        screenshot_bytes = await page.screenshot(full_page=True)

        self.storage.save_bytes(screenshot_key, screenshot_bytes)
        self.storage.save_bytes(dom_key, dom_html.encode("utf-8"))

        step = Step(
            flow_id=flow.id,
            step_index=step_index,
            state_label=label,
            description=description or action_description or "",
            url=page.url,
            url_changed=url_changed,
            state_kind=state_kind,
            screenshot_key=screenshot_key,
            dom_key=dom_key,
            diff_summary=diff_summary,
            diff_score=diff_score,
        )
        self.db_session.add(step)
        self.db_session.commit()
        self.db_session.refresh(step)
        return step

    def finish_flow(self, flow: Flow, status: str) -> None:
        flow.status = status
        flow.finished_at = datetime.now(timezone.utc)
        self.db_session.add(flow)
        self.db_session.commit()
