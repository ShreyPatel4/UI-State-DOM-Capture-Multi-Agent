from datetime import datetime, timezone

from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from sqlalchemy.orm import Session

from .task_spec import TaskSpec
from .dom_scanner import scan_candidate_actions
from .policy import Policy
from .browser import BrowserSession
from .capture import CaptureManager
from ..models import Flow
from .state_diff import compute_dom_diff

MAX_STEPS = 10
MAX_LOW_DIFF_IN_A_ROW = 4
LOW_DIFF_THRESHOLD = 0.02


def _set_cancelled(flow: Flow, session: Session) -> None:
    flow.status = "cancelled"
    flow.status_reason = "cancel_requested"
    flow.finished_at = datetime.now(timezone.utc)
    session.add(flow)
    session.commit()


async def _check_cancel_requested(session: Session, flow: Flow) -> bool:
    refreshed = session.get(Flow, flow.id)
    if refreshed and refreshed.cancel_requested:
        _set_cancelled(flow, session)
        return True
    return False


async def run_agent_loop(
    task: TaskSpec,
    flow: Flow,
    capture_manager: CaptureManager,
    policy: Policy,
    start_url: str,
) -> None:
    session = capture_manager.db_session
    async with BrowserSession() as browser:
        await browser.goto(start_url)

        page = browser.page
        if page is None:
            return

        dom_html = await capture_manager.get_dom_snapshot(page)
        await capture_manager.capture_step(
            page=page,
            flow=flow,
            label="initial_state",
            dom_html=dom_html,
            diff_summary=None,
            diff_score=None,
            action_description="initial page load",
        )

        candidates = await scan_candidate_actions(page, max_actions=60)
        if not candidates:
            print("[agent_loop] No candidate actions after initial load, finishing")
            flow.status = "no_actions"
            flow.save()
            return

        history_summary = ""

        page = browser.page
        if page is None:
            return

        dom_initial = await capture_manager.get_dom_snapshot(page)
        await capture_manager.capture_step(
            page=page,
            flow=flow,
            label="initial",
            description="Initial state",
            dom_html=dom_initial,
            step_index=0,
            state_kind="initial",
            url_changed=False,
        )

        if await _check_cancel_requested(session, flow):
            return

        previous_url = page.url

        goal_reached = False
        low_diff_streak = 0

        for step_index in range(1, MAX_STEPS + 1):
            page = browser.page
            if page is None:
                break

            if await _check_cancel_requested(session, flow):
                return

            dom_before = await capture_manager.get_dom_snapshot(page)

            if step_index != 1:
                candidates = await scan_candidate_actions(page, max_actions=60)
            print(f"[agent_loop] URL={page.url} candidates={len(candidates)}")
            if not candidates:
                capture_manager.finish_flow(flow, status="no_actions")
                break

            print(f"[agent_loop] history_summary length={len(history_summary or '')}")
            decision = await policy.choose_action(task, candidates, history_summary)

            if decision.get("done"):
                if decision.get("capture_after"):
                    await capture_manager.capture_step(
                        page=page,
                        flow=flow,
                        label=decision.get("state_label_after", "done"),
                        description=decision.get("reason", ""),
                        dom_html=dom_before,
                        url_changed=False,
                        state_kind="dom_change",
                        step_index=step_index,
                    )

                goal_reached = True
                break

            if decision.get("capture_before"):
                await capture_manager.capture_step(
                    page=page,
                    flow=flow,
                    label=decision.get("state_label_before") or f"before_{step_index}",
                    description=f"Before action: {decision.get('reason', '')}",
                    dom_html=dom_before,
                    url_changed=False,
                    state_kind="dom_change",
                    step_index=step_index,
                )

            cand = next((c for c in candidates if c.id == decision.get("chosen_action_id")), candidates[0])

            if decision.get("action_type") == "click":
                locator = page.locator(cand.locator)

                try:
                    if not await locator.is_visible():
                        print(
                            f"[agent_loop] Skipping action {cand.id}: locator {cand.locator} is not visible"
                        )
                        continue
                except PlaywrightTimeoutError:
                    print(
                        f"[agent_loop] Visibility check timed out for {cand.id} ({cand.locator}), skipping"
                    )
                    continue

                try:
                    await locator.click(timeout=2000)
                except PlaywrightTimeoutError:
                    print(
                        f"[agent_loop] Click timed out for {cand.id} ({cand.locator}), skipping this action"
                    )
                    history_summary = (history_summary or "") + (
                        f"\nAction {cand.id} with locator {cand.locator} failed (click timeout)."
                    )
                    continue
            elif decision.get("action_type") == "type":
                await page.locator(cand.locator).fill(decision.get("input_text", ""))

            await page.wait_for_timeout(1000)

            dom_after = await capture_manager.get_dom_snapshot(page)
            diff_summary, diff_score = compute_dom_diff(dom_before, dom_after)
            url_changed = page.url != previous_url
            state_kind = "url_change" if url_changed else "dom_change"

            should_capture_after = bool(decision.get("capture_after"))
            if diff_score is not None and diff_score > 0.1:
                should_capture_after = True

            if should_capture_after:
                await capture_manager.capture_step(
                    page=page,
                    flow=flow,
                    label=decision.get("state_label_after") or "state_changed",
                    description=decision.get("reason", ""),
                    dom_html=dom_after,
                    diff_summary=diff_summary,
                    diff_score=diff_score,
                    url_changed=url_changed,
                    state_kind=state_kind,
                    step_index=step_index,
                )

            summary_line = f"{step_index}. {decision.get('reason', '')}".strip()
            history_summary = "\n".join(
                [line for line in [history_summary, summary_line] if line]
            )

            previous_url = page.url

            if diff_score is None or diff_score < LOW_DIFF_THRESHOLD:
                low_diff_streak += 1
            else:
                low_diff_streak = 0

            if low_diff_streak >= MAX_LOW_DIFF_IN_A_ROW:
                flow.status = "finished"
                flow.status_reason = "low_diff_loop"
                flow.finished_at = datetime.now(timezone.utc)
                session.add(flow)
                session.commit()
                break

        else:
            capture_manager.finish_flow(flow, status="max_steps_reached")

        if goal_reached:
            flow.status = "finished"
            flow.status_reason = "goal_reached"
            flow.finished_at = datetime.now(timezone.utc)
            session.add(flow)
            session.commit()
