from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Callable

from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from sqlalchemy.orm import Session

from ..config import settings
from ..models import Flow, log_flow_event
from .browser import BrowserSession
from .capture import CaptureManager
from .dom_scanner import CandidateAction, scan_candidate_actions
from .policy import PolicyLLMClient, PolicyDecision, choose_action_with_llm
from .state_diff import summarize_state_change
from .task_spec import TaskSpec


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
        log_flow_event(session, flow, "info", "Flow cancelled by user request")
        return True
    return False


def _candidate_key(cand: CandidateAction) -> str:
    return f"{cand.action_type}:{cand.locator}:{cand.description}"


async def _maybe_capture_state(
    capture_manager: CaptureManager,
    page,
    flow: Flow,
    label: str,
    description: str,
    current_dom: str,
    last_captured_dom: str | None,
    last_captured_url: str | None,
    diff_threshold: float,
    step_index: int | None,
    force: bool = False,
):
    url_changed, diff_summary, diff_score, state_kind, changed = summarize_state_change(
        last_captured_dom, current_dom, last_captured_url, page.url, diff_threshold
    )

    if not changed and not force:
        return None, last_captured_dom, last_captured_url, changed, state_kind, diff_score

    step = await capture_manager.capture_step(
        page=page,
        flow=flow,
        label=label,
        description=description,
        dom_html=current_dom,
        diff_summary=diff_summary,
        diff_score=diff_score,
        action_description=description,
        url_changed=url_changed,
        state_kind=state_kind,
        step_index=step_index,
    )
    return step, current_dom, page.url, changed, state_kind, diff_score


async def run_agent_loop(
    task: TaskSpec,
    flow: Flow,
    capture_manager: CaptureManager,
    hf_pipeline,
    start_url: str,
    browser_factory: Callable[[], BrowserSession] = BrowserSession,
    max_steps: int | None = None,
) -> None:
    session = capture_manager.db_session
    llm_client = PolicyLLMClient(hf_pipeline)
    max_steps = max_steps or settings.max_steps
    diff_threshold = settings.dom_diff_threshold
    max_action_failures = settings.max_action_failures
    goal_text = task.goal.lower()

    failure_counts: dict[str, int] = defaultdict(int)
    banned_actions: set[str] = set()

    async with browser_factory() as browser:
        await browser.goto(start_url)

        page = browser.page
        if page is None:
            return

        dom_html = await capture_manager.get_dom_snapshot(page)
        prev_url = page.url
        prev_dom = dom_html
        last_captured_dom = dom_html
        last_captured_url = prev_url

        await capture_manager.capture_step(
            page=page,
            flow=flow,
            label="initial_state",
            dom_html=prev_dom,
            diff_summary=None,
            diff_score=None,
            action_description="initial page load",
            url_changed=True,
            state_kind="url_change",
        )

        log_flow_event(session, flow, "info", "Captured initial_state")

        candidates = await scan_candidate_actions(page, max_actions=40)
        candidates = [c for c in candidates if _candidate_key(c) not in banned_actions]
        if not candidates:
            flow.status = "no_actions"
            flow.status_reason = "no_candidates"
            session.add(flow)
            session.commit()
            log_flow_event(session, flow, "warning", "No candidate actions after initial load")
            return

        history_summary = ""

        if await _check_cancel_requested(session, flow):
            return

        goal_reached = False

        for step_index in range(1, max_steps + 1):
            page = browser.page
            if page is None:
                break

            if await _check_cancel_requested(session, flow):
                return

            if step_index != 1:
                candidates = await scan_candidate_actions(page, max_actions=40)
                candidates = [c for c in candidates if _candidate_key(c) not in banned_actions]

            if not candidates:
                capture_manager.finish_flow(flow, status="no_actions")
                log_flow_event(session, flow, "warning", "No candidate actions available")
                break

            current_url = page.url
            decision: PolicyDecision = choose_action_with_llm(
                llm_client,
                task,
                task.app_name,
                current_url,
                history_summary,
                candidates,
                session=session,
                flow=flow,
            )

            if decision.action_id is None:
                if decision.should_capture:
                    await _maybe_capture_state(
                        capture_manager,
                        page,
                        flow,
                        decision.label or "fallback_capture",
                        decision.reason or "LLM fallback",
                        prev_dom,
                        last_captured_dom,
                        last_captured_url,
                        diff_threshold,
                        step_index,
                        force=True,
                    )
                flow.status = "finished"
                flow.status_reason = "llm_fallback"
                flow.finished_at = datetime.now(timezone.utc)
                session.add(flow)
                session.commit()
                break

            selected_candidate = next((c for c in candidates if c.id == decision.action_id), None)
            if selected_candidate is None:
                log_flow_event(session, flow, "warning", "Selected candidate missing after filtering")
                continue

            if decision.capture_before:
                step, last_captured_dom, last_captured_url, _, _, _ = await _maybe_capture_state(
                    capture_manager,
                    page,
                    flow,
                    f"before_action_{decision.action_id}",
                    f"Before action: {decision.reason or selected_candidate.description}",
                    prev_dom,
                    last_captured_dom,
                    last_captured_url,
                    diff_threshold,
                    step_index,
                )
                if step:
                    log_flow_event(
                        session,
                        flow,
                        "info",
                        f"Captured before_action for {decision.action_id}",
                    )

            locator = page.locator(selected_candidate.locator)
            try:
                if decision.action_type == "click":
                    if not await locator.is_visible():
                        failure_counts[_candidate_key(selected_candidate)] += 1
                        continue
                    await locator.click(timeout=2000)
                elif decision.action_type == "type":
                    if decision.text is None:
                        failure_counts[_candidate_key(selected_candidate)] += 1
                        continue
                    await locator.fill(decision.text)
            except PlaywrightTimeoutError:
                key = _candidate_key(selected_candidate)
                failure_counts[key] += 1
                if failure_counts[key] > max_action_failures:
                    banned_actions.add(key)
                    log_flow_event(session, flow, "warning", f"Banned {selected_candidate.id} after repeated timeouts")
                continue

            await page.wait_for_timeout(800)

            current_dom = await capture_manager.get_dom_snapshot(page)
            current_url = page.url

            url_changed, diff_summary, diff_score, state_kind, changed = summarize_state_change(
                prev_dom, current_dom, prev_url, current_url, diff_threshold
            )

            if not changed:
                key = _candidate_key(selected_candidate)
                failure_counts[key] += 1
                if failure_counts[key] > max_action_failures:
                    banned_actions.add(key)
                    log_flow_event(session, flow, "warning", f"Banned {selected_candidate.id} for no meaningful change")
            else:
                failure_counts[_candidate_key(selected_candidate)] = 0

            should_force_capture = bool(decision.capture_after or decision.should_capture or decision.done)
            step, last_captured_dom, last_captured_url, captured_changed, captured_state_kind, captured_diff = await _maybe_capture_state(
                capture_manager,
                page,
                flow,
                decision.label or f"after_action_{decision.action_id}",
                decision.reason or selected_candidate.description,
                current_dom,
                last_captured_dom,
                last_captured_url,
                diff_threshold,
                step_index,
                force=should_force_capture,
            )
            if step:
                log_flow_event(
                    session,
                    flow,
                    "info",
                    f"step={step_index} url={current_url} action='{selected_candidate.description}' diff={captured_diff} captured=True",
                )

            summary_line = f"{step_index}. {decision.reason or selected_candidate.description}".strip()
            history_summary = "\n".join([line for line in [history_summary, summary_line] if line])

            prev_url = current_url
            prev_dom = current_dom

            if decision.done:
                creation_like = any(
                    keyword in goal_text
                    for keyword in [
                        "create",
                        "add",
                        "new",
                        "save",
                        "submit",
                        "enable",
                        "disable",
                        "change",
                        "update",
                    ]
                )
                capture_goal = any(term in goal_text for term in ["capture", "screenshot"])
                meaningful = bool(
                    url_changed
                    or (diff_score is not None and diff_score >= diff_threshold)
                    or captured_state_kind == "dom_change_modal"
                    or captured_changed
                )
                if meaningful or (capture_goal and not creation_like and (captured_changed or step_index == 1)):
                    goal_reached = True
                else:
                    flow.status_reason = "uncertain_goal"
                    log_flow_event(
                        session,
                        flow,
                        "warning",
                        "LLM indicated done but no meaningful change detected; finishing as uncertain",
                    )
                    break

            if banned_actions:
                candidates = [c for c in candidates if _candidate_key(c) not in banned_actions]

            if goal_reached:
                break

        else:
            flow.status_reason = "max steps reached"
            capture_manager.finish_flow(flow, status="max_steps_reached")
            log_flow_event(session, flow, "info", "Max steps reached, stopping loop")

        if goal_reached:
            flow.status = "finished"
            flow.status_reason = "goal_reached"
            flow.finished_at = datetime.now(timezone.utc)
            session.add(flow)
            session.commit()
        elif flow.status == "running":
            flow.status = "finished"
            flow.status_reason = flow.status_reason or "stopped"
            flow.finished_at = datetime.now(timezone.utc)
            session.add(flow)
            session.commit()
