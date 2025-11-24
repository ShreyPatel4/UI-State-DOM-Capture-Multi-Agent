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

        candidates, type_ids = await scan_candidate_actions(page, max_actions=40)
        candidates = [c for c in candidates if _candidate_key(c) not in banned_actions]
        type_ids = [c.id for c in candidates if c.action_type == "type"]
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
                candidates, type_ids = await scan_candidate_actions(page, max_actions=40)
                candidates = [c for c in candidates if _candidate_key(c) not in banned_actions]
                type_ids = [c.id for c in candidates if c.action_type == "type"]

            if not candidates:
                capture_manager.finish_flow(flow, status="no_actions")
                log_flow_event(session, flow, "warning", "No candidate actions available")
                break

            current_url = page.url
            log_flow_event(
                session,
                flow,
                "info",
                "policy_call step={step} url={url} candidates={count} type_ids={types}".format(
                    step=step_index,
                    url=current_url,
                    count=len(candidates),
                    types=type_ids[:5],
                ),
            )
            decision: PolicyDecision = choose_action_with_llm(
                llm_client,
                task,
                task.app_name,
                current_url,
                history_summary,
                candidates,
                session=session,
                flow=flow,
                step_index=step_index,
            )

            if decision.action_id is None:
                if decision.capture:
                    await _maybe_capture_state(
                        capture_manager,
                        page,
                        flow,
                        "fallback_capture",
                        decision.notes or "LLM fallback",
                        prev_dom,
                        last_captured_dom,
                        last_captured_url,
                        diff_threshold,
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

            action_description = decision.notes or selected_candidate.description
            if decision.action_type == "type" and decision.text_to_type:
                preview_text = decision.text_to_type
                if len(preview_text) > 80:
                    preview_text = preview_text[:77] + "..."
                action_description = f"{selected_candidate.description}: \"{preview_text}\""

            locator = page.locator(selected_candidate.locator)
            try:
                if decision.action_type == "click":
                    if not await locator.is_visible():
                        failure_counts[_candidate_key(selected_candidate)] += 1
                        continue
                    await locator.click(timeout=2000)
                elif decision.action_type == "type":
                    if not await locator.is_visible():
                        failure_counts[_candidate_key(selected_candidate)] += 1
                        log_flow_event(
                            session,
                            flow,
                            "warning",
                            f"Target for {selected_candidate.id} not visible; skipping",
                        )
                        continue
                    if not decision.text_to_type:
                        log_flow_event(
                            session,
                            flow,
                            "warning",
                            "LLM did not provide text_to_type for type action; skipping",
                        )
                        failure_counts[_candidate_key(selected_candidate)] += 1
                        continue
                    typed_value = decision.text_to_type
                    logged_value = typed_value if len(typed_value) <= 120 else typed_value[:117] + "..."
                    try:
                        await locator.click(timeout=2000)
                    except PlaywrightTimeoutError:
                        key = _candidate_key(selected_candidate)
                        failure_counts[key] += 1
                        if failure_counts[key] > max_action_failures:
                            banned_actions.add(key)
                            log_flow_event(
                                session,
                                flow,
                                "warning",
                                f"Banned {selected_candidate.id} after repeated timeouts",
                            )
                        continue
                    try:
                        await locator.fill(typed_value, timeout=4000)
                        log_flow_event(
                            session,
                            flow,
                            "info",
                            "typed value='{val}' into action='{desc}' selector='{loc}'".format(
                                val=logged_value,
                                desc=selected_candidate.description,
                                loc=selected_candidate.locator,
                            ),
                        )
                    except PlaywrightTimeoutError:
                        key = _candidate_key(selected_candidate)
                        failure_counts[key] += 1
                        log_flow_event(
                            session,
                            flow,
                            "warning",
                            f"Typing timed out for {selected_candidate.id}; attempting fallback type",
                        )
                        try:
                            await locator.type(typed_value, timeout=4000)
                            log_flow_event(
                                session,
                                flow,
                                "info",
                                "typed value='{val}' into action='{desc}' selector='{loc}'".format(
                                    val=logged_value,
                                    desc=selected_candidate.description,
                                    loc=selected_candidate.locator,
                                ),
                            )
                        except PlaywrightTimeoutError:
                            if failure_counts[key] > max_action_failures:
                                banned_actions.add(key)
                                log_flow_event(
                                    session,
                                    flow,
                                    "warning",
                                    f"Banned {selected_candidate.id} after repeated timeouts",
                                )
                            continue
                        except Exception as exc:
                            log_flow_event(
                                session,
                                flow,
                                "warning",
                                f"Fallback type failed for {selected_candidate.id}: {exc}",
                            )
                            continue
                    except Exception as exc:
                        log_flow_event(
                            session,
                            flow,
                            "warning",
                            f"Typing failed for {selected_candidate.id}: {exc}",
                        )
                        failure_counts[_candidate_key(selected_candidate)] += 1
                        continue
            except PlaywrightTimeoutError:
                key = _candidate_key(selected_candidate)
                failure_counts[key] += 1
                if failure_counts[key] > max_action_failures:
                    banned_actions.add(key)
                    log_flow_event(session, flow, "warning", f"Banned {selected_candidate.id} after repeated timeouts")
                continue
            except Exception as exc:
                key = _candidate_key(selected_candidate)
                failure_counts[key] += 1
                log_flow_event(
                    session,
                    flow,
                    "warning",
                    f"Action {selected_candidate.id} failed: {exc}",
                )
                if failure_counts[key] > max_action_failures:
                    banned_actions.add(key)
                    log_flow_event(
                        session, flow, "warning", f"Banned {selected_candidate.id} after repeated failures"
                    )
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

            should_force_capture = bool(decision.capture or decision.done)
            step, last_captured_dom, last_captured_url, captured_changed, captured_state_kind, captured_diff = await _maybe_capture_state(
                capture_manager,
                page,
                flow,
                f"after_action_{decision.action_id}",
                action_description,
                current_dom,
                last_captured_dom,
                last_captured_url,
                diff_threshold,
                force=should_force_capture,
            )
            if step:
                log_flow_event(
                    session,
                    flow,
                    "info",
                    f"step={step_index} url={current_url} action='{action_description}' diff={captured_diff} captured=True",
                )

            summary_line = f"{step_index}. {action_description}".strip()
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
