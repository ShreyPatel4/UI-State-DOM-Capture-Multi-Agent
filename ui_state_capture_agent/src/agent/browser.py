from __future__ import annotations

import logging
import os
from typing import Any

from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    TimeoutError as PlaywrightTimeoutError,
    async_playwright,
)

from src.config import settings
from .page_snapshot import AXNode, PageSnapshot, SnapshotNode


class BrowserSession:
    def __init__(self, user_data_dir: str | None = None) -> None:
        self.browser: Browser | None = None
        self.context: BrowserContext | None = None
        self.page: Page | None = None
        self._playwright: Playwright | None = None
        self.user_data_dir = os.path.expanduser(
            user_data_dir or getattr(settings, "user_data_dir", None) or "~/.softlight_profiles/default"
        )

    async def __aenter__(self) -> "BrowserSession":
        self._playwright = await async_playwright().start()
        self.context = await self._playwright.chromium.launch_persistent_context(
            user_data_dir=self.user_data_dir,
            headless=settings.headless,
        )
        self.page = await self.context.new_page()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.context:
            await self.context.close()
        if self._playwright:
            await self._playwright.stop()

    async def goto(self, url: str, wait_ms: int = 1500) -> None:
        """
        Navigate to a URL and give the app a moment to hydrate.
        """
        if not self.page:
            raise RuntimeError("Browser page is not initialized. Use within an async context manager.")

        await self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
        try:
            await self.page.wait_for_load_state("networkidle", timeout=5000)
        except PlaywrightTimeoutError:
            print("[browser] networkidle wait timed out, continuing anyway")

        if wait_ms > 0:
            await self.page.wait_for_timeout(wait_ms)

    async def screenshot(self, path: str):
        if not self.page:
            raise RuntimeError("Browser page is not initialized. Use within an async context manager.")
        return await self.page.screenshot(path=path, full_page=True)

    async def get_dom(self) -> str:
        if not self.page:
            raise RuntimeError("Browser page is not initialized. Use within an async context manager.")
        return await self.page.content()

    def __repr__(self) -> str:
        return f"BrowserSession(headless={settings.headless})"

    async def capture_page_snapshot(self) -> PageSnapshot | None:
        """
        Use Chrome DevTools Protocol to capture a DOMSnapshot and accessibility tree
        for the current page, normalize it into a PageSnapshot, and return it.

        If CDP is not available or the browser is not Chromium, log a warning and
        return None so that the caller can fall back to the old behavior.
        """

        if not self.page or not self.context:
            return None

        # Only Chromium contexts support CDP sessions in Playwright.
        try:
            session = await self.context.new_cdp_session(self.page)
        except Exception as exc:  # pragma: no cover - depends on runtime browser
            logging.warning("cdp_snapshot_unavailable reason=%s", exc)
            return None

        try:
            dom_snapshot: dict[str, Any] = await session.send(
                "DOMSnapshot.captureSnapshot",
                {
                    "computedStyles": [],
                    "includeDOMRects": False,
                    "includePaintOrder": False,
                    "includeEventListeners": False,
                },
            )
            ax_snapshot: dict[str, Any] = await session.send("Accessibility.getFullAXTree")
        except Exception as exc:  # pragma: no cover - depends on runtime browser
            logging.warning("cdp_snapshot_failed reason=%s", exc)
            return None

        dom_nodes: list[SnapshotNode] = []
        ax_nodes: list[AXNode] = []

        try:
            documents = dom_snapshot.get("documents", [])
            if not documents:
                return None
            node_data = documents[0].get("nodes", {})
            strings = dom_snapshot.get("strings", [])

            def resolve_string(idx: int | None) -> str:
                if idx is None:
                    return ""
                if 0 <= idx < len(strings):
                    return strings[idx]
                return ""

            raw_attributes = node_data.get("attributes", []) or []
            node_names = node_data.get("nodeName", []) or []
            text_values = node_data.get("textValue", []) or []
            node_values = node_data.get("nodeValue", []) or []
            backend_ids = node_data.get("backendNodeId", []) or []

            backend_index_map: dict[int, int] = {}

            for idx, name_idx in enumerate(node_names):
                node_name = resolve_string(name_idx).lower()
                attr_pairs = raw_attributes[idx] if idx < len(raw_attributes) else []
                attrs: dict[str, str] = {}
                for j in range(0, len(attr_pairs), 2):
                    key = resolve_string(attr_pairs[j])
                    val = resolve_string(attr_pairs[j + 1]) if j + 1 < len(attr_pairs) else ""
                    if key:
                        attrs[key] = val
                snippet = resolve_string(text_values[idx] if idx < len(text_values) else None) or resolve_string(
                    node_values[idx] if idx < len(node_values) else None
                )
                dom_nodes.append(
                    SnapshotNode(
                        index=idx,
                        node_name=node_name,
                        attributes=attrs,
                        text_snippet=snippet or None,
                    )
                )
                if idx < len(backend_ids) and backend_ids[idx] is not None:
                    backend_index_map[backend_ids[idx]] = idx

            for ax in ax_snapshot.get("nodes", []) or []:
                node_id = str(ax.get("nodeId") or ax.get("backendDOMNodeId") or len(ax_nodes))
                role_val = ax.get("role", {})
                name_val = ax.get("name", {})
                role = role_val.get("value") if isinstance(role_val, dict) else role_val
                name = name_val.get("value") if isinstance(name_val, dict) else name_val
                dom_indices: list[int] = []
                if "backendDOMNodeId" in ax:
                    backend_id = ax.get("backendDOMNodeId")
                    if backend_id in backend_index_map:
                        dom_indices.append(backend_index_map[backend_id])
                if "nodeId" in ax:
                    backend_id = ax.get("nodeId")
                    if backend_id in backend_index_map:
                        dom_indices.append(backend_index_map[backend_id])
                ax_nodes.append(
                    AXNode(
                        node_id=str(node_id),
                        role=role,
                        name=name,
                        dom_node_indices=dom_indices,
                    )
                )
        except Exception as exc:  # pragma: no cover - defensive parsing
            reason = exc.args[0] if getattr(exc, "args", None) else exc
            if reason == 0 or str(reason) == "0":
                logging.debug("cdp_snapshot_parse_failed reason=%s", exc)
            else:
                logging.warning("cdp_snapshot_parse_failed reason=%s", exc)
            return None

        return PageSnapshot.from_nodes(dom_nodes, ax_nodes)
