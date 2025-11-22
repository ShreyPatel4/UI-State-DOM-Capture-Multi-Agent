from __future__ import annotations

import os

from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    TimeoutError as PlaywrightTimeoutError,
    async_playwright,
)

from src.config import settings


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
