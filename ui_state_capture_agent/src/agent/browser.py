from __future__ import annotations

from playwright.async_api import Browser, BrowserContext, Page, Playwright, async_playwright

from src.config import settings


class BrowserSession:
    def __init__(self) -> None:
        self.browser: Browser | None = None
        self.context: BrowserContext | None = None
        self.page: Page | None = None
        self._playwright: Playwright | None = None

    async def __aenter__(self) -> "BrowserSession":
        self._playwright = await async_playwright().start()
        self.browser = await self._playwright.chromium.launch(headless=settings.headless)
        self.context = await self.browser.new_context()
        self.page = await self.context.new_page()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self._playwright:
            await self._playwright.stop()

    async def goto(self, url: str):
        if not self.page:
            raise RuntimeError("Browser page is not initialized. Use within an async context manager.")
        return await self.page.goto(url)

    async def screenshot(self, path: str):
        if not self.page:
            raise RuntimeError("Browser page is not initialized. Use within an async context manager.")
        return await self.page.screenshot(path=path, full_page=True)

    async def get_dom(self) -> str:
        if not self.page:
            raise RuntimeError("Browser page is not initialized. Use within an async context manager.")
        return await self.page.content()
