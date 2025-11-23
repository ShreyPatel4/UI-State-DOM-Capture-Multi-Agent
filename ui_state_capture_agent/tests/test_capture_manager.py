import sys
import asyncio
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.capture import CaptureManager  # noqa: E402


class DummySession:
    def query(self, *_args, **_kwargs):
        class Q:
            def filter(self, *_):
                return self

            def order_by(self, *_):
                return self

            def first(self):
                return None

        return Q()

    def add(self, *_args, **_kwargs):
        return None

    def commit(self):
        return None

    def refresh(self, *_args, **_kwargs):
        return None


class DummyStorage:
    def save_bytes(self, *_args, **_kwargs):
        return None


class DummyPage:
    url = "http://example.com"

    async def screenshot(self, full_page: bool = True):
        return b"image"

    async def content(self):
        return "<html></html>"

def test_capture_step_accepts_dom_changed_kwargs():
    capture_manager = CaptureManager(DummySession(), DummyStorage())
    flow = SimpleNamespace(id="flow1", prefix="pref")

    step = asyncio.run(
        capture_manager.capture_step(
            page=DummyPage(),
            flow=flow,
            label="test_state",
            dom_html="<html></html>",
            diff_summary=None,
            diff_score=None,
            action_description="",
            url_changed=True,
            dom_changed=True,
            modal_change=False,
            state_kind="dom_change",
        )
    )

    assert step.state_label == "test_state"
    assert step.url_changed is True
    assert step.dom_changed is True
