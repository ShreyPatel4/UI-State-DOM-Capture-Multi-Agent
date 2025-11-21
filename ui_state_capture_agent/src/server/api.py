from __future__ import annotations

import io
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..agent.orchestrator import run_task_query_async
from ..models import Flow, Step, get_db
from ..storage.base import StorageBackend
from ..storage.minio_store import get_storage

BASE_DIR = Path(__file__).parent

app = FastAPI()
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


class RunTaskRequest(BaseModel):
    query: str


class RunTaskResponse(BaseModel):
    flow_id: str
    run_id: str
    app_name: str
    task_title: str
    status: str


@app.post("/agent/run", response_model=RunTaskResponse)
async def run_agent_task(payload: RunTaskRequest):
    """
    Agent A calls this route with a natural language query.
    We run the full agent loop and return the Flow summary.
    """

    flow = await run_task_query_async(payload.query)
    return RunTaskResponse(
        flow_id=str(flow.id),
        run_id=flow.run_id,
        app_name=flow.app_name,
        task_title=flow.task_title,
        status=flow.status,
    )


@app.get("/", response_class=HTMLResponse)
def list_flows(request: Request, db: Session = Depends(get_db)) -> Any:
    flows = db.query(Flow).order_by(Flow.started_at.desc()).limit(50).all()
    return templates.TemplateResponse("flows_list.html", {"request": request, "flows": flows})


@app.post("/api/runs")
async def start_run(payload: RunTaskRequest) -> dict[str, Any]:
    try:
        flow = await run_task_query_async(payload.query)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"flow_id": str(flow.id), "status": flow.status}


@app.get("/api/flows/{flow_id}")
def get_flow_status(flow_id: UUID, db: Session = Depends(get_db)) -> dict[str, Any]:
    flow = db.query(Flow).filter(Flow.id == flow_id).first()
    if flow is None:
        raise HTTPException(status_code=404, detail="Flow not found")

    steps = (
        db.query(Step)
        .filter(Step.flow_id == flow_id)
        .order_by(Step.step_index.asc())
        .all()
    )

    return {
        "id": str(flow.id),
        "status": flow.status,
        "started_at": flow.started_at,
        "finished_at": flow.finished_at,
        "steps": [
            {
                "index": step.step_index,
                "label": step.state_label,
                "description": step.description,
                "url": step.url,
            }
            for step in steps
        ],
    }


@app.get("/flows/{flow_id}", response_class=HTMLResponse)
def flow_detail(flow_id: UUID, request: Request, db: Session = Depends(get_db)) -> Any:
    flow = db.query(Flow).filter(Flow.id == flow_id).first()
    if flow is None:
        raise HTTPException(status_code=404, detail="Flow not found")

    steps = (
        db.query(Step)
        .filter(Step.flow_id == flow_id)
        .order_by(Step.step_index.asc())
        .all()
    )

    return templates.TemplateResponse(
        "flow_detail.html", {"request": request, "flow": flow, "steps": steps}
    )


@app.get("/assets/{flow_id}/{step_index}/screenshot")
def get_screenshot(
    flow_id: UUID,
    step_index: int,
    db: Session = Depends(get_db),
    storage: StorageBackend = Depends(get_storage),
) -> StreamingResponse:
    step = (
        db.query(Step)
        .filter(Step.flow_id == flow_id, Step.step_index == step_index)
        .first()
    )
    if step is None:
        raise HTTPException(status_code=404, detail="Step not found")

    image_bytes = storage.get_bytes(step.screenshot_key)
    return StreamingResponse(io.BytesIO(image_bytes), media_type="image/png")
