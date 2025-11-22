from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
from typing import Any, List
from uuid import UUID
import uuid

from fastapi import Depends, FastAPI, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..agent.orchestrator import run_task_query_async
from ..models import Flow, FlowLog, Step, get_db
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


class FlowSummary(BaseModel):
    id: str
    app_name: str
    task_title: str
    status: str
    started_at: datetime
    finished_at: datetime | None
    run_id: str


class StepSummary(BaseModel):
    index: int
    state_label: str
    description: str
    url: str | None


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


@app.post("/run_from_ui")
async def run_from_ui(
    request: Request, query: str = Form(...), db: Session = Depends(get_db)
):
    try:
        flow = await run_task_query_async(query)
    except Exception as exc:  # noqa: BLE001
        flows = db.query(Flow).order_by(Flow.started_at.desc()).limit(50).all()
        return templates.TemplateResponse(
            "flows_list.html",
            {
                "request": request,
                "flows": flows,
                "error_message": "Could not start run: " + str(exc),
            },
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    return RedirectResponse(url=f"/flows/{flow.id}", status_code=303)


@app.get("/", response_class=HTMLResponse)
def list_flows(request: Request, db: Session = Depends(get_db)) -> Any:
    flows = db.query(Flow).order_by(Flow.started_at.desc()).limit(50).all()
    return templates.TemplateResponse(
        "flows_list.html", {"request": request, "flows": flows}
    )


@app.get("/api/flows", response_model=List[FlowSummary])
def list_flows_json(db=Depends(get_db)):
    flows = db.query(Flow).order_by(Flow.started_at.desc()).limit(50).all()
    return [
        FlowSummary(
            id=str(f.id),
            app_name=f.app_name,
            task_title=f.task_title,
            status=f.status,
            started_at=f.started_at,
            finished_at=f.finished_at,
            run_id=f.run_id,
        )
        for f in flows
    ]


@app.get("/api/flows/{flow_id}/steps", response_model=List[StepSummary])
def list_flow_steps(flow_id: str, db=Depends(get_db)):
    flow = db.query(Flow).get(flow_id)
    if not flow:
        raise HTTPException(status_code=404, detail="Flow not found")
    steps = (
        db.query(Step)
        .filter(Step.flow_id == flow_id)
        .order_by(Step.step_index.asc())
        .all()
    )
    return [
        StepSummary(
            index=s.step_index,
            state_label=s.state_label,
            description=s.description,
            url=s.url,
        )
        for s in steps
    ]


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
        "cancel_requested": flow.cancel_requested,
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


@app.get("/flows/{flow_id}/logs")
def flow_logs(flow_id: uuid.UUID, db: Session = Depends(get_db)):
    logs = (
        db.query(FlowLog)
        .filter(FlowLog.flow_id == flow_id)
        .order_by(FlowLog.created_at.asc())
        .all()
    )
    return [
        {"created_at": l.created_at.isoformat(), "level": l.level, "message": l.message}
        for l in logs
    ]


@app.post("/flows/{flow_id}/cancel")
def cancel_flow(flow_id: UUID, db: Session = Depends(get_db)):
    flow = db.get(Flow, flow_id)
    if not flow:
        raise HTTPException(status_code=404, detail="Flow not found")
    flow.cancel_requested = True
    db.commit()
    return {"id": str(flow.id), "status": flow.status, "cancel_requested": flow.cancel_requested}


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
