from __future__ import annotations

import argparse
import csv
import json
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import asc, desc, select
from sqlalchemy.engine import Row
from sqlalchemy.exc import SQLAlchemyError

from src.models import Base, Flow, FlowLog, SessionLocal, Step
from src.storage.minio_store import get_storage


def _serialize(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def export_table_data(db, db_dir: Path) -> dict[str, Any]:
    """Export all ORM tables to CSV and return schema metadata."""
    db_dir.mkdir(parents=True, exist_ok=True)
    schema: dict[str, Any] = {}

    for table_name, table in Base.metadata.tables.items():
        columns = [col.name for col in table.columns]
        schema[table_name] = {
            "columns": [
                {
                    "name": col.name,
                    "type": str(col.type),
                    "nullable": col.nullable,
                    "primary_key": col.primary_key,
                    "default": str(col.default.arg) if col.default is not None else None,
                }
                for col in table.columns
            ]
        }

        csv_path = db_dir / f"{table_name}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            try:
                rows = db.execute(select(table)).fetchall()
            except SQLAlchemyError as exc:
                writer.writerow([f"ERROR: {exc}"])
                continue
            for row in rows:
                if isinstance(row, Row):
                    values = [row._mapping.get(col) for col in columns]
                else:
                    values = [getattr(row, col, None) for col in columns]
                writer.writerow([_serialize(v) for v in values])

    (db_dir / "schema.json").write_text(json.dumps(schema, indent=2))
    return schema


def export_flows(out_dir: Path, statuses: list[str], limit_flows: int | None) -> dict[str, Any]:
    """Export flows, steps, screenshots, and DOM snapshots."""
    manifest = []
    summary: dict[str, Any] = {"flows": 0, "steps": 0, "errors": []}

    try:
        storage = get_storage()
    except Exception as exc:  # noqa: BLE001
        storage = None
        summary["errors"].append(f"Storage init failed: {exc}")

    with SessionLocal() as db:
        query = db.query(Flow).order_by(desc(Flow.started_at))
        if statuses:
            query = query.filter(Flow.status.in_(statuses))
        if limit_flows and limit_flows > 0:
            query = query.limit(limit_flows)

        flows = query.all()

        for flow in flows:
            flow_dir = out_dir / str(flow.id)
            steps_dir = flow_dir / "steps"
            flow_dir.mkdir(parents=True, exist_ok=True)
            steps_dir.mkdir(parents=True, exist_ok=True)

            steps = (
                db.query(Step)
                .filter(Step.flow_id == flow.id)
                .order_by(asc(Step.step_index))
                .all()
            )
            logs = (
                db.query(FlowLog)
                .filter(FlowLog.flow_id == flow.id)
                .order_by(asc(FlowLog.created_at))
                .all()
            )
            steps_meta = []
            for step in steps:
                summary["steps"] += 1
                ss_path = steps_dir / f"{step.step_index:03d}_screenshot.png"
                dom_path = steps_dir / f"{step.step_index:03d}_dom.html"

                if storage:
                    try:
                        ss_path.write_bytes(storage.get_bytes(step.screenshot_key))
                    except Exception as exc:  # noqa: BLE001
                        summary["errors"].append(f"{flow.id} step {step.step_index} screenshot: {exc}")
                    try:
                        dom_path.write_bytes(storage.get_bytes(step.dom_key))
                    except Exception as exc:  # noqa: BLE001
                        summary["errors"].append(f"{flow.id} step {step.step_index} dom: {exc}")

                steps_meta.append(
                    {
                        "index": step.step_index,
                        "state_label": step.state_label,
                        "description": step.description,
                        "url": step.url,
                        "url_changed": step.url_changed,
                        "state_kind": step.state_kind,
                        "diff_summary": step.diff_summary,
                        "diff_score": step.diff_score,
                        "screenshot": ss_path.name,
                        "dom": dom_path.name,
                    }
                )

            (flow_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "flow_id": str(flow.id),
                        "app_name": flow.app_name,
                        "task_title": flow.task_title,
                        "task_blurb": flow.task_blurb,
                        "run_id": flow.run_id,
                        "status": flow.status,
                        "status_reason": flow.status_reason,
                        "started_at": flow.started_at.isoformat() if flow.started_at else None,
                        "finished_at": flow.finished_at.isoformat() if flow.finished_at else None,
                        "steps": steps_meta,
                        "logs": [
                            {
                                "created_at": log.created_at.isoformat(),
                                "level": log.level,
                                "message": log.message,
                            }
                            for log in logs
                        ],
                    },
                    indent=2,
                )
            )

            manifest.append({"flow_id": str(flow.id), "status": flow.status, "dir": flow_dir.name})
            summary["flows"] += 1

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return summary


def export_bundle(out_dir: Path, statuses: list[str], limit_flows: int | None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    db_dir = out_dir / "db"
    with SessionLocal() as db:
        tables_schema = export_table_data(db, db_dir)
    flows_summary = export_flows(out_dir, statuses, limit_flows)

    summary = {
        "tables_exported": list(tables_schema.keys()),
        "flows_exported": flows_summary["flows"],
        "steps_exported": flows_summary["steps"],
        "errors": flows_summary["errors"],
    }
    (out_dir / "export_summary.json").write_text(json.dumps(summary, indent=2))
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="Export flows, artifacts, and DB tables.")
    parser.add_argument("--out", required=True, help="Output directory (recommended under dataset/)")
    parser.add_argument(
        "--status",
        action="append",
        default=[],
        help="Flow status to include (repeatable). Omit to include all.",
    )
    parser.add_argument(
        "--limit-flows",
        type=int,
        default=10,
        help="Export only the most recent N flows (by started_at desc). Use 0 for no limit.",
    )
    parser.add_argument("--tar", help="Optional path to write a .tar.gz archive")
    args = parser.parse_args()

    out_dir = export_bundle(Path(args.out), args.status, args.limit_flows)

    if args.tar:
        with tarfile.open(args.tar, "w:gz") as tf:
            tf.add(out_dir, arcname=Path(args.out).name)
        print(f"Archive written to {args.tar}")

    print(f"Exported to {out_dir}")


if __name__ == "__main__":
    main()
