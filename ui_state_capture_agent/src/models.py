from __future__ import annotations
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, create_engine, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, Session, declarative_base, mapped_column, relationship, sessionmaker

from .config import settings

engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Flow(Base):
    __tablename__ = "flows"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    app_name: Mapped[str] = mapped_column(String, nullable=False)
    task_id: Mapped[str] = mapped_column(String, nullable=False)
    task_title: Mapped[str] = mapped_column(String, nullable=False)
    task_blurb: Mapped[str] = mapped_column(String, nullable=False)
    run_id: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    bucket: Mapped[str] = mapped_column(String, nullable=False)
    prefix: Mapped[str] = mapped_column(String, nullable=False)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    status_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    steps: Mapped[list["Step"]] = relationship("Step", back_populates="flow", cascade="all, delete-orphan")
    logs: Mapped[list["FlowLog"]] = relationship("FlowLog", back_populates="flow", cascade="all, delete-orphan")


class Step(Base):
    __tablename__ = "steps"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    flow_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("flows.id"), nullable=False)
    step_index: Mapped[int] = mapped_column(Integer, nullable=False)
    state_label: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(String, nullable=False)
    url: Mapped[str] = mapped_column(String, nullable=False)
    url_changed: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    state_kind: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    screenshot_key: Mapped[str] = mapped_column(String, nullable=False)
    dom_key: Mapped[str] = mapped_column(String, nullable=False)
    diff_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    diff_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    flow: Mapped["Flow"] = relationship("Flow", back_populates="steps")


class FlowLog(Base):
    __tablename__ = "flow_logs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    flow_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("flows.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    level: Mapped[str] = mapped_column(String(16))
    message: Mapped[str] = mapped_column(Text)

    flow: Mapped["Flow"] = relationship("Flow", back_populates="logs")


def log_flow_event(session: Session, flow: Flow, level: str, message: str) -> None:
    log = FlowLog(flow_id=flow.id, level=level, message=message)
    session.add(log)
    session.commit()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    Base.metadata.create_all(engine)
