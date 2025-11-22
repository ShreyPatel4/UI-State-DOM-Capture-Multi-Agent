from __future__ import annotations

import uuid

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, create_engine, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from .config import settings

engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Flow(Base):
    __tablename__ = "flows"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    app_name = Column(String, nullable=False)
    task_id = Column(String, nullable=False)
    task_title = Column(String, nullable=False)
    task_blurb = Column(String, nullable=False)
    run_id = Column(String, nullable=False)
    status = Column(String, nullable=False)
    started_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    finished_at = Column(DateTime(timezone=True), nullable=True)
    bucket = Column(String, nullable=False)
    prefix = Column(String, nullable=False)

    steps = relationship("Step", back_populates="flow", cascade="all, delete-orphan")


class Step(Base):
    __tablename__ = "steps"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    flow_id = Column(UUID(as_uuid=True), ForeignKey("flows.id"), nullable=False)
    step_index = Column(Integer, nullable=False)
    state_label = Column(String, nullable=False)
    description = Column(String, nullable=False)
    url = Column(String, nullable=False)
    screenshot_key = Column(String, nullable=False)
    dom_key = Column(String, nullable=False)
    diff_summary = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    flow = relationship("Flow", back_populates="steps")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    Base.metadata.create_all(engine)
