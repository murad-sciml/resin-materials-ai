"""
Integrated data schema for eco-friendly binder materials.

Combines experimental, literature, and process-log data into one canonical
relational schema so downstream modules (AI Reverse Design Core, Validation
Protocol Engine, Autonomous Experiment Module) can query a single source of
truth regardless of where a record originated.
"""
from __future__ import annotations

import datetime as dt
from typing import Optional

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

# The "five major properties" tracked per material system. This list is the
# single place to edit if the material system changes.
FIVE_MAJOR_PROPERTIES = [
    "compressive_strength",
    "flexural_strength",
    "viscosity",
    "curing_time",
    "carbon_emission",
]

SOURCE_TYPES = ("experimental", "literature", "process_log", "synthetic")


class Base(DeclarativeBase):
    pass


class Material(Base):
    """One formulation record: a binder recipe plus its process parameters."""

    __tablename__ = "materials"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), index=True)
    source_type: Mapped[str] = mapped_column(String(20), default="experimental")

    # Required-data fields from the core-function table
    resin_voc_ratio: Mapped[Optional[float]] = mapped_column(Float)       # low-VOC resin ratio
    sio2_na2o_ratio: Mapped[Optional[float]] = mapped_column(Float)       # SiO2 / Na2O molar ratio
    additives: Mapped[Optional[dict]] = mapped_column(JSON)               # {name: fraction, ...}
    curing_method: Mapped[Optional[str]] = mapped_column(String(80))

    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)

    properties: Mapped[list["MaterialProperty"]] = relationship(
        back_populates="material", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return f"<Material {self.name} ({self.source_type})>"


class MaterialProperty(Base):
    """One measured/predicted property value for a Material (long/tidy format)."""

    __tablename__ = "material_properties"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    material_id: Mapped[int] = mapped_column(ForeignKey("materials.id"))
    property_name: Mapped[str] = mapped_column(String(60))
    property_value: Mapped[float] = mapped_column(Float)
    unit: Mapped[Optional[str]] = mapped_column(String(30))
    is_synthetic: Mapped[bool] = mapped_column(default=False)

    material: Mapped["Material"] = relationship(back_populates="properties")


class IngestLog(Base):
    """Audit trail + data-quality metrics for every ETL ingestion run."""

    __tablename__ = "ingest_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_format: Mapped[str] = mapped_column(String(20))          # csv / json / xlsx
    source_type: Mapped[str] = mapped_column(String(20))            # experimental / literature / process_log
    ingested_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    record_count: Mapped[int] = mapped_column(Integer, default=0)
    completeness: Mapped[Optional[float]] = mapped_column(Float)
    missing_value_rate: Mapped[Optional[float]] = mapped_column(Float)
    outlier_reduction_rate: Mapped[Optional[float]] = mapped_column(Float)
    schema_consistency: Mapped[Optional[float]] = mapped_column(Float)
    notes: Mapped[Optional[str]] = mapped_column(String(500))


def get_engine(db_path: str = "sqlite:///data_integration.db"):
    return create_engine(db_path, echo=False)


def init_db(engine=None):
    engine = engine or get_engine()
    Base.metadata.create_all(engine)
    return engine


def get_session(engine=None):
    engine = engine or init_db()
    return sessionmaker(bind=engine)()
