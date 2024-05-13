from datetime import datetime
from enum import Enum

from sqlalchemy import Column, DateTime, func, JSON
from sqlmodel import SQLModel, Field, Relationship

from backend.app.app.models import User
from backend.app.app.models.base_uuid_model import BaseUUIDModel
from uuid import UUID


class BaseImage(SQLModel):
    file: str
    filename: str = Field(index=True, min_length=1, max_length=255)
    # created_at: datetime = Field(sa_column=Column(DateTime(timezone=True), nullable=True, server_default=func.now()))
    # updated_at: datetime | None = Field(
    #     default=None,
    #     sa_column=Column(DateTime(timezone=True), nullable=True, server_default=func.now(), server_onupdate=func.now())
    # )
    ground_truth: str | None = None


class Image(BaseUUIDModel, BaseImage, table=True):
    created_by: UUID | None = Field(default=None, foreign_key="User.id")
    owner: "User" = Relationship(
        sa_relationship_kwargs={
            "lazy": "joined",
            "primaryjoin": "Image.created_by==User.id",
        }
    )
    predictions: dict | None = Field(default=None, sa_column=Column(JSON, nullable=False))

    class Config:
        arbitrary_types_allowed = True
