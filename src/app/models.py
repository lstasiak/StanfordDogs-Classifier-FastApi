from enum import Enum
from typing import Optional

from pydantic import BaseModel, Json


class Device(Enum):
    cpu = "cpu"
    gpu = "gpu"
    mps = "mps"


class IDModelMixin:
    id: str


class BaseImage(BaseModel):
    filename: Optional[str]
    file: str
    predictions: Optional[Json]
    ground_truth: Optional[str]

    class Config:
        orm_mode = True
        arbitrary_types_allowed = True


class ImageCreate(BaseImage):
    file: str


class ImageUpdate(BaseImage):
    predictions: Json


class ImageInDB(IDModelMixin, BaseImage):
    id: int
    file: str
    predictions: Json
