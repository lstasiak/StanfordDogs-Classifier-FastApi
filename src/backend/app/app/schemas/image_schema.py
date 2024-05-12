from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, EmailStr

from backend.app.app.models.image_model import BaseImage
from backend.app.app.schemas.user_schema import IUserRead


class BaseOwnerView(BaseModel):
    first_name: str
    last_name: str
    email: EmailStr


class BaseImageView(BaseModel):
    id: UUID
    filename: str
    created_at: datetime | None = None
    updated_at: datetime | None = None
    created_by: UUID | None = None
    owner: BaseOwnerView | None = None
    predictions: dict | None = None


class IImageCreate(BaseImage):
    pass


class IImagePredict(BaseModel):
    predictions: dict
    updated_at: datetime | None = None


class IImageRead(BaseImageView):
    pass

