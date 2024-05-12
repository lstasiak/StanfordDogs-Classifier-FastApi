from pydantic import BaseModel
from enum import Enum
from backend.app.app.schemas.role_schema import IRoleRead


class Device(str, Enum):
    cpu = "cpu"
    gpu = "gpu"
    mps = "mps"


class IGenderEnum(str, Enum):
    female = "female"
    male = "male"
    other = "other"


class IMetaGeneral(BaseModel):
    roles: list[IRoleRead]


class IOrderEnum(str, Enum):
    ascendant = "ascendant"
    descendant = "descendant"


class TokenType(str, Enum):
    ACCESS = "access_token"
    REFRESH = "refresh_token"
