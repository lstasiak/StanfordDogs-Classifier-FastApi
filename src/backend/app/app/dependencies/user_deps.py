from backend.app.app import crud
from backend.app.app.models.role_model import Role
from backend.app.app.models.user_model import User
from backend.app.app.schemas.user_schema import IUserCreate
from backend.app.app.schemas.user_schema import IUserRead
from backend.app.app.utils.exceptions.common_exceptions import IdNotFoundException
from uuid import UUID
from fastapi import HTTPException, Path, status
from typing_extensions import Annotated


async def user_exists(new_user: IUserCreate) -> IUserCreate:
    user = await crud.user.get_by_email(email=new_user.email)
    if user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="There is already a user with the same email",
        )
    role = await crud.role.get(id=new_user.role_id)
    if not role:
        raise IdNotFoundException(Role, id=new_user.role_id)

    return new_user


async def is_valid_user(
    user_id: Annotated[UUID, Path(title="The UUID id of the user")]
) -> IUserRead:
    user = await crud.user.get(id=user_id)
    if not user:
        raise IdNotFoundException(User, id=user_id)

    return user


async def is_valid_user_id(
    user_id: Annotated[UUID, Path(title="The UUID id of the user")]
) -> IUserRead:
    user = await crud.user.get(id=user_id)
    if not user:
        raise IdNotFoundException(User, id=user_id)

    return user_id