from typing import Annotated
from uuid import UUID
from backend.app.app.utils.exceptions import (
    UserSelfDeleteException,
)
from backend.app.app import crud
from backend.app.app.api import api_deps
from backend.app.app.dependencies import user_deps
from backend.app.app.models import User
from backend.app.app.models.role_model import Role

from fastapi import (
    APIRouter,
    Depends,
    Query,
    status,
)
from backend.app.app.schemas.response_schema import (
    IDeleteResponseBase,
    IGetResponseBase,
    IGetResponsePaginated,
    IPostResponseBase,
    create_response,
)
from backend.app.app.schemas.role_schema import IRoleEnum
from backend.app.app.schemas.user_schema import (
    IUserCreate,
    IUserRead,
    IUserReadWithoutGroups,
    IUserStatus,
)

from fastapi_pagination import Params
from sqlmodel import and_, select, col, or_, text

router = APIRouter()


@router.get("/list")
async def read_users_list(
    params: Params = Depends(),
    current_user: User = Depends(
        api_deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.manager])
    ),
) -> IGetResponsePaginated[IUserReadWithoutGroups]:
    """
    Retrieve users. Requires admin or manager role

    Required roles:
    - admin
    - manager
    """
    users = await crud.user.get_multi_paginated(params=params)
    return create_response(data=users)


@router.get("/list/by_role_name")
async def read_users_list_by_role_name(
    name: str = "",
    user_status: Annotated[
        IUserStatus,
        Query(
            title="User status",
            description="User status, It is optional. Default is active",
        ),
    ] = IUserStatus.active,
    role_name: str = "",
    params: Params = Depends(),
    current_user: User = Depends(
        api_deps.get_current_user(required_roles=[IRoleEnum.admin])
    ),
) -> IGetResponsePaginated[IUserReadWithoutGroups]:
    """
    Retrieve users by role name and status. Requires admin role

    Required roles:
    - admin
    """
    user_status = True if user_status == IUserStatus.active else False
    query = (
        select(User)
        .join(Role, User.role_id == Role.id)
        .where(
            and_(
                col(Role.name).ilike(f"%{role_name}%"),
                User.is_active == user_status,
                or_(
                    col(User.first_name).ilike(f"%{name}%"),
                    col(User.last_name).ilike(f"%{name}%"),
                    text(
                        f"""'{name}' % concat("User".last_name, ' ', "User".first_name)"""
                    ),
                    text(
                        f"""'{name}' % concat("User".first_name, ' ', "User".last_name)"""
                    ),
                ),
            )
        )
        .order_by(User.first_name)
    )
    users = await crud.user.get_multi_paginated(query=query, params=params)
    return create_response(data=users)


@router.get("/order_by_created_at")
async def get_user_list_order_by_created_at(
    params: Params = Depends(),
    current_user: User = Depends(
        api_deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.manager])
    ),
) -> IGetResponsePaginated[IUserReadWithoutGroups]:
    """
    Gets a paginated list of users ordered by created datetime

    Required roles:
    - admin
    - manager
    """
    users = await crud.user.get_multi_paginated_ordered(
        params=params, order_by="created_at"
    )
    return create_response(data=users)



@router.get("/{user_id}")
async def get_user_by_id(
    user: User = Depends(user_deps.is_valid_user),
    current_user: User = Depends(
        api_deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.manager])
    ),
) -> IGetResponseBase[IUserRead]:
    """
    Gets a user by his/her id

    Required roles:
    - admin
    - manager
    """
    return create_response(data=user)


@router.get("")
async def get_my_data(
    current_user: User = Depends(api_deps.get_current_user()),
) -> IGetResponseBase[IUserRead]:
    """
    Gets my user profile information
    """
    return create_response(data=current_user)


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_user(
    new_user: IUserCreate = Depends(user_deps.user_exists),
    current_user: User = Depends(
        api_deps.get_current_user(required_roles=[IRoleEnum.admin])
    ),
) -> IPostResponseBase[IUserRead]:
    """
    Creates a new user

    Required roles:
    - admin
    """
    user = await crud.user.create_with_role(obj_in=new_user)
    return create_response(data=user)


@router.delete("/{user_id}")
async def remove_user(
    user_id: UUID = Depends(user_deps.is_valid_user_id),
    current_user: User = Depends(
        api_deps.get_current_user(required_roles=[IRoleEnum.admin])
    ),
) -> IDeleteResponseBase[IUserRead]:
    """
    Deletes a user by his/her id

    Required roles:
    - admin
    """
    if current_user.id == user_id:
        raise UserSelfDeleteException()

    user = await crud.user.remove(id=user_id)
    return create_response(data=user, message="User removed")
