from backend.app.app.schemas.response_schema import IGetResponseBase, create_response
from datetime import datetime
from fastapi import APIRouter
from fastapi_cache.decorator import cache

router = APIRouter()


@router.get("/cached")
@cache(expire=10)
async def get_a_cached_response() -> IGetResponseBase[str | datetime]:
    """
    Gets a cached datetime
    """
    return create_response(data=datetime.now())


@router.get("/no_cached")
async def get_a_normal_response() -> IGetResponseBase[str | datetime]:
    """
    Gets a real-time datetime
    """
    return create_response(data=datetime.now())


# TODO: Add example image count cached/no-cached
