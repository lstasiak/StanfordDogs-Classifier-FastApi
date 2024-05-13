from uuid import UUID

from fastapi import Query, Path
from typing_extensions import Annotated

from backend.app.app import crud
from backend.app.app.models.image_model import Image
from backend.app.app.utils.exceptions import NameNotFoundException, IdNotFoundException, ImageWithoutPredictionsException


async def get_image_by_filename(
        filename: Annotated[str, Query(description="File name to get image from")] = "") -> str:
    image = await crud.image.get_image_by_filename(filename=filename)
    if not image:
        raise NameNotFoundException(Image, name=filename)
    return image


async def get_image_by_id(
        image_id: Annotated[UUID, Path(description="The UUID of the image")]
) -> Image:
    image = await crud.image.get(id=image_id)
    if not image:
        raise IdNotFoundException(Image, id=image_id)
    return image


async def is_valid_image_id(
        image_id: Annotated[UUID, Path(title="The UUID id of the image")]
) -> UUID:
    image = await crud.image.get(id=image_id)
    if not image:
        raise IdNotFoundException(Image, id=image_id)

    return image_id


async def has_image_predictions(image_id: Annotated[UUID, Path(title="The UUID id of the image")]) -> UUID:
    # first check if image exists
    image_id = await is_valid_image_id(image_id)

    image = await crud.image.get(id=image_id)
    if not image.predictions:
        raise ImageWithoutPredictionsException(id=image_id)

    return image_id
