from typing import Union

from fastapi import HTTPException
from starlette.status import HTTP_400_BAD_REQUEST

import src.app.db.queries as queries
from src.app.db.repositories.base import BaseRepository
from src.app.models import ImageCreate, ImageInDB


class ImagesRepository(BaseRepository):
    """ "
    All database actions associated with the Image to predict
    """

    async def create_image(self, *, new_image: ImageCreate) -> ImageInDB:
        query_values = new_image.dict()
        image = await self.db.fetch_one(
            query=queries.CREATE_IMAGE_QUERY, values=query_values
        )
        return ImageInDB(**image._mapping)  # type: ignore

    async def update_image_predictions(self, *, id: int, predictions):
        image = await self.get_image_by_id(id=id)
        if not image:
            return None

        try:
            updated_image = await self.db.fetch_one(
                query=queries.UPDATE_IMAGE_PREDICTIONS_QUERY,
                values={"id": id, "predictions": predictions},
            )
            return ImageInDB(**updated_image._mapping)  # type: ignore

        except Exception as e:
            print(e)
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST, detail="Invalid update params."
            )

    async def get_image_by_id(self, *, id: int) -> Union[ImageInDB, None]:
        image = await self.db.fetch_one(
            query=queries.GET_IMAGE_BY_ID_QUERY, values={"id": id}
        )
        if not image:
            return None
        return ImageInDB(**image._mapping)

    async def list_images(self):
        images = await self.db.fetch_all(query=queries.LIST_ALL_IMAGES_QUERY)
        if len(images) == 0:
            return []
        return [ImageInDB(**image._mapping) for image in images]

    async def delete_image_by_id(self, *, id: int):
        image = await self.get_image_by_id(id=id)

        if not image:
            return None

        deleted_id = await self.db.execute(
            query=queries.DELETE_IMAGE_BY_ID_QUERY, values={"id": id}
        )

        return deleted_id
