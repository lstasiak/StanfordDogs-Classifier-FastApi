import time
from io import BytesIO
from uuid import UUID

from fastapi import APIRouter, Depends, status, UploadFile, File, HTTPException
from fastapi_pagination import Params
from starlette.responses import StreamingResponse

from backend.app.app import crud
from backend.app.app.api import api_deps
from backend.app.app.api.celery_task import make_predictions
from backend.app.app.dependencies import image_deps
from backend.app.app.models import User
from backend.app.app.models.image_model import Image
from backend.app.app.schemas.common_schema import Device
from backend.app.app.schemas.image_schema import IImageRead
from backend.app.app.schemas.response_schema import IGetResponsePaginated, create_response, IGetResponseBase, \
    IPostResponseBase, IDeleteResponseBase, IPutResponseBase
from backend.app.app.utils.exceptions import NameExistException
from backend.app.app.utils.image_processing import encode_decode_img
from PIL import Image as PILImage

from ml.services import view_prediction

router = APIRouter()


@router.get("")
async def get_images(
        params: Params = Depends(),
        current_user: User = Depends(api_deps.get_current_user()),
) -> IGetResponsePaginated[IImageRead]:
    """
    Gets a paginated list of images
    """
    images = await crud.image.get_multi_paginated(params=params)

    return create_response(data=images)


@router.get("/{image_id}")
async def get_image_by_id(
        image: Image = Depends(image_deps.get_image_by_id),
        current_user: User = Depends(api_deps.get_current_user()),
) -> IGetResponseBase[IImageRead]:
    """
    Gets an image by its id
    """
    return create_response(data=image)


@router.get("/{file_name}")
async def get_image_by_filename(
        image: Image = Depends(image_deps.get_image_by_filename),
        current_user: User = Depends(api_deps.get_current_user()),
) -> IGetResponseBase[IImageRead]:
    """
    Gets an image by filename
    """
    return create_response(data=image)


@router.post("", status_code=status.HTTP_201_CREATED)
async def upload_image(filename: str, file: UploadFile = File(...), ground_truth: str | None = None,
                       current_user: User = Depends(
                           api_deps.get_current_user())) -> IPostResponseBase[IImageRead]:
    current_image = await crud.image.get_image_by_filename(filename=filename)
    if current_image:
        raise NameExistException(Image, name=current_image.filename)

    if file.content_type not in [f"image/{ext}" for ext in ("jpg", "jpeg", "png")]:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid image file extension.",
        )

    decoded_file: str = encode_decode_img(file.file.read(), serialize=True)

    image = Image(filename=filename, file=decoded_file, ground_truth=ground_truth, created_by=current_user.id,
                  owner=current_user)
    new_image = await crud.image.create(obj_in=image)
    return create_response(data=new_image)


@router.put("/{image_id}", status_code=status.HTTP_202_ACCEPTED)
async def predict(image_id: UUID = Depends(image_deps.is_valid_image_id), device: Device | None = None,
                  current_user: User = Depends(
                           api_deps.get_current_user())) -> \
        IPutResponseBase:
    if device is not None:
        device_val = device.value
    else:
        device_val = "cpu"

    task = make_predictions.delay(image_id=image_id, device=device_val)
    # time.sleep(0.2)
    return create_response(message="Image prediction task received successfully", data={"task_id": task.task_id})


@router.get("/{image_id}/view", status_code=status.HTTP_200_OK)
async def view_image_with_predictions(image_id: UUID = Depends(image_deps.has_image_predictions)):
    image = await crud.image.get(id=image_id)
    bytes_content = encode_decode_img(image.file, serialize=False)
    img = PILImage.open(BytesIO(bytes_content))
    io = BytesIO()
    view_prediction(img, image.predictions, ground_truth=image.ground_truth, save=io)
    io.seek(0)

    return StreamingResponse(io, media_type="image/jpeg")


@router.delete("/{image_id}")
async def remove_image(
        image_id: UUID = Depends(image_deps.is_valid_image_id),
        current_user: User = Depends(
            api_deps.get_current_user()
        ),
) -> IDeleteResponseBase[IImageRead]:
    """
    Deletes image by its id
    """
    image = await crud.image.remove(id=image_id)
    return create_response(data=image, message="Image removed")
