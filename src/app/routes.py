from io import BytesIO

import pandas as pd
from celery.result import AsyncResult
from fastapi import APIRouter, Depends, File, HTTPException, Path, UploadFile
from PIL import Image
from starlette.responses import JSONResponse, StreamingResponse
from starlette.status import HTTP_200_OK, HTTP_201_CREATED, HTTP_404_NOT_FOUND

from src.app.config import API_PREFIX
from src.app.db.repositories.images import ImagesRepository
from src.app.dependencies.database import get_repository
from src.app.models import Device, ImageCreate, ImageInDB
from src.app.tasks.tasks import get_task_info, make_predictions
from src.ml.services import view_prediction
from src.settings import DEFAULT_TRAINING_HISTORY_SAMPLE
from src.utils import encode_decode_img

api = APIRouter(prefix=API_PREFIX, tags=["api"])


@api.post("/upload", response_model=ImageInDB, status_code=HTTP_201_CREATED)
async def upload_image(
    file: UploadFile = File(...),
    repo: ImagesRepository = Depends(get_repository(ImagesRepository)),
    filename: str = None,
    ground_truth: str = None,
):
    """
    Endpoint to handle image object creation
    """
    if not file:
        raise HTTPException(status_code=404, detail="Image file not found.")

    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return HTTPException(status_code=400, detail="Image must be jpg or png format!")

    decoded_file: str = encode_decode_img(file.file.read(), serialize=True)

    image = ImageCreate(filename=filename, file=decoded_file, ground_truth=ground_truth)
    # image_repo = get_repository(ImagesRepository)
    new_image = await repo.create_image(new_image=image)

    return new_image


@api.post("/predict", name="Perform prediction on selected image")
async def predict(img_id: int, device: Device = None):
    """
    Perform predictions on selected image and update object from db.
    """
    if device is not None:
        device = device.value
    else:
        device = "cpu"

    task = make_predictions.delay(img_id, device)
    result = AsyncResult(task.task_id)
    response = {
        "data": {"task_id": task.task_id, "status": result.status},
        "message": "Task received",
    }

    return JSONResponse(response)


@api.get("/images", response_model=None, status_code=HTTP_200_OK)
async def list_images(
    repo: ImagesRepository = Depends(get_repository(ImagesRepository)),
):
    """
    Get the list of images in database.
    """
    images = await repo.list_images()

    return {"data": [image.dict(exclude={"file"}) for image in images]}


@api.get("/images/{img_id}", response_model=None, status_code=HTTP_200_OK)
async def get_image(
    img_id: str, repo: ImagesRepository = Depends(get_repository(ImagesRepository))
):
    """
    Fetch image object by id
    """
    image = await repo.get_image_by_id(id=int(img_id))

    return {"data": image.dict(exclude={"file"})}


@api.get("/images/{img_id}/view", status_code=HTTP_200_OK)
async def view_prediction_result(
    img_id: str, repo: ImagesRepository = Depends(get_repository(ImagesRepository))
):
    """
    View prediction results on selected image.
    """
    image = await repo.get_image_by_id(id=int(img_id))

    predictions = image.predictions
    if predictions is not None:
        bytes_content = encode_decode_img(image.file, serialize=False)
        img = Image.open(BytesIO(bytes_content))
        io = BytesIO()
        view_prediction(img, predictions, ground_truth=image.ground_truth, save=io)
        io.seek(0)

        return StreamingResponse(io, media_type="image/jpeg")
    else:
        return JSONResponse(
            {"message": f"Selected image with id={img_id} has no predictions"}
        )


@api.get("/stats/", status_code=HTTP_200_OK)
async def get_training_stats():
    """
    Get recent data: F_score, Accuracy, Loss from model training
    from file specified in DEFAULT_TRAINING_HISTORY_SAMPLE
    """
    stats = pd.read_csv(DEFAULT_TRAINING_HISTORY_SAMPLE, index_col=0)
    return JSONResponse(stats.to_dict())


@api.get("/tasks/{task_id}", status_code=HTTP_200_OK)
async def get_task_status(task_id: str):
    """
    Return the status of the submitted Task
    """
    return JSONResponse(get_task_info(task_id))


@api.delete("/images/{id}/", response_model=int, name="Delete Image")
async def delete_image_by_id(
    id: int = Path(..., ge=1, title="The ID of the cleaning to delete."),
    repo: ImagesRepository = Depends(get_repository(ImagesRepository)),
) -> int:
    """
    Delete image object from database.
    """
    deleted_id = await repo.delete_image_by_id(id=id)

    if not deleted_id:
        raise HTTPException(
            status_code=HTTP_404_NOT_FOUND, detail="No images found with that id."
        )

    return deleted_id
