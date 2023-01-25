from typing import Union

import pandas as pd  # type: ignore
from fastapi import APIRouter, File, HTTPException, UploadFile
from starlette.responses import JSONResponse

from src.app.tasks.tasks import get_task_info, make_predictions  # type: ignore
from src.app.utils import encode_decode_img  # type: ignore
from src.settings import DEFAULT_TRAINING_HISTORY_SAMPLE  # type: ignore

api = APIRouter(prefix="/api", tags=["api"])


@api.post("/predict")
async def predict(file: UploadFile = File(...), device: Union[str, None] = None):
    """
    Make POST to predict class on uploaded image file
    """
    if not file:
        raise HTTPException(status_code=404, detail="Image file not found.")

    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return HTTPException(status_code=400, detail="Image must be jpg or png format!")

    # see the NOTE inside docstring in encode_decode_img function
    decoded_file: str = encode_decode_img(file.file.read(), serialize=True)
    task = make_predictions.delay(decoded_file, device)
    return JSONResponse(task.get())


@api.get("/stats/")
async def get_training_stats():
    """
    Get recent data: F_score, Accuracy, Loss from model training
    from file specified in DEFAULT_TRAINING_HISTORY_SAMPLE
    """
    stats = pd.read_csv(DEFAULT_TRAINING_HISTORY_SAMPLE, index_col=0)
    return JSONResponse(stats.to_dict())


@api.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """
    Return the status of the submitted Task
    """
    return JSONResponse(get_task_info(task_id))
