import json
from typing import Any, Dict

from asgiref.sync import async_to_sync
from celery.result import AsyncResult
from databases import Database

from src.app.config import DATABASE_URL
from src.app.db.repositories.images import ImagesRepository
from src.app.tasks.celery import celery_app
from src.ml.predictors import ImagePredictor
from src.settings import DEFAULT_MODEL_LOC
from src.utils import encode_decode_img

classifier = ImagePredictor(model_path=DEFAULT_MODEL_LOC, as_state_dict=False)

database = Database(DATABASE_URL)


async def schedule_update_predictions(img_id: int, device: str) -> None:
    """
    asynchronously predict the img class and confidence levels on given input image
    and make db update in separated session.
    """
    await database.connect()

    repo = ImagesRepository(db=database)

    # fetch image by id
    image = await repo.get_image_by_id(id=img_id)

    # decode img file to bytes
    deserialized_file = encode_decode_img(image.file, serialize=False)
    predictions, _ = classifier.predict(file=deserialized_file, device=device)

    # update predictions in db
    await repo.update_image_predictions(id=img_id, predictions=json.dumps(predictions))

    await database.disconnect()


@celery_app.task(name="tasks:make_predictions")
def make_predictions(img_id: int, device: str) -> None:
    """
    run async task in celery to get predictions
    """
    async_to_sync(schedule_update_predictions)(img_id, device)


def get_task_info(task_id) -> Dict[str, Any]:
    """
    return task info for the given task_id
    """
    task_result = AsyncResult(task_id, app=celery_app)
    result = {
        "task_id": task_id,
        "task_status": task_result.status,
        "task_result": task_result.result,
    }
    return result
