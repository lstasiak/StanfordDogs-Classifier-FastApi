import json
from typing import Any, Dict

from asgiref.sync import async_to_sync
from celery import states
from celery.exceptions import Ignore  # type: ignore
from celery.result import AsyncResult  # type: ignore
from databases import Database

from src.app.config import DATABASE_URL
from src.app.db.repositories.images import ImagesRepository
from src.app.tasks.celery import celery_app
from src.ml.predictors import ImagePredictor
from src.settings import DEFAULT_MODEL_LOC
from src.utils import encode_decode_img

classifier = ImagePredictor(model_path=DEFAULT_MODEL_LOC, as_state_dict=False)


@celery_app.task(bind=True, name="tasks:make_predictions", ignore_result=True)
def make_predictions(self, img_id: int, device: str) -> None:
    """
    run async task in celery to get predictions
    """
    try:
        return async_to_sync(schedule_update_predictions)(img_id, device)
    except AttributeError:
        self.update_state(state=states.FAILURE)
        raise Ignore()


async def schedule_update_predictions(img_id: int, device: str) -> None:
    """
    asynchronously predict the img class and confidence levels on given input image
    and make db update in separated session.
    """

    async with Database(url=str(DATABASE_URL)) as database:
        repo = ImagesRepository(db=database)
        # fetch image by id

        image = await repo.get_image_by_id(id=img_id)
        if image is not None:
            # decode img file to bytes
            deserialized_file = encode_decode_img(image.file, serialize=False)
        else:
            raise AttributeError
        predictions, _ = classifier.predict(file=deserialized_file, device=device)

        # update predictions in db
        image_updated = await repo.update_image_predictions(
            id=img_id, predictions=json.dumps(predictions)
        )

    return image_updated.dict(exclude={"file"})


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
