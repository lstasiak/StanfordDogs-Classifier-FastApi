from asgiref.sync import async_to_sync
from celery import states
from celery.exceptions import Ignore

from backend.app.app import crud
from backend.app.app.core.celery import celery


@celery.task(bind=True, name="tasks:make_predictions",
             task_name="image classification", ignore_result=True)
def make_predictions(self, image_id: int, device: str) -> None:
    """
    run async task in celery to get predictions
    """
    try:
        return async_to_sync(crud.image.predict_image)(image_id=image_id, device=device)
    except AttributeError:
        self.update_state(state=states.FAILURE)
        raise Ignore()
