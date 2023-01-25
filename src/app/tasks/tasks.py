from typing import Any, Dict, Union

from celery.result import AsyncResult  # type: ignore

from src.app.tasks.celery import celery  # type: ignore
from src.app.utils import encode_decode_img  # type: ignore
from src.ml.predictors import ImagePredictor  # type: ignore
from src.settings import DEFAULT_MODEL_LOC, PARAMETERS  # type: ignore
from src.utils import get_user_device

classifier = ImagePredictor(model_path=DEFAULT_MODEL_LOC, as_state_dict=False)


@celery.task
def make_predictions(img_file: str, device: Union[str, None] = None) -> Dict[str, Any]:
    """
    predict the img class and confidence levels on given input image
    """
    deserialized = encode_decode_img(img_file, serialize=False)

    predictions, _ = classifier.predict(file=deserialized, device=device)

    results = {
        "predicted_class": max(predictions, key=predictions.get),
        "confidence": {k: float(v) for k, v in predictions.items()},
        "device": device if get_user_device(device) else str(PARAMETERS["device"]),
    }

    return {"results": results}


def get_task_info(task_id) -> Dict[str, Any]:
    """
    return task info for the given task_id
    """
    task_result = AsyncResult(task_id)
    result = {
        "task_id": task_id,
        "task_status": task_result.status,
        "task_result": task_result.result,
    }
    return result
