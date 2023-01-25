import os

from celery import Celery  # type: ignore

celery = Celery(__name__, include=["src.app.tasks.tasks"])

celery.conf.broker_url = os.environ.get("CELERY_BROKER_URL", "redis://127.0.0.1:6379/1")
celery.conf.result_backend = os.environ.get(
    "CELERY_RESULT_BACKEND", "redis://127.0.0.1:6379/2"
)


if __name__ == "__main__":
    celery.start()
