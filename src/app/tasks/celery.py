import os

from celery import Celery  # type: ignore

celery_app = Celery(__name__, include=["src.app.tasks.tasks"])

celery_app.conf.broker_url = os.environ.get(
    "CELERY_BROKER_URL", "redis://127.0.0.1:6379/1"
)
celery_app.conf.result_backend = os.environ.get(
    "CELERY_RESULT_BACKEND", "redis://127.0.0.1:6379/2"
)


if __name__ == "__main__":
    celery_app.start()
