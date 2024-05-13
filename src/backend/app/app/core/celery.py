from celery import Celery

from backend.app.app.core.config import settings

celery = Celery(
    __name__,
    broker=f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}",
    backend=str(settings.SYNC_CELERY_DATABASE_URI),
    include="backend.app.app.api.celery_task",
)

celery.conf.update({"beat_dburi": str(settings.SYNC_CELERY_BEAT_DATABASE_URI)})
celery.autodiscover_tasks()

# if __name__ == "__main__":
#     celery.start()
