import os
import warnings

import alembic
import pytest
import pytest_asyncio
from alembic.config import Config
from asgi_lifespan import LifespanManager
from databases import Database
from fastapi import FastAPI
from httpx import AsyncClient

from src.app.db.repositories.images import ImagesRepository
from src.app.models import ImageCreate, ImageInDB
from src.settings import PROJECT_ROOT
from src.utils import encode_decode_img


# Apply migrations at beginning and end of testing session
# setting scope allows us to use one db session for all tests
@pytest.fixture(scope="session")
def apply_migrations():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    os.environ["TESTING"] = "1"
    config = Config("alembic.ini")
    alembic.command.upgrade(config, "head")
    yield
    alembic.command.downgrade(config, "base")


# Create new application for testing and make it available in tests
@pytest.fixture
def app(apply_migrations: None) -> FastAPI:
    from src.app.main import get_fastapi_application

    return get_fastapi_application()


# Grab a reference to our database when needed
@pytest.fixture
def db(app: FastAPI) -> Database:
    return app.state._db


# Make fastapi client available in all tests
@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncClient:
    async with LifespanManager(app):
        async with AsyncClient(
            app=app,
            base_url="http://testserver",
        ) as test_client:
            yield test_client


@pytest_asyncio.fixture
async def test_image(db: Database) -> ImageInDB:
    test_resource_path = os.path.join(PROJECT_ROOT, "resources", "test_resources")
    filename = "test_image_file.jpg"
    filepath = os.path.join(test_resource_path, filename)
    repo = ImagesRepository(db)

    with open(filepath, "rb") as file:
        image = ImageCreate(
            filename="test_image_file.jpg",
            file=encode_decode_img(file.read(), serialize=True),
            ground_truth="collie",
        )
    return await repo.create_image(new_image=image)


@pytest.fixture
def celery_app(app: FastAPI):
    from celery.contrib.testing import tasks

    yield celery_app


@pytest.fixture(scope="session")
def celery_config():
    return {"broker_url": "memory://", "result_backend": "redis://"}


@pytest.fixture(scope="session")
def celery_worker_parameters():
    """
    Redefine this fixture to change the init parameters of Celery workers.
    This can be used e.g. to define queues the worker will consume tasks from.

    The dict returned by your fixture will then be used
    as parameters when instantiating :class:`~celery.worker.WorkController`.
    """
    return {
        # For some reason this `celery.ping` is not registed IF our own worker is still
        # running. To avoid failing tests in that case, we disable the ping check.
        # see: https://github.com/celery/celery/issues/3642#issuecomment-369057682
        # here is the ping task: `from celery.contrib.testing.tasks import ping`
        "perform_ping_check": False,
        "task_always_eager": True,
    }
