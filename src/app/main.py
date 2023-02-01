from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse

import src.app.routes as routes  # type: ignore
from src.app import handlers
from src.app.config import PROJECT_NAME, VERSION


def get_fastapi_application():
    fastapi_app = FastAPI(title=PROJECT_NAME, version=VERSION)

    origins = [
        "http://localhost:8000",
        "localhost:8000",
        "http://127.0.0.1:8000/",
    ]

    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    fastapi_app.add_event_handler(
        "startup", handlers.create_start_app_handler(fastapi_app)
    )
    fastapi_app.add_event_handler(
        "shutdown", handlers.create_stop_app_handler(fastapi_app)
    )

    fastapi_app.include_router(routes.api)

    return fastapi_app


app = get_fastapi_application()


@app.get("/")
def root():
    """Basic HTML response."""
    body = (
        "<html>"
        "<body style='padding: 10px;'>"
        f"<h1>Welcome to {PROJECT_NAME}</h1>"
        "<div>"
        "Check the docs: <a href='/docs'>here</a>"
        "</div>"
        "</body>"
        "</html>"
    )

    return HTMLResponse(content=body)
