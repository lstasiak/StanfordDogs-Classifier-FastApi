from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

import src.app.routers as routers  # type: ignore

app = FastAPI(title="Stanford Dogs Classifier API")

origins = [
    "http://localhost:8000",
    "localhost:8000",
    "http://127.0.0.1:8000/",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(routers.api)


@app.get("/")
def root():
    return {"message": "Stanford Dogs classifier."}
