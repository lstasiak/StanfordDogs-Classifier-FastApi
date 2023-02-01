# syntax=docker/dockerfile:1

FROM python:3.9

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="${PYTHONPATH}:/code/"

WORKDIR /code

COPY requirements.txt /code/requirements.txt

RUN pip install --upgrade pip
RUN pip install -r /code/requirements.txt

COPY src /code/src
COPY dataset /code/dataset
COPY resources /code/resources
COPY .env /code/.env
COPY alembic.ini /code/alembic.ini

#CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "80"]
