# syntax=docker/dockerfile:1
# compatible image connecting python 3.12 and oso
#FROM ghcr.io/multi-py/python-oso:py3.12-LATEST
FROM python:3.12-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="${PYTHONPATH}:/code/"

# copy package oso since it's the only way to use it with python 3.12 on this arch
COPY --from=ghcr.io/multi-py/python-oso:py3.12-slim-LATEST /usr/local/lib/python3.12/site-packages/* /usr/local/lib/python3.12/site-packages/
COPY --from=ghcr.io/multi-py/python-oso:py3.12-slim-LATEST /opt/oso /opt/oso

WORKDIR /code

COPY requirements.txt /code/requirements.txt

RUN pip install --upgrade pip
RUN pip install -r /code/requirements.txt

EXPOSE 8000
