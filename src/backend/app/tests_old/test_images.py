# import json
# import os.path
# from unittest.mock import patch
#
# import pytest
# from databases import Database
# from fastapi import FastAPI
# from httpx import AsyncClient
# from starlette.status import (
#     HTTP_200_OK,
#     HTTP_201_CREATED,
#     HTTP_204_NO_CONTENT,
#     HTTP_404_NOT_FOUND,
#     HTTP_422_UNPROCESSABLE_ENTITY,
# )
#
# from backend.app.db import ImagesRepository
# from app.models import ImageInDB
# from settings import PROJECT_ROOT
#
#
# class TestCreateImages:
#     test_resource_path = os.path.join(PROJECT_ROOT, "resources", "test_resources")
#
#     @pytest.mark.asyncio
#     async def test_upload_image(self, app: FastAPI, client: AsyncClient) -> None:
#         """
#         Test if image is being created.
#         """
#         filename = "test_image_file.jpg"
#         filepath = os.path.join(self.test_resource_path, filename)
#
#         with open(filepath, "rb") as file:
#             response = await client.post(
#                 app.url_path_for("upload_image"),
#                 files={"file": (filename, file, "image/jpg")},
#                 params={"filename": filename, "ground_truth": "border_collie"},
#             )
#         assert response.status_code == HTTP_201_CREATED
#         assert response.json()["filename"] == filename
#         assert response.json()["ground_truth"] == "border_collie"
#         assert response.json()["predictions"] is None
#
#     @pytest.mark.asyncio
#     async def test_upload_image_invalid_file_ext(
#         self, app: FastAPI, client: AsyncClient
#     ) -> None:
#         """
#         Test if image file extension is correctly validated.
#         """
#         filename = "invalid_file.txt"
#         filepath = os.path.join(PROJECT_ROOT, "resources", "test_resources", filename)
#
#         with open(filepath, "rb") as file:
#             response = await client.post(
#                 app.url_path_for("upload_image"), files={"file": file}
#             )
#         assert response.status_code == HTTP_422_UNPROCESSABLE_ENTITY
#         assert response.json()["detail"] == "Invalid image file extension."
#
#
# class TestGetImages:
#     @pytest.mark.asyncio
#     async def test_list_images(self, app: FastAPI, client: AsyncClient) -> None:
#         """
#         Test get list of images stored in db
#         """
#         response = await client.get(app.url_path_for("list_images"))
#         data = response.json()["data"]
#
#         assert response.status_code == HTTP_200_OK
#         assert len(data) == 1
#         # check if "file" field is excluded from response
#         assert sorted(list(data[0].keys())) == sorted(
#             ["filename", "id", "predictions", "ground_truth"]
#         )
#         assert data[0]["ground_truth"] == "border_collie"
#
#     @pytest.mark.asyncio
#     async def test_get_image_by_id(
#         self, app: FastAPI, client: AsyncClient, test_image: ImageInDB
#     ) -> None:
#         response = await client.get(
#             app.url_path_for("get_image_by_id", img_id=test_image.id)
#         )
#         data = response.json()["data"]
#
#         assert response.status_code == HTTP_200_OK
#         assert test_image.dict(exclude={"file"}) == data
#
#     @pytest.mark.parametrize(
#         "img_id, status_code",
#         (
#             (500, HTTP_404_NOT_FOUND),
#             (-1, HTTP_422_UNPROCESSABLE_ENTITY),
#             (None, HTTP_422_UNPROCESSABLE_ENTITY),
#         ),
#     )
#     @pytest.mark.asyncio
#     async def test_get_image_by_invalid_id(
#         self, app: FastAPI, client: AsyncClient, img_id, status_code
#     ) -> None:
#         response = await client.get(app.url_path_for("get_image_by_id", img_id=img_id))
#         assert response.status_code == status_code
#
#
# class TestPerformPredictions:
#     @patch("app.routes.make_predictions.delay", autospec=True)
#     @pytest.mark.asyncio
#     async def test_predict_correctly(
#         self,
#         mock_make_predictions,
#         app: FastAPI,
#         client: AsyncClient,
#         test_image: ImageInDB,
#         celery_session_worker,
#     ):
#         mock_make_predictions.return_value.state = "PENDING"
#         mock_make_predictions.return_value.task_id = "1234"
#         response = await client.post(
#             app.url_path_for("perform_prediction"),
#             params={"img_id": test_image.id, "device": "cpu"},
#         )
#         assert response.status_code == HTTP_200_OK
#         mock_make_predictions.assert_called_once_with(
#             img_id=test_image.id, device="cpu"
#         )
#         assert response.json()["data"]["task_id"] == "1234"
#         assert response.json()["message"] == "Task received"
#
#     @pytest.mark.parametrize(
#         "img_id, status_code",
#         (
#             (500, HTTP_404_NOT_FOUND),
#             (-1, HTTP_422_UNPROCESSABLE_ENTITY),
#             (None, HTTP_422_UNPROCESSABLE_ENTITY),
#         ),
#     )
#     @patch("app.routes.make_predictions.delay")
#     @pytest.mark.asyncio
#     async def test_predict_with_invalid_id(
#         self,
#         mock_make_predictions,
#         app: FastAPI,
#         client: AsyncClient,
#         img_id,
#         status_code,
#     ):
#         mock_make_predictions.return_value.task_id = "1234"
#         mock_make_predictions.return_value.state = "FAILURE"
#
#         response = await client.post(
#             app.url_path_for("perform_prediction"),
#             params={"img_id": img_id, "device": "cpu"},
#         )
#         assert response.status_code == status_code
#
#
# class TestViewPredictionResult:
#     @patch("app.routes.view_prediction")
#     @pytest.mark.asyncio
#     async def test_view_prediction_correctly(
#         self,
#         mock_view_prediction,
#         app: FastAPI,
#         client: AsyncClient,
#         test_image: ImageInDB,
#         db: Database,
#     ):
#         test_predictions = json.dumps({"border_collie": 0.92, "collie": 0.08})
#         repo = ImagesRepository(db)
#         updated = await repo.update_image_predictions(
#             id=test_image.id, predictions=test_predictions
#         )
#
#         response = await client.get(
#             app.url_path_for("view_prediction_results", img_id=updated.id)
#         )
#         assert response.status_code == HTTP_200_OK
#         mock_view_prediction.assert_called_once()
#
#     @patch("app.routes.view_prediction")
#     @pytest.mark.asyncio
#     async def test_no_predictions_to_view(
#         self,
#         mock_view_prediction,
#         app: FastAPI,
#         client: AsyncClient,
#         test_image: ImageInDB,
#     ):
#         """
#         Test for image with no predictions returns HTTP_204_NO_CONTENT
#         """
#         response = await client.get(
#             app.url_path_for("view_prediction_results", img_id=test_image.id)
#         )
#         assert response.status_code == HTTP_204_NO_CONTENT
#         mock_view_prediction.assert_not_called()
#
#     @pytest.mark.parametrize(
#         "img_id, status_code",
#         (
#             (500, HTTP_404_NOT_FOUND),
#             (-1, HTTP_422_UNPROCESSABLE_ENTITY),
#             (None, HTTP_422_UNPROCESSABLE_ENTITY),
#         ),
#     )
#     @pytest.mark.asyncio
#     async def test_view_predictions_with_invalid_id(
#         self, app: FastAPI, client: AsyncClient, img_id, status_code
#     ):
#         response = await client.get(
#             app.url_path_for("view_prediction_results", img_id=img_id)
#         )
#         assert response.status_code == status_code
