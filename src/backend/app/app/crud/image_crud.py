from uuid import UUID

from sqlmodel import select, col
from sqlmodel.ext.asyncio.session import AsyncSession

from backend.app.app.crud.base_crud import CRUDBase
from backend.app.app.db.session import SessionLocal
from backend.app.app.models.image_model import Image
from backend.app.app.schemas.image_schema import IImagePredict, IImageCreate
from backend.app.app.utils.image_processing import encode_decode_img
# from ml.predictors import ImagePredictor
# from settings import DEFAULT_MODEL_LOC
from backend.app.app.utils.fastapi_globals import g
from ml.predictors import ImagePredictor
from settings import DEFAULT_MODEL_LOC


class CRUDImage(CRUDBase[Image, IImageCreate, IImagePredict]):

    def __init__(self, model, classifier):
        super().__init__(model)
        self.classifier = classifier

    async def get_image_by_filename(self, *, filename: str, db_session: AsyncSession | None = None) -> Image:
        db_session = db_session or super().get_db().session
        image = await db_session.execute(select(Image).where(col(Image.filename).ilike(f"{filename}")))
        return image.scalar_one_or_none()

    async def filter_images_by_predictions(self, *, predictions: bool, db_session: AsyncSession | None = None) -> list[
        Image]:
        db_session = db_session or super().get_db().session
        if predictions:
            images = await db_session.execute(select(Image).where(Image.predictions is not None))
        else:
            images = await db_session.execute(select(Image).where(Image.predictions is None))
        return images.scalars().all()

    async def predict_image(self, *, image_id: UUID, device: str):

        async with SessionLocal() as db_session:
            image = await db_session.execute(select(Image).where(Image.id == image_id))
            image = image.scalar_one_or_none()
            if image is not None:
                # decode img file to bytes
                deserialized_file = encode_decode_img(image.file, serialize=False)
            else:
                raise AttributeError("Image not found")
            predictions, _ = self.classifier.predict(file=deserialized_file, device=device)

            # update image predictions field
            setattr(image, "predictions", predictions)

            db_session.add(image)
            await db_session.commit()
            await db_session.refresh(image)
            return image


image_classifier = ImagePredictor(model_path=DEFAULT_MODEL_LOC, as_state_dict=False)
image = CRUDImage(Image, classifier=image_classifier)
