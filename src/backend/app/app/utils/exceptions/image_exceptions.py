from typing import Optional, Dict, Any, Union
from uuid import UUID

from fastapi import HTTPException, status


class ImageWithoutPredictionsException(HTTPException):
    def __init__(
        self,
        headers: Optional[Dict[str, Any]] = None,
        id: Optional[Union[UUID, str]] = None,
    ) -> None:
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image with id = {id} has no predictions to view",
            headers=headers,
        )