import io
from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import Compose

from src.ml.data_managers import ImageTransformer
from src.ml.services import get_default_model, view_prediction  # type: ignore
from src.settings import DEFAULT_CLASS_NAMES, PARAMETERS  # type: ignore
from src.utils import get_labels  # type: ignore
from src.utils import get_user_device


class ImagePredictor:
    """
    Class representation of the object to load saved model and perform predictions.
    """

    def __init__(
        self,
        model_instance: Union[nn.Module, None] = None,
        model_path: Union[str, None] = None,
        as_state_dict: bool = True,
    ) -> None:
        if model_instance is None and model_path is None:
            raise AssertionError(
                "Either model or path-to-saved-model must be specified."
            )

        if model_instance is not None:
            self.model = model_instance
        elif model_path is not None:
            if as_state_dict:
                self.model = get_default_model()
                self.model.load_state_dict(
                    torch.load(model_path, map_location=PARAMETERS["device"])
                )
            else:
                self.model = torch.jit.load(
                    model_path, map_location=PARAMETERS["device"]
                )
        self.model.to(PARAMETERS["device"])
        self.model.eval()
        self.img_transforms: Compose = ImageTransformer.get_image_transforms(
            img_size=PARAMETERS["img_size"], phase="val"
        )

    def predict(
        self,
        img_path: Union[str, None] = None,
        file=None,
        device: Union[str, None] = None,
    ) -> Tuple[Dict[str, float], Image.Image]:
        """
        Main method to perform prediction on input image.
        You need to pass at least one of the following arguments:
         img_path or file
        """
        if img_path:
            image = Image.open(img_path)
        else:
            image = Image.open(io.BytesIO(file))

        user_dev = get_user_device(device)
        self.model.to(user_dev)

        orig_img = image.copy()
        trf_image = self.img_transforms(image).to(user_dev).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(trf_image)
        predictions = torch.nn.functional.softmax(outputs, dim=1)
        pred_values = predictions.cpu().data.numpy().squeeze()
        data = {cls: float(pred_values[i]) for i, cls in enumerate(DEFAULT_CLASS_NAMES)}

        sorted_by_prob = sorted(data.items(), key=lambda x: x[1], reverse=True)
        sorted_by_prob_map = dict(sorted_by_prob)
        return sorted_by_prob_map, orig_img
