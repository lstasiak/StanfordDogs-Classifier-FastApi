import base64
import os
from typing import Any, Union

import torch


def get_labels(path_to_dataset):
    class2label = {}
    for dirname in os.listdir(path_to_dataset):
        names = dirname.split("-")
        if len(labs := names[1:]) != 0:
            class2label[dirname] = ("_".join(labs)).capitalize()
    return class2label


def get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.__version__.startswith("2"):
        # check for Apple Silicon Mac acceleration (only in pytorch-nightly)
        if torch.backends.mps.is_available():
            return torch.device("mps")
    else:
        return torch.device("cpu")


def get_user_device(device: str):
    available_devices = {"cpu": torch.device("cpu")}
    if torch.cuda.is_available():
        available_devices["gpu"] = torch.device("cuda:0")
    if torch.__version__.startswith("2"):
        # check for Apple Silicon Mac acceleration (only in pytorch-nightly)
        if torch.backends.mps.is_available():
            available_devices["mps"] = torch.device("mps")

    return available_devices.get(device, "cpu")


def encode_decode_img(
    img: Union[Any, str], serialize=True, encoding="utf-8"
) -> Union[str, bytes]:
    """
    Helper function for decoding/encoding image file,
    so it can pass celery json-encoder and be correctly
    open to pass to the ML model.

    # NOTE regarding purpose of encode_decode_img():

    To pass img file into celery task function, the type has to be json-serializable
    Since bytes-type throws exception, the first idea was to serialize UploadFile using
    jsonable_encoder() and deserialize it inside task function. Unfortunately this encoder
    corrupts every image file after deserializing data inside task.

    :param img: image file opened as bytes/str
    :param serialize: if True serialize the object and returns it in str format
    :param encoding: encoding format. For the purpose of this project - do not change it.
    :return: img coded in bytes or str format
    """
    if serialize:
        return base64.b64encode(img).decode(encoding=encoding)
    else:
        img = img.encode(encoding=encoding)
        return base64.b64decode(img)
