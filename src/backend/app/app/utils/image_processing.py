import base64
from typing import Any


def encode_decode_img(img: Any, serialize: bool = True, encoding: str = "utf-8") -> Any:
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