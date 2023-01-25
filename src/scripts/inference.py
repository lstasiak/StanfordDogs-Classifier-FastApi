import argparse

from colorama import Fore, Style
from colorama import init as colorama_init

from ml.predictors import ImagePredictor  # type: ignore
from ml.services import view_prediction
from settings import (  # type: ignore
    DEFAULT_MODEL_LOC,
    DEFAULT_TEST_SAMPLE,
    DEFAULT_TEST_SAMPLE_GT,
)


def main():
    predictor = ImagePredictor(model_path=args["model_path"], as_state_dict=False)

    preds, img = predictor.predict(img_path=args["image_path"])
    predicted_class = list(preds.keys())[0]

    if DEFAULT_TEST_SAMPLE_GT == predicted_class:
        print(
            f"The output image has been classified as {Fore.GREEN}{predicted_class}{Style.RESET_ALL}"
        )
    else:
        print(
            f"The output image has been classified as {Fore.RED}{predicted_class}{Style.RESET_ALL}"
        )

    view_prediction(img, preds, ground_truth=DEFAULT_TEST_SAMPLE_GT)


if __name__ == "__main__":
    # construct the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-img",
        "--image-path",
        type=str,
        default=DEFAULT_TEST_SAMPLE,
        dest="image_path",
        help="Full image path.",
    )

    parser.add_argument(
        "-m",
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_LOC,
        dest="model_path",
        help="Full path to the saved model file.",
    )
    args = vars(parser.parse_args())

    colorama_init()
    main()
