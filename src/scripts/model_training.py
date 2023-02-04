import argparse
import os

import torch

from src.ml.data_managers import DatasetCollector
from src.ml.services import get_default_model, get_file_name
from src.ml.trainers import Trainer
from src.settings import (
    DATA_DIR,
    DEFAULT_SAVE_MODEL_DIR,
    DEFAULT_TRAINING_HISTORY_DIR,
    NUM_CLASSES,
    PARAMETERS,
)
from src.utils import get_user_device


def main():
    # torch.manual_seed(1234)

    # load dataset
    collector = DatasetCollector(
        img_size=PARAMETERS["img_size"],
        batch_size=args["batch"],
        data_root=DATA_DIR,
        organize=True,
        split_ratio=[0.8, 0.2],
    )
    data_loaders = collector.get_dataloaders(num_workers=4)

    print(collector.get_dataset_summary())

    # build model
    model = get_default_model(
        base_model=args["model"], apply_head=args["apply_head"], num_classes=NUM_CLASSES
    )
    # print model summary:
    print(f"Loaded model: {model.base_name}")
    model.summary()
    device = get_user_device(args["device"])
    trainer = Trainer(
        criterion=PARAMETERS["criterion"],
        epochs=args["epochs"],
        optim_fcn=PARAMETERS["optim_fcn"],
        device=device,
        lr=args["learning_rate"],
    )
    print("=" * 37)
    print(f"""Start model training on device: {device}""")
    print("=" * 37)
    history_filename = get_file_name(
        model.base_name,
        ".csv",
        batch=args["batch"],
        epochs=args["epochs"],
        apply_head=model.apply_head,
        history="",
    )
    model = trainer.fit(
        model=model,
        data_loaders=data_loaders,
        save_history=os.path.join(DEFAULT_TRAINING_HISTORY_DIR, history_filename),
    )

    torch.cuda.empty_cache()

    if args["save"]:
        model_filenames = [
            get_file_name(
                model.base_name,
                ".pt",
                batch=args["batch"],
                epochs=args["epochs"],
                apply_head=model.apply_head,
                model=mtype,
            )
            for mtype in ["state_dict", "complete"]
        ]
        # save model as state_dict
        torch.save(
            model.state_dict(),
            os.path.join(DEFAULT_SAVE_MODEL_DIR, model_filenames[0]),
        )
        # save whole model scripted
        model_scripted = torch.jit.script(model)
        model_scripted.save(os.path.join(DEFAULT_SAVE_MODEL_DIR, model_filenames[1]))
        print("Model saved successfully.")


if __name__ == "__main__":
    # construct the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=PARAMETERS["epochs"],
        help="Number of epochs to train our network for",
    )
    parser.add_argument(
        "-bs", "--batch", type=int, default=PARAMETERS["batch_size"], help="Batch size"
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        dest="learning_rate",
        default=PARAMETERS["lr"],
        help="Learning rate for training the model",
    )
    parser.add_argument(
        "-s", "--save", action="store_true", help="Whether to save the model"
    )
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default=DEFAULT_SAVE_MODEL_DIR,
        help="Path where to save the model",
    )

    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=PARAMETERS["device"],
        help="Path where to save the model",
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=["ResNet", "EfficientNet"],
        default="ResNet",
        help="Base - pretrained model type",
    )

    parser.add_argument(
        "-ah",
        "--apply-head",
        dest="apply_head",
        action="store_true",
        help="Whether to apply head and freeze base model",
    )

    args = vars(parser.parse_args())

    main()
