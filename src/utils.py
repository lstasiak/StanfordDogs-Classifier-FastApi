import os

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
        available_devices["cuda"] = torch.device("cuda:0")
    if torch.__version__.startswith("2"):
        # check for Apple Silicon Mac acceleration (only in pytorch-nightly)
        if torch.backends.mps.is_available():
            available_devices["mps"] = torch.device("mps")

    return available_devices.get(device, "cpu")
