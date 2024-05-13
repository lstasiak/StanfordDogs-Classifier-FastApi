import os

from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torchvision.models import (
    EfficientNet_V2_S_Weights,
    ResNet18_Weights,
    efficientnet_v2_s,
    resnet18,
)

from project_utils import get_labels, get_user_device

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT == "/":
    PROJECT_ROOT = "./"

# RESOURCE_DIR
RESOURCES_DIR = os.path.join(PROJECT_ROOT, "resources")

# DATASET LINKS
DATA_DIR = os.path.join(PROJECT_ROOT, "dataset")
ANNOTATIONS_DIR = os.path.join(DATA_DIR, "Annotation")
LABELS_FILE = os.path.join(RESOURCES_DIR, "labels.txt")
TRAIN_DIR = os.path.join(PROJECT_ROOT, "../data/train")
VAL_DIR = os.path.join(PROJECT_ROOT, "../data/val")

# DATASET STRUCTURE
# created under condition, only for github-workflow purposes (no dataset repo on GitHub)
if os.path.exists(LABELS_FILE):
    with open(LABELS_FILE, "r") as f:
        DEFAULT_CLASS_NAMES = f.read().splitlines()
elif os.path.exists(ANNOTATIONS_DIR):
    DEFAULT_CLASS_NAMES = list(
        dict(sorted(get_labels(ANNOTATIONS_DIR).items())).values()
    )
    DEFAULT_TEST_SAMPLE_GT = get_labels(ANNOTATIONS_DIR)["n02099601-golden_retriever"]

    with open(f"{RESOURCES_DIR}/labels.txt", "w") as f:
        for label in DEFAULT_CLASS_NAMES:
            f.write(label)
            f.write("\n")
else:
    DEFAULT_CLASS_NAMES = [""]
    DEFAULT_TEST_SAMPLE_GT = "Golden_Retriever"

NUM_CLASSES = len(DEFAULT_CLASS_NAMES)

DATA_SPLIT = ["train", "val"]
DATA_DIR_LOC = [TRAIN_DIR, VAL_DIR]
DATA_DIR_STRUCT = {phase: path for phase, path in zip(DATA_SPLIT, DATA_DIR_LOC)}

# Default on save specs
DEFAULT_SAVE_MODEL_DIR = os.path.join(RESOURCES_DIR, "saved_models")
DEFAULT_TRAINING_HISTORY_DIR = os.path.join(RESOURCES_DIR, "saved_train_history")


# Default files to load
DEFAULT_TEST_SAMPLE = os.path.join(
    DATA_DIR, "n02099601-golden_retriever/n02099601_67.jpg"
)

DEFAULT_MODEL_LOC = os.path.join(
    DEFAULT_SAVE_MODEL_DIR,
    "EfficientNet_batch_64_epochs_10_apply_head_True_model_complete.pt",
)

DEFAULT_TRAINING_HISTORY_SAMPLE = os.path.join(
    DEFAULT_TRAINING_HISTORY_DIR,
    "EfficientNet_batch_64_epochs_10_apply_head_True_history_.csv",
)

# Project parameters
PARAMETERS = {
    "input_shape": (3, 224, 224),
    "img_size": (224, 224),
    "channels": 3,
    "batch_size": 64,
    "epochs": 10,
    "device": get_user_device("mps"),
    "criterion": CrossEntropyLoss(),
    "F_score_threshold": 0.4,
    "optim_fcn": Adam,
    "lr": 10e-4,
}

PRETRAINED_MODELS = {
    "EfficientNet": {
        "model": efficientnet_v2_s,
        "weights": EfficientNet_V2_S_Weights.DEFAULT,
    },
    "ResNet": {"model": resnet18, "weights": ResNet18_Weights.DEFAULT},
}
