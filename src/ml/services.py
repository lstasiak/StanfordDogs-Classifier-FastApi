from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.settings import DEFAULT_CLASS_NAMES, PRETRAINED_MODELS  # type: ignore

plt.style.use("ggplot")


def get_default_model(
    base_model: str = "ResNet",
    apply_head: bool = True,
    num_classes=len(DEFAULT_CLASS_NAMES),
):
    """
    returns model instance according to passed arguments
    :return:
    """
    from ml.models.classifiers import MultiClassClassificationModel  # type: ignore

    classifier = MultiClassClassificationModel(
        base_model=PRETRAINED_MODELS[base_model]["model"],
        weights=PRETRAINED_MODELS[base_model]["weights"],
        apply_head=apply_head,
        num_classes=num_classes,
    )
    return classifier


# quick visualization
def imshow(inp, title: Union[str, None] = None) -> None:
    """Plot image"""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)


def f1_score(
    y_true: torch.Tensor, y_pred: torch.Tensor, num_classes=None, is_training=False
) -> torch.Tensor:
    """
    Calculates F1 score using tensors
    """
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2

    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)

    if num_classes is None:
        tp = (y_true * y_pred).sum().to(torch.float32)
        # tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

        epsilon = 1e-7

        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + epsilon)
        f1.requires_grad = is_training

    else:
        tp = {}
        fp = {}
        fn = {}

    return f1


def view_prediction(
    img, predictions: dict, ground_truth: str = None, save: str = None
) -> None:
    """
    utility to visualize image and prediction results
    """
    class_names = list(predictions.keys())
    values = list(predictions.values())

    pred_class = class_names[0]
    pred_value = values[0]

    fig, (ax1, ax2) = plt.subplots(figsize=(10, 5), ncols=2)
    ax1.imshow(img)
    if ground_truth is not None:
        title_list = [
            "This is",
            f"{pred_class}",
            f"with: {100 * pred_value:.1f}% confidence",
        ]
        if ground_truth.lower() == pred_class.lower():
            colors = ["black", "green", "black"]
        else:
            colors = ["black", "red", "black"]
        color_title(title_list, colors, ax=ax1)
    else:
        ax1.set_title(
            f"This is {pred_class} with: {100 * pred_value:.1f}%\n confidence"
        )

    ax1.axis("off")

    if len(DEFAULT_CLASS_NAMES) > 10:
        class_names = class_names[:10]
        values = values[:10]

    ax2.barh(class_names, values)
    ax2.set_aspect(0.1)
    ax2.set_yticks(class_names)
    ax2.set_yticklabels(class_names)
    ax2.set_title("Predicted Class")
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
    if save is not None:
        fig.savefig(save, format="JPEG")


def get_file_name(model_name: str, extension: str, **kwargs) -> str:
    """
    helper function to generate file names.
    """
    fn = model_name
    for key, value in kwargs.items():
        fn += f"_{key}_{value}"

    return f"{fn}{extension}"


def get_on_epoch_message(
    epoch: int, num_epochs: int, train_metrics: dict, val_metrics: dict
):
    msg = f"""Epoch: {epoch + 1}/{num_epochs}\n{"=" * 10}\n"""
    lines = [f"""|{" TRAIN":7s}|""", f"""|{"  VAL":7s}|"""]
    i = 0
    for line, metrics in zip(lines, [train_metrics, val_metrics]):
        if metrics.get("loss", None):
            line += f" Loss: {metrics['loss']:.4f}"
        if metrics.get("acc", None):
            line += f" Acc: {metrics['acc']:.4f}"
        if metrics.get("score", None):
            line += f" F1_score: {metrics['score']:.4f}"
        lines[i] = line
        i += 1
    return msg + f"{lines[0]}\n{lines[1]}"


def color_title(
    labels, colors, textprops={"size": "large"}, ax=None, y=1.013, precision=10**-2
):
    """
    Creates a centered title with multiple colors.
    Borrowed from:
    https://github.com/alexanderthclark/Matplotlib-for-Storytellers/blob/main/Python/color_title.py
    """

    if ax == None:
        ax = plt.gca()

    plt.gcf().canvas.draw()
    transform = ax.transAxes  # use axes coords

    # initial params
    xT = 0  # where the text ends in x-axis coords
    shift = 0  # where the text starts

    # for text objects
    text = dict()

    while (np.abs(shift - (1 - xT)) > precision) and (shift <= xT):
        x_pos = shift

        for label, col in zip(labels, colors):

            try:
                text[label].remove()
            except KeyError:
                pass

            text[label] = ax.text(
                x_pos, y, label, transform=transform, ha="left", color=col, **textprops
            )

            x_pos = text[label].get_window_extent().transformed(transform.inverted()).x1

        xT = x_pos  # where all text ends

        shift += precision / 2  # increase for next iteration

        if x_pos > 1:  # guardrail
            break
