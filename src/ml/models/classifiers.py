import torch.nn as nn
from torchsummary import ModelStatistics, summary

from settings import PARAMETERS  # type: ignore


class MultiClassClassificationModel(nn.Module):
    """
    Class representation of the object for creating model
     from pre-trained base and Sequential head.
    """

    @staticmethod
    def get_head(num_ftrs: int, num_classes: int) -> nn.Sequential:
        """
        Head model applied in case when base model weights are frozen
        num_ftrs: int = number of output nodes in base model last layer
        """
        linear_layers = nn.Sequential(
            nn.BatchNorm1d(num_features=num_ftrs),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )
        return linear_layers

    def __init__(
        self,
        base_model,
        weights,
        apply_head: bool,
        num_classes: int,
    ):
        super().__init__()
        self.model = base_model(weights=weights)
        self.base_name = self.model.__class__.__name__  # only for informative purposes
        self.apply_head = apply_head

        if self.base_name == "EfficientNet":
            num_features = self.model.classifier[-1].in_features
            if self.apply_head:
                self.model.classifier = self.get_head(
                    num_ftrs=num_features, num_classes=num_classes
                )
            else:
                self.model.classifier = nn.Linear(
                    in_features=num_features, out_features=num_classes
                )
        else:
            num_features = self.model.fc.in_features
            if self.apply_head:
                self.model.fc = self.get_head(
                    num_ftrs=num_features, num_classes=num_classes
                )
            else:
                self.model.fc = nn.Linear(
                    in_features=num_features, out_features=num_classes
                )

        if self.apply_head:
            self.freeze()

    def forward(self, x):
        return self.model(x)

    def freeze(self) -> None:
        # To freeze the residual layers
        for param in self.model.parameters():
            param.require_grad = False

        if self.base_name == "EfficientNet":
            for param in self.model.classifier.parameters():
                param.require_grad = True
        else:
            for param in self.model.fc.parameters():
                param.require_grad = True

    def unfreeze(self) -> None:
        # Unfreeze all layers
        for param in self.model.parameters():
            param.require_grad = True

    def summary(self) -> ModelStatistics:
        return summary(self.model, PARAMETERS["input_shape"])
