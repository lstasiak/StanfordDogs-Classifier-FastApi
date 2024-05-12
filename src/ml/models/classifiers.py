import torch.nn as nn
from torchsummary import summary

from settings import PARAMETERS  # type: ignore


class MultiClassClassificationModel(nn.Module):
    """
    Class representation of the object for creating model
     from pre-trained base and Sequential head.
    """

    def __init__(
        self,
        base_model,
        weights,
        apply_head: bool,
        num_classes: int,
    ):
        """
        Creates multiclass classifier model

        :param base_model: based on torch nn.Module model
        :param weights: pretrained weights to be loaded (compatible with base model)
        :param apply_head: if True, model applies trainable "head" model and
         base model weights are set to be not trainable.
        :param num_classes: number of classes to classify
        """
        super().__init__()
        self.model = base_model(weights=weights)
        self.base_name = self.model.__class__.__name__  # only for informative purposes
        self.apply_head = apply_head

        # depending on the base model, the access
        # to output features can be through 'fc' (if single layer)
        # or 'classifier' (in case of sequential object)
        if hasattr(self.model, "fc"):
            self._model_output_attr_name = "fc"
            num_features = self.model.fc.in_features
        elif hasattr(self.model, "classifier"):
            self._model_output_attr_name = "classifier"
            num_features = self.model.classifier[-1].in_features
        else:
            raise AttributeError(
                "Could not specify output layers features in given base model"
            )

        # set output layers in the model
        if self.apply_head:
            setattr(
                self.model,
                self._model_output_attr_name,
                self.get_head(num_features=num_features, num_classes=num_classes),
            )
            self.freeze()
        else:
            setattr(
                self.model,
                self._model_output_attr_name,
                nn.Linear(in_features=num_features, out_features=num_classes),
            )

    @staticmethod
    def get_head(num_features: int, num_classes: int) -> nn.Sequential:
        """
        Head model applied for fine-tuning
        num_features: int = number of output nodes in base model last layer
        num_classes: int
        """
        linear_layers = nn.Sequential(
            nn.BatchNorm1d(num_features=num_features),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )
        return linear_layers

    def forward(self, x):
        return self.model(x)

    def freeze(self) -> None:
        # To freeze the base model layers
        self.model.requires_grad_(False)

        # set head model/layer trainable
        getattr(self.model, self._model_output_attr_name).requires_grad_(True)

    def unfreeze(self) -> None:
        self.model.requires_grad_(True)

    def summary(self):
        return summary(self.model, PARAMETERS["input_shape"])
