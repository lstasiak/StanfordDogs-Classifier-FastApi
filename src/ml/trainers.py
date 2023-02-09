import copy
import time
from typing import Dict, Union

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from src.ml.metrics import MetricCollector
from src.ml.services import get_on_epoch_message
from src.settings import DATA_SPLIT, PARAMETERS


class Trainer:
    def __init__(
        self,
        criterion,
        epochs: int,
        optim_fcn,
        device: torch.device,
        lr: float = PARAMETERS["lr"],
        metrics=(
            "score",
            "acc",
        ),
        monitor="acc",
    ):
        """

        Class representation of neural network model trainer.
        The main method `fit` integrates train and validation steps

        :param criterion: Loss function handler
        :param epochs: number of total epochs to run
        :param optim_fcn: Optimizer function handler
        :param device: device to run the training on
        :param lr: learning parameter passed to optimizer
        :param metrics: metric functions to calculate during model training/testing
        :param monitor: metric function (specified in metrics) according to which the best model
        will be saved
        """
        self.criterion = criterion
        self.num_epochs = epochs
        self.optim_function = optim_fcn
        self.device = device
        self.lr = lr
        self.metrics = list(map(lambda x: x.lower(), metrics))
        self.monitor = monitor.lower() if monitor.lower() in metrics else metrics[0]

        self.history = pd.DataFrame({f"{phase}_loss": [] for phase in DATA_SPLIT})
        for phase in DATA_SPLIT:
            for metric in self.metrics:
                self.history[f"{phase}_{metric}"] = []

    def train_step(
        self,
        model: nn.Module,
        train_dataloader,
        optimizer: Optimizer,
        metric_calculator,
    ) -> Dict[str, float]:
        """
        perform one training step (one pass through dataset)

        :param model: NN model (nn.Module)
        :param train_dataloader: dataloader for training set
        :param optimizer: optimizer object
        :param metric_calculator: MetricCalculator object

        :return: mapping from metric names to calculated values per epoch
        """
        model.train()
        metric_calculator.reset()

        for inputs, labels in train_dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = self.criterion(outputs, labels)

                loss.backward()
                optimizer.step()

            metric_calculator.send(logits=outputs, labels=labels, loss=loss.item())

        return metric_calculator.value

    def validate(
        self, model: nn.Module, test_dataloader, metric_calculator
    ) -> Dict[str, float]:
        """
        Perform one validation step (one pass through dataset)

        :param model: NN model (nn.Module)
        :param test_dataloader: dataloader for test/validation set
        :param metric_calculator: MetricCalculator object

        :return: mapping from metric names to calculated values per epoch
        """
        model.eval()
        metric_calculator.reset()

        for inputs, labels in test_dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                outputs = model(inputs)
                loss = self.criterion(outputs, labels)
            metric_calculator.send(logits=outputs, labels=labels, loss=loss.item())

        return metric_calculator.value

    def fit(
        self,
        model: nn.Module,
        data_loaders: Dict[str, DataLoader],
        save_history: Union[str, None] = None,
        verbose: bool = True,
    ) -> nn.Module:
        """
        Model training method.
        :param model: torch Model class
        :param data_loaders: Dictionary containing DataLoaders for separated datasets
        :param save_history: filepath to save training stats
        :param verbose: True/False -- can hide training messaging
        :return: trained model
        """
        model = model.to(self.device)
        optimizer = self.optim_function(model.parameters(), lr=self.lr)

        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        metric_calc = MetricCollector(metrics=self.metrics, device=self.device)

        best_model_wts = copy.deepcopy(model.state_dict())
        best_value = 0.0

        since = time.time()

        for epoch in range(self.num_epochs):
            train_metrics = self.train_step(
                model,
                data_loaders["train"],
                optimizer,
                metric_calc,
            )

            scheduler.step()

            val_metrics = self.validate(
                model,
                data_loaders["val"],
                metric_calc,
            )

            if verbose:
                print(
                    get_on_epoch_message(
                        epoch, self.num_epochs, train_metrics, val_metrics
                    )
                )
            # update history df
            train_records = {
                f"train_{metric}": value for metric, value in train_metrics.items()
            }
            val_records = {
                f"val_{metric}": value for metric, value in val_metrics.items()
            }

            self.history.loc[epoch, train_records.keys()] = train_records
            self.history.loc[epoch, val_records.keys()] = val_records

            # deep copy the model
            if val_metrics[self.monitor] > best_value:
                best_value = val_metrics[self.monitor]
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print(
            f"Training complete in {time_elapsed // 60:.0f} min {time_elapsed % 60:.0f}s"
        )
        print(f"Best validation {self.monitor}: {best_value:.4f}")

        if save_history is not None:
            self.history.to_csv(save_history)

        model.load_state_dict(best_model_wts)

        return model
