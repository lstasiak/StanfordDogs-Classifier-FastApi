from typing import Dict, Iterable, Tuple

import torch


class F1Score:
    def __init__(self, device: torch.device, average: str = "weighted"):
        """
        Class for f1 calculation in Pytorch.

        :param: average - averaging method
        """
        self.device = device
        self.average = average
        if average not in [None, "micro", "macro", "weighted"]:
            raise ValueError("Wrong value of average parameter")

    @staticmethod
    def calc_f1_micro(predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calculate f1 micro.

        Args:
            predictions: tensor with predictions
            labels: tensor with original labels

        Returns:
            f1 score
        """
        true_positive = torch.eq(labels, predictions).sum().float()
        f1_score = torch.div(true_positive, len(labels))
        return f1_score

    @staticmethod
    def calc_f1_count_for_label(
        predictions: torch.Tensor, labels: torch.Tensor, label_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate f1 and true count for the label

        Args:
            predictions: tensor with predictions
            labels: tensor with original labels
            label_id: id of current label

        Returns:
            f1 score and true count for label
        """
        # label count
        true_count = torch.eq(labels, label_id).sum()

        # true positives: labels equal to prediction and to label_id
        true_positive = (
            torch.logical_and(torch.eq(labels, predictions), torch.eq(labels, label_id))
            .sum()
            .float()
        )
        # precision for label
        precision = torch.div(
            true_positive, torch.eq(predictions, label_id).sum().float()
        )
        # replace nan values with 0
        precision = torch.where(
            torch.isnan(precision),
            torch.zeros_like(precision).type_as(true_positive),
            precision,
        )

        # recall for label
        recall = torch.div(true_positive, true_count)
        # f1
        f1 = 2 * precision * recall / (precision + recall)
        # replace nan values with 0
        f1 = torch.where(
            torch.isnan(f1), torch.zeros_like(f1).type_as(true_positive), f1
        )
        return f1, true_count

    def __call__(self, predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calculate f1 score based on averaging method defined in init.


        :param predictions: tensor with predictions
        :param labels: tensor with original labels

        Returns:
            f1 score
        """

        # simpler calculation for micro
        if self.average == "micro":
            return self.calc_f1_micro(predictions, labels)

        f1_score = torch.zeros(1, requires_grad=False, device=self.device)
        for label_id in range(1, len(labels.unique()) + 1):
            f1, true_count = self.calc_f1_count_for_label(predictions, labels, label_id)

            if self.average == "weighted":
                f1_score += f1 * true_count
            elif self.average == "macro":
                f1_score += f1

        if self.average == "weighted":
            f1_score = torch.div(f1_score, len(labels))
        elif self.average == "macro":
            f1_score = torch.div(f1_score, len(labels.unique()))

        return f1_score


class MetricCollector:
    def __init__(self, metrics: Iterable[str], device: torch.device) -> None:
        """
        Class for calculating model running metrics (e.g. accuracy, f1_score) + loss
        and collecting per epoch values.

        :param metrics: iterable with metric codes ("acc" for accuracy, "score" for f1-score)
        """
        self.metrics = metrics
        self.device = device
        self.current_total = {name: 0.0 for name in self.metrics}
        self.current_total["loss"] = 0.0
        self.iterations = 0.0

        self.metric_func = {}
        if "acc" in self.metrics:
            self.metric_func["acc"] = (
                lambda predictions, labels: (predictions == labels)
                .float()
                .mean()
                .item()
            )
        if "score" in self.metrics:
            f1_metric = F1Score(device=self.device, average="weighted")
            self.metric_func["score"] = lambda predictions, labels: f1_metric(
                labels, predictions
            ).item()

    def send(self, logits: torch.Tensor, labels: torch.Tensor, loss) -> None:
        """
        collect "running" (per batch) metric
        :param logits: output logits from NN model
        :param labels: tensor label from batch
        :param loss: loss value
        """
        predictions = logits.argmax(dim=1)
        for metric in self.metrics:
            self.current_total[metric] += self.metric_func[metric](predictions, labels)
        self.current_total["loss"] += loss
        self.iterations += 1

    @property
    def value(self) -> Dict[str, float]:
        """
        return metrics per epoch
        """
        if self.iterations == 0.0:
            return self.current_total
        else:
            return {
                metric: value / self.iterations
                for metric, value in self.current_total.items()
            }

    def reset(self) -> None:
        self.current_total = {name: 0.0 for name in self.metrics}
        self.current_total["loss"] = 0.0
        self.iterations = 0.0
