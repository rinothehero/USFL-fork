from typing import TYPE_CHECKING

import evaluate
import torch
from transformers import AutoModelForSequenceClassification

from .base_model import BaseModel

if TYPE_CHECKING:
    from numpy import ndarray
    from server_args import Config
    from torch.utils.data import DataLoader


class DistilBert(BaseModel):
    def __init__(self, config: "Config", num_classes: int) -> None:
        super().__init__(config)
        self.num_classes = num_classes
        self.torch_model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=num_classes
        )
        self.config = config
        self.torch_model.to(self.config.device)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.torch_model(input_ids=input_ids, attention_mask=attention_mask)

    def predict(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        self.torch_model.eval()
        with torch.no_grad():
            return self.forward(input_ids, attention_mask)

    def save_model(self, save_path: str) -> None:
        torch.save(self.torch_model, save_path + ".pth")

    def load_model(self, load_path: str) -> None:
        self.torch_model = torch.load(load_path + ".pth")

    def get_torch_model(self):
        return self.torch_model

    def set_torch_model(self, torch_model: torch.nn.Module):
        torch_model.to(self.config.device)
        self.torch_model = torch_model

    def _load_metrics(self) -> tuple:

        if self.config.dataset in ["mrpc", "qqp"]:
            metrics = (evaluate.load("f1"), evaluate.load("accuracy"))

        elif self.config.dataset == "cola":
            metrics = (evaluate.load("matthews_correlation"),)

        elif self.config.dataset == "sts-b":
            metrics = (evaluate.load("pearson"), evaluate.load("spearmanr"))
        else:
            metrics = (evaluate.load("accuracy"),)

        return metrics

    def evaluate(self, testloader: "DataLoader") -> list[dict]:
        self.torch_model.eval()
        self.torch_model.to(self.config.device)
        metrics = self._load_metrics()
        predictions: list["ndarray"] = []
        references: list["ndarray"] = []
        results: list[dict] = []

        with torch.no_grad():
            for data in testloader:
                inputs = data["input_ids"].to(self.config.device)
                attention_mask = data["attention_mask"].to(self.config.device)
                labels = data["label"].to(self.config.device)

                outputs = self.forward(input_ids=inputs, attention_mask=attention_mask)
                predicted = (
                    outputs.logits
                    if self.config.dataset == "sts-b"
                    else torch.argmax(outputs.logits, dim=-1)
                )

                predictions.extend(predicted.cpu().numpy())
                references.extend(labels.cpu().numpy())

        for metric in metrics:
            result = metric.compute(predictions=predictions, references=references)
            results.append(result)

        return results
