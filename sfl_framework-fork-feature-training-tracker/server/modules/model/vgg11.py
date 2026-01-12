from typing import TYPE_CHECKING

import torch
import torchvision

from .base_model import BaseModel
from .custom_model.tiny_vgg11 import TinyVGG11

if TYPE_CHECKING:
    from server_args import Config
    from torch.utils.data import DataLoader


class VGG11(BaseModel):
    def __init__(self, config: "Config", num_classes: int):
        super().__init__(config)
        self.num_classes = num_classes
        if config.model == "tiny_vgg11":
            self.torch_model = TinyVGG11(num_classes)
        else:
            self.torch_model = torchvision.models.vgg11(
                pretrained=False, num_classes=num_classes
            )

        self.disable_inplace(self.torch_model)
        self.config = config
        self.torch_model.to(self.config.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.torch_model(x)

    def predict(self, inputs) -> torch.Tensor:
        self.torch_model.eval()
        with torch.no_grad():
            return self.forward(inputs)

    def save_model(self, save_path: str) -> None:
        torch.save(self.torch_model, save_path + ".pth")

    def load_model(self, load_path: str) -> None:
        self.torch_model = torch.load(load_path + ".pth")

    def get_torch_model(self):
        return self.torch_model

    def set_torch_model(self, torch_model: torch.nn.Module):
        torch_model.to(self.config.device)
        self.torch_model = torch_model

    def evaluate(self, testloader: "DataLoader"):
        self.torch_model.to(self.config.device)
        self.torch_model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data in testloader:
                inputs, labels = data
                inputs, labels = inputs.to(self.config.device), labels.to(
                    self.config.device
                )
                outputs = self.forward(inputs)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total
