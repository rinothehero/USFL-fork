from typing import TYPE_CHECKING

import torch
from torchvision.models import alexnet

from .base_model import BaseModel
from .custom_model.alexnet_1ch import AlexNet1Ch
from .custom_model.alexnet_scala import AlexNetSCALA
from .custom_model.alexnet_cifar import AlexNetCifar
from .custom_model.alexnet_mnist import AlexNetMnist

if TYPE_CHECKING:
    from server_args import Config
    from torch.utils.data import DataLoader


class AlexNet(BaseModel):
    def __init__(self, config: "Config", num_classes: int):
        super().__init__(config)
        self.num_classes = num_classes

        if config.model == "alexnet_scala":
            self.torch_model = AlexNetSCALA(num_classes)
        elif config.model == "alexnet_legacy":
            # Legacy AlexNet1Ch for backward compatibility
            self.torch_model = AlexNet1Ch(num_classes)
        else:
            if config.dataset.lower() in ["mnist", "fmnist"]:
                self.torch_model = AlexNetMnist(num_classes)
            else:
                self.torch_model = AlexNetCifar(num_classes)

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
