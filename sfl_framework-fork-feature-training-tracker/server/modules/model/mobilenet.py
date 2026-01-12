from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torchvision

from .base_model import BaseModel

if TYPE_CHECKING:
    from server_args import Config
    from torch.utils.data import DataLoader


class MobileNet(BaseModel):
    def __init__(self, config: "Config", num_classes: int):
        super().__init__(config)
        self.num_classes = num_classes
        self.torch_model = torchvision.models.mobilenet_v2(
            pretrained=False, num_classes=num_classes
        )

        if config.dataset == "cifar100":
            first_conv = self.torch_model.features[0][0]
            in_channels = first_conv.in_channels  # 보통 3
            out_channels = first_conv.out_channels  # 보통 32
            self.torch_model.features[0][0] = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
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
