import copy
from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Union, Literal, Dict, cast, List
import torchvision.models as tv_models


def disable_inplace(module: nn.Module) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, nn.ReLU(inplace=False))
        elif isinstance(child, nn.ReLU6):
            setattr(module, name, nn.ReLU6(inplace=False))
        else:
            disable_inplace(child)


# =============================================================================
# Basic Building Blocks
# =============================================================================


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


# =============================================================================
# Legacy Simple Models (for backward compatibility)
# =============================================================================


class ClientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ServerNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class SplitModel(nn.Module):
    def __init__(self, client: nn.Module, server: nn.Module):
        super().__init__()
        self.client = client
        self.server = server

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.client(x)
        return self.server(f)


# =============================================================================
# AlexNet Models
# =============================================================================


class AlexNetCifar(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(AlexNetCifar, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 3 * 3, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, num_classes),
        )

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        out = self.conv(x)
        features = out.view(-1, 256 * 3 * 3)
        out = self.fc(features)
        if return_features:
            return out, features
        return out


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 3 * 3, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, num_classes),
        )

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        out = self.conv(x)
        features = out.view(-1, 256 * 3 * 3)
        out = self.fc(features)
        if return_features:
            return out, features
        return out


class AlexNetDownCifar(nn.Module):
    def __init__(self):
        super(AlexNetDownCifar, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class AlexNetUpCifar(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(AlexNetUpCifar, self).__init__()
        self.model2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 3 * 3, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model2(x)
        x = x.view(-1, 256 * 3 * 3)
        x = self.classifier(x)
        return x


class AlexNetDownCifarLight(nn.Module):
    def __init__(self):
        super(AlexNetDownCifarLight, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class AlexNetUpCifarHeavy(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(AlexNetUpCifarHeavy, self).__init__()
        self.model2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 3 * 3, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model2(x)
        x = x.view(-1, 256 * 3 * 3)
        x = self.classifier(x)
        return x


class AlexNetDown(nn.Module):
    def __init__(self):
        super(AlexNetDown, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class AlexNetUp(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(AlexNetUp, self).__init__()
        self.model2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 3 * 3, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model2(x)
        x = x.view(-1, 256 * 3 * 3)
        x = self.classifier(x)
        return x


class AlexNetSplitClient(nn.Module):
    def __init__(self, conv_layers: List[nn.Module]):
        super().__init__()
        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AlexNetSplitServer(nn.Module):
    def __init__(
        self,
        conv_layers: List[nn.Module],
        fc_layers: List[nn.Module],
        conv_offset: Optional[int] = None,
    ):
        super().__init__()
        if conv_layers:
            if conv_offset is None:
                self.conv = nn.Sequential(*conv_layers)
            else:
                self.conv = nn.Sequential(
                    OrderedDict(
                        (str(i + conv_offset), layer)
                        for i, layer in enumerate(conv_layers)
                    )
                )
        else:
            self.conv = nn.Identity()
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def _resolve_alexnet_split_index(
    split_layer: Optional[str], default_index: int, total_layers: int
) -> int:
    if not split_layer:
        return default_index

    normalized = split_layer.strip().lower()
    if normalized in {"default", "conv2"}:
        return default_index
    if normalized in {"light", "conv1"}:
        return min(2, total_layers - 1)
    if normalized.startswith("conv."):
        idx_str = normalized.split(".", 1)[1]
        if idx_str.isdigit():
            idx = int(idx_str)
            if 0 <= idx < total_layers:
                return idx
            raise ValueError(f"AlexNet split_layer index out of range: {split_layer}")
    if normalized.startswith("conv") and normalized[4:].isdigit():
        idx = int(normalized[4:])
        if 0 <= idx < total_layers:
            return idx
        raise ValueError(f"AlexNet split_layer index out of range: {split_layer}")
    if normalized.startswith("layer"):
        return default_index

    raise ValueError(f"Unsupported AlexNet split_layer: {split_layer}")


def _build_alexnet_split_models(
    full_model: nn.Module, split_index: int
) -> Tuple[nn.Module, nn.Module]:
    conv_layers = list(full_model.conv.children())
    if not (0 <= split_index < len(conv_layers)):
        raise ValueError(
            f"AlexNet split index must be within [0, {len(conv_layers) - 1}]"
        )

    client_layers = [copy.deepcopy(layer) for layer in conv_layers[: split_index + 1]]
    server_layers = [copy.deepcopy(layer) for layer in conv_layers[split_index + 1 :]]
    fc_layers = [copy.deepcopy(layer) for layer in full_model.fc.children()]

    return AlexNetSplitClient(client_layers), AlexNetSplitServer(
        server_layers, fc_layers, conv_offset=split_index + 1
    )


# =============================================================================
# ResNet-18 Models
# =============================================================================


class ResNet18Cifar(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(ResNet18Cifar, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(
        self, out_channels: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ResNet18ImageStyle(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(ResNet18ImageStyle, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        disable_inplace(self)

    def _make_layer(
        self, out_channels: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ResNet18DownCifar(nn.Module):
    def __init__(self):
        super(ResNet18DownCifar, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)

    def _make_layer(
        self, out_channels: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class ResNet18UpCifar(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(ResNet18UpCifar, self).__init__()
        self.in_channels = 128
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(
        self, out_channels: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ResNet18DownCifarLight(nn.Module):
    def __init__(self):
        super(ResNet18DownCifarLight, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64, 2, stride=1)

    def _make_layer(
        self, out_channels: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        return x


class ResNet18UpCifarHeavy(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(ResNet18UpCifarHeavy, self).__init__()
        self.in_channels = 64
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(
        self, out_channels: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# =============================================================================
# Flexible Split ResNet-18
# =============================================================================


def parse_split_layer(split_layer: str) -> dict:
    parts = split_layer.split(".")
    result: Dict[str, Union[int, str, None]] = {
        "layer": None,
        "block": None,
        "sublayer": None,
    }
    if len(parts) >= 1:
        result["layer"] = int(parts[0].replace("layer", ""))
    if len(parts) >= 2:
        result["block"] = int(parts[1])
    if len(parts) >= 3:
        result["sublayer"] = parts[2]
    return result


class ResNet18FlexibleClient(nn.Module):
    def __init__(self, split_layer: str = "layer2"):
        super(ResNet18FlexibleClient, self).__init__()
        self.split_config = parse_split_layer(split_layer)
        self.split_layer = split_layer
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64, 2, stride=1)
        if self.split_config["layer"] >= 2:
            self.layer2 = self._make_layer(128, 2, stride=2)
        if self.split_config["layer"] >= 3:
            self.layer3 = self._make_layer(256, 2, stride=2)
        if self.split_config["layer"] >= 4:
            self.layer4 = self._make_layer(512, 2, stride=2)

    def _make_layer(
        self, out_channels: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _forward_partial_block(
        self, block: BasicBlock, x: torch.Tensor, stop_at: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        identity = x
        if block.downsample is not None:
            identity = block.downsample(x)
        out = block.conv1(x)
        if stop_at == "conv1":
            return out, identity
        out = block.bn1(out)
        if stop_at == "bn1":
            return out, identity
        out = block.relu(out)
        if stop_at == "relu1":
            return out, identity
        out = block.conv2(out)
        if stop_at == "conv2":
            return out, identity
        out = block.bn2(out)
        if stop_at == "bn2":
            return out, identity
        out += identity
        out = block.relu(out)
        return out, identity

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        split_layer = self.split_config["layer"]
        split_block = self.split_config["block"]
        split_sublayer = self.split_config["sublayer"]

        if split_layer == 1:
            if split_block is not None and split_sublayer is not None:
                for i, block in enumerate(self.layer1):
                    block = cast(BasicBlock, block)
                    if i < split_block:
                        x = block(x)
                    elif i == split_block:
                        return self._forward_partial_block(block, x, split_sublayer)
            elif split_block is not None:
                for i, block in enumerate(self.layer1):
                    if i <= split_block:
                        x = block(x)
                return x
            else:
                x = self.layer1(x)
            return x
        else:
            x = self.layer1(x)

        if split_layer == 2:
            if split_block is not None and split_sublayer is not None:
                for i, block in enumerate(self.layer2):
                    block = cast(BasicBlock, block)
                    if i < split_block:
                        x = block(x)
                    elif i == split_block:
                        return self._forward_partial_block(block, x, split_sublayer)
            elif split_block is not None:
                for i, block in enumerate(self.layer2):
                    if i <= split_block:
                        x = block(x)
                return x
            else:
                x = self.layer2(x)
            return x
        elif hasattr(self, "layer2"):
            x = self.layer2(x)

        if split_layer == 3:
            if split_block is not None and split_sublayer is not None:
                for i, block in enumerate(self.layer3):
                    block = cast(BasicBlock, block)
                    if i < split_block:
                        x = block(x)
                    elif i == split_block:
                        return self._forward_partial_block(block, x, split_sublayer)
            elif split_block is not None:
                for i, block in enumerate(self.layer3):
                    if i <= split_block:
                        x = block(x)
                return x
            else:
                x = self.layer3(x)
            return x
        elif hasattr(self, "layer3"):
            x = self.layer3(x)

        if split_layer == 4:
            if hasattr(self, "layer4"):
                x = self.layer4(x)
            return x

        return x


class ResNet18FlexibleServer(nn.Module):
    def __init__(self, split_layer: str = "layer2", num_classes: int = 10):
        super(ResNet18FlexibleServer, self).__init__()
        self.split_config = parse_split_layer(split_layer)
        self.split_layer = split_layer

        channel_map = {1: 64, 2: 128, 3: 256, 4: 512}
        split_layer_num = self.split_config["layer"]
        self.in_channels = channel_map.get(split_layer_num, 64)

        if split_layer_num <= 1:
            self.layer2 = self._make_layer(128, 2, stride=2)
        if split_layer_num <= 2:
            self.layer3 = self._make_layer(256, 2, stride=2)
        if split_layer_num <= 3:
            self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self._build_partial_block_completer()

    def _make_layer(
        self, out_channels: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _build_partial_block_completer(self):
        split_sublayer = self.split_config["sublayer"]
        if split_sublayer is None:
            self.partial_block = None
            return

        split_layer_num = self.split_config["layer"]
        channel_map = {1: 64, 2: 128, 3: 256, 4: 512}
        out_channels = channel_map[split_layer_num]

        self.partial_block = nn.ModuleDict()
        if split_sublayer in ["conv1", "bn1", "relu1"]:
            if split_sublayer == "conv1":
                self.partial_block["bn1"] = nn.BatchNorm2d(out_channels)
            self.partial_block["relu"] = nn.ReLU(inplace=True)
            self.partial_block["conv2"] = nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            self.partial_block["bn2"] = nn.BatchNorm2d(out_channels)
            self.partial_block["final_relu"] = nn.ReLU(inplace=True)
        elif split_sublayer == "conv2":
            self.partial_block["bn2"] = nn.BatchNorm2d(out_channels)
            self.partial_block["final_relu"] = nn.ReLU(inplace=True)
        elif split_sublayer == "bn2":
            self.partial_block["final_relu"] = nn.ReLU(inplace=True)

    def _complete_partial_block(
        self, activation: torch.Tensor, identity: torch.Tensor
    ) -> torch.Tensor:
        split_sublayer = self.split_config["sublayer"]
        x = activation

        if self.partial_block is not None:
            if "bn1" in self.partial_block:
                x = self.partial_block["bn1"](x)
            if "relu" in self.partial_block and split_sublayer in ["conv1", "bn1"]:
                x = self.partial_block["relu"](x)
            if "conv2" in self.partial_block:
                x = self.partial_block["conv2"](x)
            if "bn2" in self.partial_block and split_sublayer != "bn2":
                x = self.partial_block["bn2"](x)
            x = x + identity
            x = self.partial_block["final_relu"](x)
        return x

    def forward(
        self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        split_layer_num = self.split_config["layer"]
        split_block = self.split_config["block"]

        x_tensor: torch.Tensor
        if isinstance(x, tuple):
            activation, identity = x
            x_tensor = self._complete_partial_block(activation, identity)
            if split_block is not None and split_block < 1:
                layer_attr = f"layer{split_layer_num}"
                if hasattr(self, layer_attr):
                    layer = getattr(self, layer_attr)
                    for i, block in enumerate(layer):
                        if i > split_block:
                            x_tensor = block(x_tensor)
        else:
            x_tensor = x

        if split_layer_num <= 1 and hasattr(self, "layer2"):
            x_tensor = self.layer2(x_tensor)
        if split_layer_num <= 2 and hasattr(self, "layer3"):
            x_tensor = self.layer3(x_tensor)
        if split_layer_num <= 3 and hasattr(self, "layer4"):
            x_tensor = self.layer4(x_tensor)

        x_tensor = self.avgpool(x_tensor)
        x_tensor = torch.flatten(x_tensor, 1)
        return self.fc(x_tensor)


class ResNet18ImageStyleFlexibleClient(nn.Module):
    def __init__(self, split_layer: str = "layer2"):
        super(ResNet18ImageStyleFlexibleClient, self).__init__()
        self.split_config = parse_split_layer(split_layer)
        self.split_layer = split_layer
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 2, stride=1)
        if self.split_config["layer"] >= 2:
            self.layer2 = self._make_layer(128, 2, stride=2)
        if self.split_config["layer"] >= 3:
            self.layer3 = self._make_layer(256, 2, stride=2)
        if self.split_config["layer"] >= 4:
            self.layer4 = self._make_layer(512, 2, stride=2)
        disable_inplace(self)

    def _make_layer(
        self, out_channels: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _forward_partial_block(
        self, block: BasicBlock, x: torch.Tensor, stop_at: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        identity = x
        if block.downsample is not None:
            identity = block.downsample(x)
        out = block.conv1(x)
        if stop_at == "conv1":
            return out, identity
        out = block.bn1(out)
        if stop_at == "bn1":
            return out, identity
        out = block.relu(out)
        if stop_at == "relu1":
            return out, identity
        out = block.conv2(out)
        if stop_at == "conv2":
            return out, identity
        out = block.bn2(out)
        if stop_at == "bn2":
            return out, identity
        out += identity
        out = block.relu(out)
        return out, identity

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        split_layer = self.split_config["layer"]
        split_block = self.split_config["block"]
        split_sublayer = self.split_config["sublayer"]

        if split_layer == 1:
            if split_block is not None and split_sublayer is not None:
                for i, block in enumerate(self.layer1):
                    block = cast(BasicBlock, block)
                    if i < split_block:
                        x = block(x)
                    elif i == split_block:
                        return self._forward_partial_block(block, x, split_sublayer)
            elif split_block is not None:
                for i, block in enumerate(self.layer1):
                    if i <= split_block:
                        x = block(x)
                return x
            else:
                x = self.layer1(x)
            return x
        else:
            x = self.layer1(x)

        if split_layer == 2:
            if split_block is not None and split_sublayer is not None:
                for i, block in enumerate(self.layer2):
                    block = cast(BasicBlock, block)
                    if i < split_block:
                        x = block(x)
                    elif i == split_block:
                        return self._forward_partial_block(block, x, split_sublayer)
            elif split_block is not None:
                for i, block in enumerate(self.layer2):
                    if i <= split_block:
                        x = block(x)
                return x
            else:
                x = self.layer2(x)
            return x
        elif hasattr(self, "layer2"):
            x = self.layer2(x)

        if split_layer == 3:
            if split_block is not None and split_sublayer is not None:
                for i, block in enumerate(self.layer3):
                    block = cast(BasicBlock, block)
                    if i < split_block:
                        x = block(x)
                    elif i == split_block:
                        return self._forward_partial_block(block, x, split_sublayer)
            elif split_block is not None:
                for i, block in enumerate(self.layer3):
                    if i <= split_block:
                        x = block(x)
                return x
            else:
                x = self.layer3(x)
            return x
        elif hasattr(self, "layer3"):
            x = self.layer3(x)

        if split_layer == 4:
            if hasattr(self, "layer4"):
                x = self.layer4(x)
            return x

        return x


class ResNet18ImageStyleFlexibleServer(nn.Module):
    def __init__(self, split_layer: str = "layer2", num_classes: int = 10):
        super(ResNet18ImageStyleFlexibleServer, self).__init__()
        self.split_config = parse_split_layer(split_layer)
        self.split_layer = split_layer

        channel_map = {1: 64, 2: 128, 3: 256, 4: 512}
        split_layer_num = self.split_config["layer"]
        self.in_channels = channel_map.get(split_layer_num, 64)

        if split_layer_num <= 1:
            self.layer2 = self._make_layer(128, 2, stride=2)
        if split_layer_num <= 2:
            self.layer3 = self._make_layer(256, 2, stride=2)
        if split_layer_num <= 3:
            self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self._build_partial_block_completer()
        disable_inplace(self)

    def _make_layer(
        self, out_channels: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _build_partial_block_completer(self):
        split_sublayer = self.split_config["sublayer"]
        if split_sublayer is None:
            self.partial_block = None
            return

        split_layer_num = self.split_config["layer"]
        channel_map = {1: 64, 2: 128, 3: 256, 4: 512}
        out_channels = channel_map[split_layer_num]

        self.partial_block = nn.ModuleDict()
        if split_sublayer in ["conv1", "bn1", "relu1"]:
            if split_sublayer == "conv1":
                self.partial_block["bn1"] = nn.BatchNorm2d(out_channels)
            self.partial_block["relu"] = nn.ReLU(inplace=True)
            self.partial_block["conv2"] = nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            self.partial_block["bn2"] = nn.BatchNorm2d(out_channels)
            self.partial_block["final_relu"] = nn.ReLU(inplace=True)
        elif split_sublayer == "conv2":
            self.partial_block["bn2"] = nn.BatchNorm2d(out_channels)
            self.partial_block["final_relu"] = nn.ReLU(inplace=True)
        elif split_sublayer == "bn2":
            self.partial_block["final_relu"] = nn.ReLU(inplace=True)

    def _complete_partial_block(
        self, activation: torch.Tensor, identity: torch.Tensor
    ) -> torch.Tensor:
        split_sublayer = self.split_config["sublayer"]
        x = activation

        if self.partial_block is not None:
            if "bn1" in self.partial_block:
                x = self.partial_block["bn1"](x)
            if "relu" in self.partial_block and split_sublayer in ["conv1", "bn1"]:
                x = self.partial_block["relu"](x)
            if "conv2" in self.partial_block:
                x = self.partial_block["conv2"](x)
            if "bn2" in self.partial_block and split_sublayer != "bn2":
                x = self.partial_block["bn2"](x)
            x = x + identity
            x = self.partial_block["final_relu"](x)
        return x

    def forward(
        self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        split_layer_num = self.split_config["layer"]
        split_block = self.split_config["block"]

        x_tensor: torch.Tensor
        if isinstance(x, tuple):
            activation, identity = x
            x_tensor = self._complete_partial_block(activation, identity)
            if split_block is not None and split_block < 1:
                layer_attr = f"layer{split_layer_num}"
                if hasattr(self, layer_attr):
                    layer = getattr(self, layer_attr)
                    for i, block in enumerate(layer):
                        if i > split_block:
                            x_tensor = block(x_tensor)
        else:
            x_tensor = x

        if split_layer_num <= 1 and hasattr(self, "layer2"):
            x_tensor = self.layer2(x_tensor)
        if split_layer_num <= 2 and hasattr(self, "layer3"):
            x_tensor = self.layer3(x_tensor)
        if split_layer_num <= 3 and hasattr(self, "layer4"):
            x_tensor = self.layer4(x_tensor)

        x_tensor = self.avgpool(x_tensor)
        x_tensor = torch.flatten(x_tensor, 1)
        return self.fc(x_tensor)


# =============================================================================
# Model Selection Factory
# =============================================================================

ModelType = Literal[
    "simple",
    "alexnet",
    "alexnet_light",
    "resnet18",
    "resnet18_light",
    "resnet18_flex",
    "resnet18_image_style",
]
DatasetType = Literal["cifar10", "cifar100", "mnist", "fmnist", "synthetic"]


def get_split_models(
    model_type: ModelType = "simple",
    dataset: DatasetType = "cifar10",
    num_classes: int = 10,
    split_layer: Optional[str] = None,
) -> Tuple[nn.Module, nn.Module]:
    """
    Factory function to get client and server models for split learning.

    Args:
        model_type: Type of model architecture
            - 'simple': Legacy simple CNN (ClientNet/ServerNet)
            - 'alexnet': AlexNet split after Conv2 (default) or split_layer override
            - 'alexnet_light': AlexNet split after Conv1 (lighter client, default for CIFAR)
            - 'resnet18': ResNet-18 split after layer2 (half)
            - 'resnet18_light': ResNet-18 split after layer1 (quarter)
            - 'resnet18_flex': ResNet-18 with flexible split point
        dataset: Target dataset (affects input channels)
        num_classes: Number of output classes
        split_layer: For AlexNet (e.g., 'conv.2', 'conv.5') or ResNet flex (e.g., 'layer1', 'layer2.0.bn1')

    Returns:
        Tuple of (client_model, server_model)
    """
    is_grayscale = dataset in ["mnist", "fmnist"]

    if model_type == "simple":
        return ClientNet(), ServerNet(num_classes=num_classes)

    elif model_type in ["alexnet", "alexnet_light"]:
        base_model = (
            AlexNet(num_classes=num_classes)
            if is_grayscale
            else AlexNetCifar(num_classes=num_classes)
        )
        conv_layers = list(base_model.conv.children())
        if model_type == "alexnet":
            default_index = 5
        else:
            default_index = 5 if is_grayscale else 2
        split_index = _resolve_alexnet_split_index(
            split_layer, default_index, len(conv_layers)
        )
        return _build_alexnet_split_models(base_model, split_index)

    elif model_type == "resnet18":
        return ResNet18DownCifar(), ResNet18UpCifar(num_classes=num_classes)

    elif model_type == "resnet18_light":
        return ResNet18DownCifarLight(), ResNet18UpCifarHeavy(num_classes=num_classes)

    elif model_type == "resnet18_flex":
        layer = split_layer or "layer2"
        return ResNet18FlexibleClient(split_layer=layer), ResNet18FlexibleServer(
            split_layer=layer, num_classes=num_classes
        )

    elif model_type == "resnet18_image_style":
        layer = split_layer or "layer2"
        client = ResNet18ImageStyleFlexibleClient(split_layer=layer)
        server = ResNet18ImageStyleFlexibleServer(
            split_layer=layer, num_classes=num_classes
        )
        disable_inplace(client)
        disable_inplace(server)
        return client, server

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_full_model(
    model_type: ModelType = "simple",
    dataset: DatasetType = "cifar10",
    num_classes: int = 10,
) -> nn.Module:
    """
    Factory function to get a complete (non-split) model for evaluation.

    Args:
        model_type: Type of model architecture
        dataset: Target dataset
        num_classes: Number of output classes

    Returns:
        Complete model
    """
    is_grayscale = dataset in ["mnist", "fmnist"]

    if model_type in ["simple", "alexnet", "alexnet_light"]:
        if is_grayscale:
            return AlexNet(num_classes=num_classes)
        else:
            return AlexNetCifar(num_classes=num_classes)

    elif model_type in [
        "resnet18",
        "resnet18_light",
        "resnet18_flex",
    ]:
        return ResNet18Cifar(num_classes=num_classes)

    elif model_type == "resnet18_image_style":
        model = ResNet18ImageStyle(num_classes=num_classes)
        disable_inplace(model)
        return model

    else:
        raise ValueError(f"Unknown model type: {model_type}")


MODEL_INFO = {
    "simple": {
        "description": "Simple CNN (legacy)",
        "client_output_shape": (256,),
        "suitable_for": ["cifar10", "synthetic"],
    },
    "alexnet": {
        "description": "AlexNet split after Conv2",
        "client_output_shape": (64, 8, 8),
        "suitable_for": ["cifar10", "cifar100", "mnist", "fmnist"],
    },
    "alexnet_light": {
        "description": "AlexNet split after Conv1 (lighter client)",
        "client_output_shape": (32, 16, 16),
        "suitable_for": ["cifar10", "cifar100"],
    },
    "resnet18": {
        "description": "ResNet-18 split after layer2 (half)",
        "client_output_shape": (128, 16, 16),
        "suitable_for": ["cifar10", "cifar100"],
    },
    "resnet18_light": {
        "description": "ResNet-18 split after layer1 (quarter)",
        "client_output_shape": (64, 32, 32),
        "suitable_for": ["cifar10", "cifar100"],
    },
    "resnet18_flex": {
        "description": "ResNet-18 with flexible split point",
        "client_output_shape": "varies",
        "suitable_for": ["cifar10", "cifar100"],
    },
    "resnet18_image_style": {
        "description": "ResNet-18 with ImageNet-style stem",
        "client_output_shape": (128, 4, 4),
        "suitable_for": ["cifar10", "cifar100"],
    },
}


def load_torchvision_resnet18_init(
    client_model: nn.Module,
    server_model: nn.Module,
    split_layer: str = "layer2",
    image_style: bool = True,
) -> Tuple[nn.Module, nn.Module]:
    """
    Load torchvision ResNet18 state_dict into split client/server models.

    Ensures GAS/MultiSFL models start with the same Kaiming normal
    initialization as SFL's torchvision-based ResNet18.
    """
    if not image_style:
        print("[Warning] Torchvision init only supports image_style=True.")
        return client_model, server_model

    tv_resnet = tv_models.resnet18(weights=None)
    tv_state = tv_resnet.state_dict()

    split_config = parse_split_layer(split_layer)
    split_layer_num = split_config["layer"]
    split_block = split_config["block"]
    split_sublayer = split_config["sublayer"]

    client_keys: set = set()
    server_keys: set = set()

    for key in tv_state.keys():
        if key.startswith("conv1") or key.startswith("bn1"):
            client_keys.add(key)

    for layer_num in range(1, 5):
        layer_prefix = f"layer{layer_num}"
        layer_keys = [k for k in tv_state.keys() if k.startswith(layer_prefix)]

        if layer_num < split_layer_num:
            client_keys.update(layer_keys)
        elif layer_num > split_layer_num:
            server_keys.update(layer_keys)
        else:
            if split_block is None:
                client_keys.update(layer_keys)
            else:
                for key in layer_keys:
                    parts = key.split(".")
                    if len(parts) >= 2:
                        try:
                            block_num = int(parts[1])
                        except ValueError:
                            block_num = 0

                        if block_num < split_block:
                            client_keys.add(key)
                        elif block_num > split_block:
                            server_keys.add(key)
                        else:
                            if split_sublayer is None:
                                client_keys.add(key)
                            else:
                                sublayer_order = ["conv1", "bn1", "conv2", "bn2"]
                                key_sublayer = parts[2] if len(parts) > 2 else ""

                                if "downsample" in key:
                                    client_keys.add(key)
                                elif key_sublayer in sublayer_order:
                                    key_idx = sublayer_order.index(key_sublayer)
                                    split_idx = (
                                        sublayer_order.index(split_sublayer)
                                        if split_sublayer in sublayer_order
                                        else -1
                                    )
                                    if key_idx <= split_idx:
                                        client_keys.add(key)
                                    else:
                                        server_keys.add(key)
                                else:
                                    client_keys.add(key)
                    else:
                        client_keys.add(key)

    for key in tv_state.keys():
        if key.startswith("fc"):
            server_keys.add(key)

    client_state = client_model.state_dict()
    loaded_client = 0
    for key in client_keys:
        if key in client_state and key in tv_state:
            if client_state[key].shape == tv_state[key].shape:
                client_state[key] = tv_state[key]
                loaded_client += 1
    client_model.load_state_dict(client_state)

    server_state = server_model.state_dict()
    loaded_server = 0

    for key in server_keys:
        if key in server_state and key in tv_state:
            if server_state[key].shape == tv_state[key].shape:
                server_state[key] = tv_state[key]
                loaded_server += 1

    if (
        split_sublayer is not None
        and hasattr(server_model, "partial_block")
        and server_model.partial_block is not None
    ):
        block_prefix = f"layer{split_layer_num}.{split_block}"
        sublayer_map = {
            "bn1": f"{block_prefix}.bn1",
            "conv2": f"{block_prefix}.conv2",
            "bn2": f"{block_prefix}.bn2",
        }

        for pb_key, tv_key_prefix in sublayer_map.items():
            if pb_key in server_model.partial_block:
                for tv_key in tv_state.keys():
                    if tv_key.startswith(tv_key_prefix):
                        param_type = tv_key.split(".")[-1]
                        pb_full_key = f"partial_block.{pb_key}.{param_type}"

                        if pb_full_key in server_state and tv_key in tv_state:
                            if (
                                server_state[pb_full_key].shape
                                == tv_state[tv_key].shape
                            ):
                                server_state[pb_full_key] = tv_state[tv_key]
                                loaded_server += 1

    server_model.load_state_dict(server_state)

    print(
        f"[TorchvisionInit] Loaded {loaded_client} client params, {loaded_server} server params from torchvision ResNet18"
    )

    return client_model, server_model
