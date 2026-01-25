"""
FlexibleResNet18 - Layer boundary based split support for SFL Framework.

Unlike the existing ResnetSplitter which only splits at leaf modules (e.g., layer1.1.bn2),
this module allows splitting at layer boundaries (e.g., layer2, layer3, layer4),
similar to GAS and MultiSFL implementations.

Split at layer boundary means:
- No tuple handling needed (residual connections fully contained)
- Cleaner gradient flow
- Consistent with GAS/MultiSFL for fair comparison
"""

from typing import TYPE_CHECKING, Tuple, Union, Optional
import copy

import torch
from torch import nn
import torchvision

from .base_model import BaseModel

if TYPE_CHECKING:
    from server_args import Config
    from torch.utils.data import DataLoader


def disable_inplace(module: nn.Module) -> None:
    """Disable inplace operations for gradient computation."""
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, nn.ReLU(inplace=False))
        elif isinstance(child, nn.ReLU6):
            setattr(module, name, nn.ReLU6(inplace=False))
        else:
            disable_inplace(child)


def parse_split_layer(split_layer: str) -> dict:
    """
    Parse split_layer string into components.

    Examples:
        "layer2" -> {"layer": 2, "block": None, "sublayer": None}
        "layer1.1.bn2" -> {"layer": 1, "block": 1, "sublayer": "bn2"}
    """
    result = {"layer": 1, "block": None, "sublayer": None}
    parts = split_layer.split(".")

    if parts[0].startswith("layer"):
        result["layer"] = int(parts[0].replace("layer", ""))
    if len(parts) >= 2:
        result["block"] = int(parts[1])
    if len(parts) >= 3:
        result["sublayer"] = parts[2]

    return result


class BasicBlock(nn.Module):
    """Basic residual block for ResNet-18."""

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
        self.relu = nn.ReLU(inplace=False)
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


class FlexibleResNetClient(nn.Module):
    """
    Client-side ResNet-18 with flexible split point.

    Supports both:
    - Layer boundary splits (layer2, layer3, layer4) -> returns Tensor
    - Mid-block splits (layer1.1.bn2) -> returns (activation, identity) Tuple
    """

    def __init__(self, split_layer: str = "layer2", cifar_style: bool = False):
        super(FlexibleResNetClient, self).__init__()
        self.split_config = parse_split_layer(split_layer)
        self.split_layer = split_layer
        self.in_channels = 64
        self.cifar_style = cifar_style

        # Initial conv layer
        if cifar_style:
            # CIFAR-10 style: 3x3 conv, no maxpool
            self.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.maxpool = nn.Identity()
        else:
            # ImageNet style: 7x7 conv with maxpool
            self.conv1 = nn.Conv2d(
                3, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)

        # Build layers up to split point
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
        """Forward through a BasicBlock partially, returning (activation, identity) tuple."""
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

        # Full block completed
        out += identity
        out = block.relu(out)
        return out

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

        # Layer 1
        if split_layer == 1:
            if split_block is not None and split_sublayer is not None:
                for i, block in enumerate(self.layer1):
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

        # Layer 2
        if split_layer == 2:
            if split_block is not None and split_sublayer is not None:
                for i, block in enumerate(self.layer2):
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

        # Layer 3
        if split_layer == 3:
            if split_block is not None and split_sublayer is not None:
                for i, block in enumerate(self.layer3):
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

        # Layer 4
        if split_layer == 4:
            if hasattr(self, "layer4"):
                x = self.layer4(x)
            return x

        return x


class FlexibleResNetServer(nn.Module):
    """
    Server-side ResNet-18 with flexible split point.

    Handles both:
    - Tensor input (from layer boundary split)
    - Tuple input (from mid-block split) - completes residual add
    """

    def __init__(self, split_layer: str = "layer2", num_classes: int = 10):
        super(FlexibleResNetServer, self).__init__()
        self.split_config = parse_split_layer(split_layer)
        self.split_layer = split_layer

        channel_map = {1: 64, 2: 128, 3: 256, 4: 512}
        split_layer_num = self.split_config["layer"]
        self.in_channels = channel_map.get(split_layer_num, 64)

        # Build remaining layers after split point
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
        """Build layers to complete a partial block if split in middle."""
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
            self.partial_block["relu"] = nn.ReLU(inplace=False)
            self.partial_block["conv2"] = nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            self.partial_block["bn2"] = nn.BatchNorm2d(out_channels)
            self.partial_block["final_relu"] = nn.ReLU(inplace=False)
        elif split_sublayer == "conv2":
            self.partial_block["bn2"] = nn.BatchNorm2d(out_channels)
            self.partial_block["final_relu"] = nn.ReLU(inplace=False)
        elif split_sublayer == "bn2":
            self.partial_block["final_relu"] = nn.ReLU(inplace=False)

    def _complete_partial_block(
        self, out: torch.Tensor, identity: torch.Tensor, stop_at: str
    ) -> torch.Tensor:
        """Complete a partial block by finishing remaining operations."""
        if self.partial_block is None:
            raise ValueError("Partial block is not initialized")

        if "bn1" in self.partial_block:
            out = self.partial_block["bn1"](out)
        if "relu" in self.partial_block and stop_at in ["conv1", "bn1"]:
            out = self.partial_block["relu"](out)
        if "conv2" in self.partial_block:
            out = self.partial_block["conv2"](out)
        if "bn2" in self.partial_block and stop_at != "bn2":
            out = self.partial_block["bn2"](out)

        out += identity
        out = self.partial_block["final_relu"](out)
        return out

    def forward(
        self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        split_layer = self.split_config["layer"]
        split_sublayer = self.split_config["sublayer"]

        # Handle tuple input from mid-block split
        if isinstance(x, tuple) and self.partial_block is not None:
            out, identity = x
            out = self._complete_partial_block(out, identity, split_sublayer)
        else:
            out = x

        # Continue through remaining layers
        if split_layer <= 1 and hasattr(self, "layer2"):
            out = self.layer2(out)
        if split_layer <= 2 and hasattr(self, "layer3"):
            out = self.layer3(out)
        if split_layer <= 3 and hasattr(self, "layer4"):
            out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


class FlexibleResNet(BaseModel):
    """
    FlexibleResNet wrapper for SFL Framework.

    Provides layer-boundary split support (layer2, layer3, layer4)
    in addition to existing leaf-based splits.

    Usage:
        config.model = "resnet18_flex"
        config.split_layer = "layer2"  # or "layer3", "layer4", "layer1.1.bn2"
    """

    def __init__(self, config: "Config", num_classes: int):
        super().__init__(config)
        self.num_classes = num_classes
        self.config = config

        split_layer = getattr(config, "split_layer", "layer2")
        force_imagenet_style = getattr(config, "force_imagenet_style", False)
        cifar_style = (
            config.dataset in ["cifar10", "cifar100", "fmnist", "mnist"]
            and not force_imagenet_style
        )

        self.split_layer = split_layer
        self.cifar_style = cifar_style

        # Create client and server models
        self.client_model = FlexibleResNetClient(split_layer, cifar_style)
        self.server_model = FlexibleResNetServer(split_layer, num_classes)

        # Create full model for evaluation
        self._create_full_model()

        self.client_model.to(config.device)
        self.server_model.to(config.device)
        self.torch_model.to(config.device)

    def _create_full_model(self):
        """Create a full model for evaluation."""
        if self.cifar_style:
            self.torch_model = torchvision.models.resnet18(
                weights=None, num_classes=self.num_classes
            )
            self.torch_model.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.torch_model.maxpool = nn.Identity()
        else:
            self.torch_model = torchvision.models.resnet18(
                weights=None, num_classes=self.num_classes
            )

        disable_inplace(self.torch_model)

    def get_split_models(self) -> Tuple[nn.Module, nn.Module]:
        """Return (client_model, server_model) tuple."""
        return self.client_model, self.server_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass (for evaluation)."""
        return self.torch_model(x)

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        self.torch_model.eval()
        with torch.no_grad():
            return self.forward(inputs)

    def save_model(self, save_path: str) -> None:
        torch.save(
            {
                "client": self.client_model.state_dict(),
                "server": self.server_model.state_dict(),
                "full": self.torch_model.state_dict(),
            },
            save_path + ".pth",
        )

    def load_model(self, load_path: str) -> None:
        state = torch.load(load_path + ".pth")
        self.client_model.load_state_dict(state["client"])
        self.server_model.load_state_dict(state["server"])
        self.torch_model.load_state_dict(state["full"])

    def get_torch_model(self) -> nn.Module:
        return self.torch_model

    def set_torch_model(self, torch_model: nn.Module):
        torch_model.to(self.config.device)
        self.torch_model = torch_model

    def sync_full_model_from_split(self):
        """Sync full model parameters from split client/server models."""
        split_config = parse_split_layer(self.split_layer)
        split_layer_num = split_config["layer"]

        # Copy client params
        client_state = self.client_model.state_dict()
        full_state = self.torch_model.state_dict()

        for key in client_state:
            if key in full_state:
                full_state[key] = client_state[key].clone()

        # Copy server params
        server_state = self.server_model.state_dict()
        for key in server_state:
            if key in full_state:
                full_state[key] = server_state[key].clone()

        self.torch_model.load_state_dict(full_state)

    def evaluate(self, testloader: "DataLoader") -> float:
        """Evaluate using full model."""
        self.sync_full_model_from_split()
        self.torch_model.eval()
        self.torch_model.to(self.config.device)
        correct = 0
        total = 0

        with torch.no_grad():
            for data in testloader:
                inputs, labels = data
                inputs, labels = (
                    inputs.to(self.config.device),
                    labels.to(self.config.device),
                )
                outputs = self.forward(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total
