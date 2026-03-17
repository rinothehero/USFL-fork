"""
Model definitions with split-point support for SFL.

Each model can be split into (client_model, server_model) at a configurable layer.
Evaluation runs client → server sequentially — no state sync needed.

Supported: resnet18, vgg11, alexnet, mobilenet, lenet, deit_tiny
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Split model wrapper
# ---------------------------------------------------------------------------


class SplitModel:
    """
    Holds client_model + server_model as two independent nn.Modules.

    Evaluation: client_forward → server_forward (no separate full model needed).
    Aggregation: only client_model is aggregated across clients.
    """

    def __init__(
        self,
        client_model: nn.Module,
        server_model: nn.Module,
        num_classes: int,
        name: str = "",
    ):
        self.client_model = client_model
        self.server_model = server_model
        self.num_classes = num_classes
        self.name = name

    def evaluate(self, testloader, device) -> float:
        """Evaluate accuracy by running client → server sequentially."""
        self.client_model.eval()
        self.server_model.eval()
        self.client_model.to(device)
        self.server_model.to(device)

        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                activation = self.client_model(images)
                logits = self.server_model(activation)
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        return correct / total if total > 0 else 0.0

    def to(self, device):
        self.client_model.to(device)
        self.server_model.to(device)
        return self


# ---------------------------------------------------------------------------
# Layer splitting utilities
# ---------------------------------------------------------------------------


def _get_children_dict(model: nn.Module) -> OrderedDict:
    """Get named children as an OrderedDict."""
    return OrderedDict(model.named_children())


def _split_sequential(modules: OrderedDict, split_after: str) -> Tuple[nn.Sequential, nn.Sequential]:
    """Split an OrderedDict of modules into two nn.Sequential at the given key."""
    client_layers = OrderedDict()
    server_layers = OrderedDict()
    found = False
    for name, module in modules.items():
        if not found:
            client_layers[name] = module
            if name == split_after:
                found = True
        else:
            server_layers[name] = module

    if not found:
        raise ValueError(
            f"Split layer '{split_after}' not found. "
            f"Available: {list(modules.keys())}"
        )

    return nn.Sequential(client_layers), nn.Sequential(server_layers)


# ---------------------------------------------------------------------------
# ResNet18
# ---------------------------------------------------------------------------


class _ResNetClient(nn.Module):
    """Client portion of ResNet (up to and including split_layer)."""

    def __init__(self, base: nn.Module, split_layer: str):
        super().__init__()
        # ResNet layers in order: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4
        all_layers = ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4"]

        if split_layer not in all_layers:
            raise ValueError(f"Invalid split_layer '{split_layer}' for ResNet. Choose from {all_layers}")

        for name in all_layers:
            setattr(self, name, getattr(base, name))
            if name == split_layer:
                break

        self._split_layer = split_layer
        self._layers = all_layers[:all_layers.index(split_layer) + 1]

    def forward(self, x):
        for name in self._layers:
            x = getattr(self, name)(x)
        return x


class _ResNetServer(nn.Module):
    """Server portion of ResNet (after split_layer to output)."""

    def __init__(self, base: nn.Module, split_layer: str):
        super().__init__()
        all_layers = ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4"]

        idx = all_layers.index(split_layer) + 1
        self._layers = all_layers[idx:]

        for name in self._layers:
            setattr(self, name, getattr(base, name))

        self.avgpool = base.avgpool
        self.fc = base.fc

    def forward(self, x):
        for name in self._layers:
            x = getattr(self, name)(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def _create_resnet18(num_classes: int, split_layer: str, in_channels: int = 3) -> SplitModel:
    """Create split ResNet18."""
    import torchvision.models as models

    base = models.resnet18(weights=None, num_classes=num_classes)

    # Handle grayscale input (MNIST, FMNIST)
    if in_channels == 1:
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    client = _ResNetClient(base, split_layer)
    server = _ResNetServer(base, split_layer)
    return SplitModel(client, server, num_classes, name="resnet18")


# ---------------------------------------------------------------------------
# VGG11
# ---------------------------------------------------------------------------


def _create_vgg11(num_classes: int, split_layer: str, in_channels: int = 3) -> SplitModel:
    """Create split VGG11."""
    import torchvision.models as models

    base = models.vgg11(weights=None, num_classes=num_classes)

    if in_channels == 1:
        base.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)

    # VGG11 feature layers: 0-7 are conv/relu/pool groups
    # Named split points map to feature layer indices
    vgg_splits = {
        "features.3":  3,   # After first conv block
        "features.6":  6,   # After second conv block
        "features.8":  8,   # After third conv (first of double)
        "features.11": 11,  # After third conv block
        "features.13": 13,  # After fourth conv (first of double)
        "features.16": 16,  # After fourth conv block
        "features.18": 18,  # After fifth conv (first of double)
    }

    if split_layer not in vgg_splits:
        raise ValueError(f"Invalid split_layer '{split_layer}' for VGG11. Choose from {list(vgg_splits)}")

    idx = vgg_splits[split_layer]

    client_features = base.features[:idx + 1]
    server_features = base.features[idx + 1:]

    client = nn.Sequential(client_features)
    server = nn.Sequential(
        server_features,
        base.avgpool,
        nn.Flatten(),
        base.classifier,
    )

    return SplitModel(client, server, num_classes, name="vgg11")


# ---------------------------------------------------------------------------
# AlexNet
# ---------------------------------------------------------------------------


def _create_alexnet(num_classes: int, split_layer: str, in_channels: int = 3) -> SplitModel:
    """Create split AlexNet."""
    import torchvision.models as models

    base = models.alexnet(weights=None, num_classes=num_classes)

    if in_channels == 1:
        base.features[0] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)

    # AlexNet feature layers: conv-relu-pool groups
    alex_splits = {
        "features.2": 2,   # After first conv+relu+pool
        "features.5": 5,   # After second conv+relu+pool
        "features.7": 7,   # After third conv+relu
        "features.9": 9,   # After fourth conv+relu
        "features.12": 12, # After fifth conv+relu+pool (all features)
    }

    if split_layer not in alex_splits:
        raise ValueError(f"Invalid split_layer '{split_layer}' for AlexNet. Choose from {list(alex_splits)}")

    idx = alex_splits[split_layer]
    client = nn.Sequential(base.features[:idx + 1])
    server = nn.Sequential(
        base.features[idx + 1:],
        base.avgpool,
        nn.Flatten(),
        base.classifier,
    )

    return SplitModel(client, server, num_classes, name="alexnet")


# ---------------------------------------------------------------------------
# MobileNetV2
# ---------------------------------------------------------------------------


def _create_mobilenet(num_classes: int, split_layer: str, in_channels: int = 3) -> SplitModel:
    """Create split MobileNetV2."""
    import torchvision.models as models

    base = models.mobilenet_v2(weights=None, num_classes=num_classes)

    if in_channels == 1:
        first_conv = base.features[0][0]
        base.features[0][0] = nn.Conv2d(
            1, first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=False,
        )

    # MobileNetV2 has 19 feature blocks (0-18)
    n_features = len(base.features)
    valid_splits = {f"features.{i}": i for i in range(n_features)}

    if split_layer not in valid_splits:
        raise ValueError(f"Invalid split_layer '{split_layer}' for MobileNet. Choose from {list(valid_splits)}")

    idx = valid_splits[split_layer]
    client = nn.Sequential(*list(base.features[:idx + 1]))

    server_parts = list(base.features[idx + 1:])
    server = nn.Sequential(
        *server_parts,
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        base.classifier[-1] if isinstance(base.classifier, nn.Sequential) else base.classifier,
    )

    return SplitModel(client, server, num_classes, name="mobilenet")


# ---------------------------------------------------------------------------
# LeNet-5
# ---------------------------------------------------------------------------


class _LeNet5(nn.Module):
    """Simple LeNet-5 for MNIST/FMNIST."""

    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(nn.Linear(16 * 5 * 5, 120), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(120, 84), nn.ReLU())
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def _create_lenet(num_classes: int, split_layer: str, in_channels: int = 1) -> SplitModel:
    """Create split LeNet-5."""
    base = _LeNet5(in_channels, num_classes)

    lenet_layers = ["conv1", "conv2", "flatten", "fc1", "fc2", "fc3"]
    if split_layer not in lenet_layers:
        raise ValueError(f"Invalid split_layer '{split_layer}' for LeNet. Choose from {lenet_layers}")

    idx = lenet_layers.index(split_layer) + 1
    client_layers = OrderedDict()
    server_layers = OrderedDict()
    for i, name in enumerate(lenet_layers):
        if i < idx:
            client_layers[name] = getattr(base, name)
        else:
            server_layers[name] = getattr(base, name)

    client = nn.Sequential(client_layers)
    server = nn.Sequential(server_layers)
    return SplitModel(client, server, num_classes, name="lenet")


# ---------------------------------------------------------------------------
# DeiT (tiny)
# ---------------------------------------------------------------------------


def _create_deit_tiny(num_classes: int, split_layer: str, in_channels: int = 3) -> SplitModel:
    """
    Create split DeiT-tiny using timm.

    Split points are transformer block indices: "blocks.0", "blocks.1", ..., "blocks.11"
    """
    try:
        import timm
    except ImportError:
        raise ImportError("timm is required for DeiT models: pip install timm")

    base = timm.create_model("deit_tiny_patch16_224", pretrained=False, num_classes=num_classes)

    if in_channels == 1:
        old_proj = base.patch_embed.proj
        base.patch_embed.proj = nn.Conv2d(
            1, old_proj.out_channels,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            padding=old_proj.padding,
        )

    # Parse split point
    if not split_layer.startswith("blocks."):
        valid = [f"blocks.{i}" for i in range(len(base.blocks))]
        raise ValueError(f"Invalid split_layer '{split_layer}' for DeiT. Choose from {valid}")

    split_idx = int(split_layer.split(".")[1]) + 1

    class DeiTClient(nn.Module):
        def __init__(self, base, split_idx):
            super().__init__()
            self.patch_embed = base.patch_embed
            self.cls_token = base.cls_token
            self.pos_embed = base.pos_embed
            self.pos_drop = base.pos_drop
            self.blocks = nn.Sequential(*list(base.blocks[:split_idx]))

        def forward(self, x):
            x = self.patch_embed(x)
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            x = x + self.pos_embed
            x = self.pos_drop(x)
            x = self.blocks(x)
            return x

    class DeiTServer(nn.Module):
        def __init__(self, base, split_idx):
            super().__init__()
            self.blocks = nn.Sequential(*list(base.blocks[split_idx:]))
            self.norm = base.norm
            self.head = base.head

        def forward(self, x):
            x = self.blocks(x)
            x = self.norm(x)
            x = x[:, 0]  # CLS token
            x = self.head(x)
            return x

    client = DeiTClient(base, split_idx)
    server = DeiTServer(base, split_idx)
    return SplitModel(client, server, num_classes, name="deit_tiny")


# ---------------------------------------------------------------------------
# MLP Classifier
# ---------------------------------------------------------------------------


class _MLPClient(nn.Module):
    """Client portion of MLP (up to and including split_layer)."""

    def __init__(self, in_features: int, split_layer: str):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Sequential(nn.Linear(in_features, 512), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(512, 256), nn.ReLU())

        if split_layer == "layer1":
            self._layers = ["layer1"]
        elif split_layer == "layer2":
            self._layers = ["layer1", "layer2"]
        else:
            raise ValueError(f"Invalid split_layer '{split_layer}' for MLP. Choose from ['layer1', 'layer2']")

    def forward(self, x):
        x = self.flatten(x)
        for name in self._layers:
            x = getattr(self, name)(x)
        return x


class _MLPServer(nn.Module):
    """Server portion of MLP (after split_layer to output)."""

    def __init__(self, split_layer: str, num_classes: int):
        super().__init__()
        if split_layer == "layer1":
            self.layer2 = nn.Sequential(nn.Linear(512, 256), nn.ReLU())
            self.layer3 = nn.Linear(256, num_classes)
            self._layers = ["layer2", "layer3"]
        elif split_layer == "layer2":
            self.layer3 = nn.Linear(256, num_classes)
            self._layers = ["layer3"]
        else:
            raise ValueError(f"Invalid split_layer '{split_layer}' for MLP. Choose from ['layer1', 'layer2']")

    def forward(self, x):
        for name in self._layers:
            x = getattr(self, name)(x)
        return x


# Input feature sizes by dataset for MLP
_MLP_IN_FEATURES = {
    "cifar10": 3 * 32 * 32,    # 3072
    "cifar100": 3 * 32 * 32,   # 3072
    "svhn": 3 * 32 * 32,       # 3072
    "mnist": 1 * 28 * 28,      # 784
    "fmnist": 1 * 28 * 28,     # 784
}


def _create_mlp(num_classes: int, split_layer: str, in_channels: int = 3, dataset: str = "cifar10") -> SplitModel:
    """Create split MLP classifier."""
    in_features = _MLP_IN_FEATURES.get(dataset, 3 * 32 * 32)

    client = _MLPClient(in_features, split_layer)
    server = _MLPServer(split_layer, num_classes)
    return SplitModel(client, server, num_classes, name="mlp")


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

# Default split points per model
_DEFAULT_SPLITS = {
    "resnet18": "layer2",
    "resnet18_flex": "layer2",  # Alias for compatibility
    "vgg11": "features.8",
    "alexnet": "features.5",
    "mobilenet": "features.7",
    "lenet": "conv2",
    "deit_tiny": "blocks.5",
    "mlp": "layer1",
}

# Input channels by dataset
_IN_CHANNELS = {
    "cifar10": 3, "cifar100": 3, "svhn": 3,
    "mnist": 1, "fmnist": 1,
}


def create_model(
    model_name: str,
    num_classes: int,
    split_layer: Optional[str] = None,
    dataset: str = "cifar10",
) -> SplitModel:
    """
    Create a split model.

    Args:
        model_name: Model architecture name
        num_classes: Number of output classes
        split_layer: Where to split (None = use default for model)
        dataset: Dataset name (determines input channels)

    Returns: SplitModel with client_model and server_model
    """
    # Resolve model name aliases
    canonical = model_name.lower().replace("-", "_")
    if canonical == "resnet18_flex":
        canonical = "resnet18"

    if split_layer is None:
        split_layer = _DEFAULT_SPLITS.get(canonical)
        if split_layer is None:
            raise ValueError(f"No default split for model '{canonical}'")

    in_channels = _IN_CHANNELS.get(dataset, 3)

    creators = {
        "resnet18": _create_resnet18,
        "vgg11": _create_vgg11,
        "alexnet": _create_alexnet,
        "mobilenet": _create_mobilenet,
        "lenet": _create_lenet,
        "deit_tiny": _create_deit_tiny,
        "mlp": lambda nc, sl, ic: _create_mlp(nc, sl, ic, dataset),
    }

    if canonical not in creators:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(creators)}")

    return creators[canonical](num_classes, split_layer, in_channels)
