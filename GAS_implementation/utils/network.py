import torch
import torch.nn as nn
import numpy as np
from typing import Optional, TypedDict


def disable_inplace(module: nn.Module) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, nn.ReLU(inplace=False))
        elif isinstance(child, nn.ReLU6):
            setattr(module, name, nn.ReLU6(inplace=False))
        else:
            disable_inplace(child)


def model_selection(
    cifar=False,
    mnist=False,
    fmnist=False,
    cinic=False,
    cifar100=False,
    SVHN=False,
    split=False,
    twoLogit=False,
    resnet=False,
    resnet_image_style=False,
    split_ratio="half",
    split_layer=None,
    split_alexnet="default",
):
    """
    Model selection for Split Federated Learning.

    Args:
        split_ratio: Legacy ResNet split point ('half', 'quarter')
        split_layer: Fine-grained split point (e.g., 'layer1', 'layer1.0.bn1', 'layer2')
                     If specified, overrides split_ratio
        split_alexnet: AlexNet split point ('default': Conv2 후, 'light': Conv1 후)
    """
    num_classes = 10
    if cifar100:
        num_classes = 100
    if split:
        server_local_model = None
        if cifar or cinic or cifar100 or SVHN:
            if resnet:
                if split_layer is not None or resnet_image_style:
                    if resnet_image_style:
                        effective_split_layer = split_layer or "layer2"
                        user_model = ResNet18ImageStyleFlexibleClient(
                            split_layer=effective_split_layer
                        )
                        server_model = ResNet18ImageStyleFlexibleServer(
                            split_layer=effective_split_layer, num_classes=num_classes
                        )
                        if twoLogit:
                            server_local_model = ResNet18ImageStyleFlexibleServer(
                                split_layer=effective_split_layer,
                                num_classes=num_classes,
                            )
                    else:
                        # Use flexible split with fine-grained control
                        effective_split_layer = split_layer or "layer2"
                        user_model = ResNet18FlexibleClient(
                            split_layer=effective_split_layer
                        )
                        server_model = ResNet18FlexibleServer(
                            split_layer=effective_split_layer, num_classes=num_classes
                        )
                        if twoLogit:
                            server_local_model = ResNet18FlexibleServer(
                                split_layer=effective_split_layer,
                                num_classes=num_classes,
                            )
                elif split_ratio == "quarter":
                    # Legacy: 1/4 Down, 3/4 Up (lighter client)
                    user_model = ResNet18DownCifarLight()
                    server_model = ResNet18UpCifarHeavy(num_classes=num_classes)
                    if twoLogit:
                        server_local_model = ResNet18UpCifarHeavy(
                            num_classes=num_classes
                        )
                else:  # 'half' or default
                    # Legacy: 1/2 Down, 1/2 Up (balanced)
                    user_model = ResNet18DownCifar()
                    server_model = ResNet18UpCifar(num_classes=num_classes)
                    if twoLogit:
                        server_local_model = ResNet18UpCifar(num_classes=num_classes)
            else:
                # Use AlexNet for Split Learning with configurable split point
                if split_alexnet == "light":
                    # Option A: Conv1 이후 (lighter client, more communication)
                    user_model = AlexNetDownCifarLight()
                    server_model = AlexNetUpCifarHeavy(num_classes=num_classes)
                    if twoLogit:
                        server_local_model = AlexNetUpCifarHeavy(
                            num_classes=num_classes
                        )
                else:  # 'default'
                    # Option B: Conv2 이후 (default, balanced)
                    user_model = AlexNetDownCifar()
                    server_model = AlexNetUpCifar(num_classes=num_classes)
                    if twoLogit:
                        server_local_model = AlexNetUpCifar(num_classes=num_classes)
        elif mnist or fmnist:
            user_model = AlexNetDown()
            server_model = AlexNetUp()
            if twoLogit:
                server_local_model = AlexNetUp()
        else:
            user_model = None
            server_model = None
        if twoLogit:
            return user_model, server_model, server_local_model
        else:
            return user_model, server_model
    else:
        if cifar or cinic or cifar100 or SVHN:
            if resnet:
                # Use complete ResNet-18 (not split)
                if resnet_image_style:
                    model = ResNet18ImageStyle(num_classes=num_classes)
                else:
                    model = ResNet18Cifar(num_classes=num_classes)
            else:
                # Use complete AlexNet (default)
                model = AlexNetCifar(num_classes=num_classes)
        elif mnist or fmnist:
            model = AlexNet()
        else:
            model = None

        return model


def inversion_model(feature_size=None):
    model = custom_AE(input_nc=64, output_nc=3, input_dim=8, output_dim=32)
    return model


class custom_AE(nn.Module):
    def __init__(
        self,
        input_nc=256,
        output_nc=3,
        input_dim=8,
        output_dim=32,
        activation="sigmoid",
    ):
        super(custom_AE, self).__init__()
        upsampling_num = int(np.log2(output_dim // input_dim))
        # get [b, 3, 8, 8]
        model = []
        nc = input_nc
        for num in range(upsampling_num - 1):
            # TODO: change to Conv2d
            model += [nn.Conv2d(nc, int(nc / 2), kernel_size=3, stride=1, padding=1)]
            model += [nn.ReLU()]
            model += [
                nn.ConvTranspose2d(
                    int(nc / 2),
                    int(nc / 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                )
            ]
            model += [nn.ReLU()]
            nc = int(nc / 2)
        if upsampling_num >= 1:
            model += [
                nn.Conv2d(
                    int(input_nc / (2 ** (upsampling_num - 1))),
                    int(input_nc / (2 ** (upsampling_num - 1))),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            ]
            model += [nn.ReLU()]
            model += [
                nn.ConvTranspose2d(
                    int(input_nc / (2 ** (upsampling_num - 1))),
                    output_nc,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                )
            ]
            if activation == "sigmoid":
                model += [nn.Sigmoid()]
            elif activation == "tanh":
                model += [nn.Tanh()]
        else:
            model += [nn.Conv2d(input_nc, input_nc, kernel_size=3, stride=1, padding=1)]
            model += [nn.ReLU()]
            model += [
                nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1)
            ]
            if activation == "sigmoid":
                model += [nn.Sigmoid()]
            elif activation == "tanh":
                model += [nn.Tanh()]
        self.m = nn.Sequential(*model)

    def forward(self, x):
        output = self.m(x)
        return output


class AlexNetCifar(nn.Module):
    def __init__(self, num_classes=10):
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

    def forward(self, x, return_features=False):
        out = self.conv(x)
        features = out.view(-1, 256 * 3 * 3)
        out = self.fc(features)
        if return_features:
            return out, features
        else:
            return out


class AlexNet(nn.Module):
    def __init__(self):
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
            nn.Linear(512, 10),
        )

    def forward(self, x, return_features=False):
        out = self.conv(x)
        features = out.view(-1, 256 * 3 * 3)
        out = self.fc(features)
        if return_features:
            return out, features
        else:
            return out


class AlexNetDownCifar(nn.Module):
    def __init__(self, width_mult=1):
        super(AlexNetDownCifar, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class AlexNetUpCifar(nn.Module):
    def __init__(self, width_mult=1, num_classes=10):
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

    def forward(self, x):
        x = self.model2(x)
        x = x.view(-1, 256 * 3 * 3)
        x = self.classifier(x)
        return x


class AlexNetDownCifarLight(nn.Module):
    """Client-side AlexNet Light for CIFAR-10 (split after Conv1 - Option A)"""

    def __init__(self, width_mult=1):
        super(AlexNetDownCifarLight, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Input: [B, 3, 32, 32]
        # Output: [B, 32, 16, 16] - activation size 8,192
        x = self.model(x)
        return x


class AlexNetUpCifarHeavy(nn.Module):
    """Server-side AlexNet Heavy for CIFAR-10 (receives Conv1 output - Option A)"""

    def __init__(self, width_mult=1, num_classes=10):
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

    def forward(self, x):
        # Input: [B, 32, 16, 16] from client
        x = self.model2(x)
        x = x.view(-1, 256 * 3 * 3)
        x = self.classifier(x)
        return x


class AlexNetDown(nn.Module):
    def __init__(self, width_mult=1):
        super(AlexNetDown, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class AlexNetUp(nn.Module):
    def __init__(self, width_mult=1):
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
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.model2(x)
        x = x.view(-1, 256 * 3 * 3)
        x = self.classifier(x)
        return x


# ============== ResNet for CIFAR-10 ==============


class BasicBlock(nn.Module):
    """Basic residual block for ResNet-18/34"""

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
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

    def forward(self, x):
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


class ResNet18Cifar(nn.Module):
    """ResNet-18 for CIFAR-10 (complete model, not split)"""

    def __init__(self, num_classes=10):
        super(ResNet18Cifar, self).__init__()
        self.in_channels = 64

        # Initial convolution (smaller for CIFAR-10's 32x32 images)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # ResNet layers
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        disable_inplace(self)

    def _make_layer(self, out_channels, num_blocks, stride):
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

    def forward(self, x):
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
    """Client-side ResNet-18 for CIFAR-10 (split after layer2)"""

    def __init__(self):
        super(ResNet18DownCifar, self).__init__()
        self.in_channels = 64

        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # First two ResNet layers (client side)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)

    def _make_layer(self, out_channels, num_blocks, stride):
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

    def forward(self, x):
        # Input: [B, 3, 32, 32]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # After conv1: [B, 64, 32, 32]

        x = self.layer1(x)
        # After layer1: [B, 64, 32, 32]

        x = self.layer2(x)
        # After layer2: [B, 128, 16, 16]
        # This is the activation sent to server

        return x


class ResNet18UpCifar(nn.Module):
    """Server-side ResNet-18 for CIFAR-10 (receives layer2 output)"""

    def __init__(self, num_classes=10):
        super(ResNet18UpCifar, self).__init__()
        self.in_channels = 128

        # Remaining ResNet layers (server side)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
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

    def forward(self, x):
        # Input: [B, 128, 16, 16] from client

        x = self.layer3(x)
        # After layer3: [B, 256, 8, 8]

        x = self.layer4(x)
        # After layer4: [B, 512, 4, 4]

        x = self.avgpool(x)
        # After avgpool: [B, 512, 1, 1]

        x = torch.flatten(x, 1)
        # After flatten: [B, 512]

        x = self.fc(x)
        # Output: [B, num_classes]

        return x


class ResNet18DownCifarLight(nn.Module):
    """Client-side ResNet-18 Light for CIFAR-10 (split after layer1 only - 1/4 computation)"""

    def __init__(self):
        super(ResNet18DownCifarLight, self).__init__()
        self.in_channels = 64

        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Only first ResNet layer (lighter client side)
        self.layer1 = self._make_layer(64, 2, stride=1)

    def _make_layer(self, out_channels, num_blocks, stride):
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

    def forward(self, x):
        # Input: [B, 3, 32, 32]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # After conv1: [B, 64, 32, 32]

        x = self.layer1(x)
        # After layer1: [B, 64, 32, 32]
        # This is the activation sent to server (smaller than half split)

        return x


class ResNet18UpCifarHeavy(nn.Module):
    """Server-side ResNet-18 Heavy for CIFAR-10 (receives layer1 output - 3/4 computation)"""

    def __init__(self, num_classes=10):
        super(ResNet18UpCifarHeavy, self).__init__()
        self.in_channels = 64

        # Remaining ResNet layers (heavier server side)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
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

    def forward(self, x):
        # Input: [B, 64, 32, 32] from client (layer1 output)

        x = self.layer2(x)
        # After layer2: [B, 128, 16, 16]

        x = self.layer3(x)
        # After layer3: [B, 256, 8, 8]

        x = self.layer4(x)
        # After layer4: [B, 512, 4, 4]

        x = self.avgpool(x)
        # After avgpool: [B, 512, 1, 1]

        x = torch.flatten(x, 1)
        # After flatten: [B, 512]

        x = self.fc(x)
        # Output: [B, num_classes]

        return x


# =============================================================================
# Flexible Split ResNet-18 (supports fine-grained split points)
# =============================================================================


class SplitConfig(TypedDict):
    layer: int
    block: Optional[int]
    sublayer: Optional[str]


def parse_split_layer(split_layer: str) -> SplitConfig:
    """
    Parse split_layer string into components.

    Examples:
        'layer1' -> {'layer': 1, 'block': None, 'sublayer': None}
        'layer1.0' -> {'layer': 1, 'block': 0, 'sublayer': None}
        'layer1.0.bn1' -> {'layer': 1, 'block': 0, 'sublayer': 'bn1'}
        'layer2.1.conv2' -> {'layer': 2, 'block': 1, 'sublayer': 'conv2'}
    """
    parts = split_layer.split(".")
    result: SplitConfig = {"layer": 1, "block": None, "sublayer": None}

    if len(parts) >= 1:
        # Extract layer number from 'layer1', 'layer2', etc.
        result["layer"] = int(parts[0].replace("layer", ""))
    if len(parts) >= 2:
        result["block"] = int(parts[1])
    if len(parts) >= 3:
        result["sublayer"] = parts[2]  # 'conv1', 'bn1', 'conv2', 'bn2'

    return result


class ResNet18ImageStyleFlexibleClient(nn.Module):
    def __init__(self, split_layer="layer2"):
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

    def _make_layer(self, out_channels, num_blocks, stride):
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

    def _forward_partial_block(self, block, x, stop_at):
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
        return out

    def forward(self, x):
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
    def __init__(self, split_layer="layer2", num_classes=10):
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

    def _make_layer(self, out_channels, num_blocks, stride):
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
        self.partial_block = None
        split_layer = self.split_config["layer"]
        split_block = self.split_config["block"]

        if split_block is None:
            return

        if split_layer == 1 and split_block < 2:
            self.partial_block = BasicBlock(64, 64)
        elif split_layer == 2 and split_block < 2:
            self.partial_block = BasicBlock(128, 128)
        elif split_layer == 3 and split_block < 2:
            self.partial_block = BasicBlock(256, 256)

    def _complete_partial_block(self, out, identity, stop_at):
        if self.partial_block is None:
            raise ValueError("Partial block is not initialized")

        partial_block = self.partial_block
        if stop_at == "conv1":
            out = partial_block.bn1(out)
            out = partial_block.relu(out)
            out = partial_block.conv2(out)
            out = partial_block.bn2(out)
        elif stop_at == "bn1":
            out = partial_block.relu(out)
            out = partial_block.conv2(out)
            out = partial_block.bn2(out)
        elif stop_at == "relu1":
            out = partial_block.conv2(out)
            out = partial_block.bn2(out)
        elif stop_at == "conv2":
            out = partial_block.bn2(out)
        elif stop_at == "bn2":
            pass
        else:
            raise ValueError(f"Unknown stop_at: {stop_at}")

        out += identity
        out = partial_block.relu(out)
        return out

    def forward(self, x):
        split_layer = self.split_config["layer"]
        split_block = self.split_config["block"]
        split_sublayer = self.split_config["sublayer"]

        if isinstance(x, tuple) and self.partial_block is not None:
            out, identity = x
            out = self._complete_partial_block(out, identity, split_sublayer)
        else:
            out = x

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


class ResNet18FlexibleClient(nn.Module):
    """
    Client-side ResNet-18 with flexible split point.

    If split is in the middle of a BasicBlock, returns (activation, identity) tuple.
    If split is at the end of a layer, returns activation only.
    """

    def __init__(self, split_layer="layer2"):
        super(ResNet18FlexibleClient, self).__init__()
        self.split_config = parse_split_layer(split_layer)
        self.split_layer = split_layer
        self.in_channels = 64

        # Always include initial conv
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Build layers up to split point
        self.layer1 = self._make_layer(64, 2, stride=1)

        if self.split_config["layer"] >= 2:
            self.layer2 = self._make_layer(128, 2, stride=2)
        if self.split_config["layer"] >= 3:
            self.layer3 = self._make_layer(256, 2, stride=2)
        if self.split_config["layer"] >= 4:
            self.layer4 = self._make_layer(512, 2, stride=2)

    def _make_layer(self, out_channels, num_blocks, stride):
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

    def _forward_partial_block(self, block, x, stop_at):
        """Forward through a BasicBlock partially, stopping at stop_at sublayer."""
        identity = x

        # downsample identity first if needed
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

        # Full block (shouldn't reach here if sublayer specified)
        out += identity
        out = block.relu(out)
        return out

    def forward(self, x):
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        split_layer = self.split_config["layer"]
        split_block = self.split_config["block"]
        split_sublayer = self.split_config["sublayer"]

        # Layer 1
        if split_layer == 1:
            if split_block is not None and split_sublayer is not None:
                # Partial layer1
                for i, block in enumerate(self.layer1):
                    if i < split_block:
                        x = block(x)
                    elif i == split_block:
                        return self._forward_partial_block(block, x, split_sublayer)
                    # blocks after split_block are not executed
            elif split_block is not None:
                # Up to specific block (complete)
                for i, block in enumerate(self.layer1):
                    if i <= split_block:
                        x = block(x)
                return x
            else:
                # Full layer1
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

        # Layer 4 (unlikely to be client-side but included for completeness)
        if split_layer == 4:
            if hasattr(self, "layer4"):
                x = self.layer4(x)
            return x

        return x


class ResNet18FlexibleServer(nn.Module):
    """
    Server-side ResNet-18 with flexible split point.

    If input is a tuple (activation, identity), continues from mid-block.
    If input is a tensor, continues from layer boundary.
    """

    def __init__(self, split_layer="layer2", num_classes=10):
        super(ResNet18FlexibleServer, self).__init__()
        self.split_config = parse_split_layer(split_layer)
        self.split_layer = split_layer

        # Determine starting in_channels based on split point
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

        # Store info for partial block completion
        self._build_partial_block_completer()

    def _make_layer(self, out_channels, num_blocks, stride):
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

        # Build the remaining sublayers of the block
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

    def _complete_partial_block(self, activation, identity):
        """Complete a partially executed BasicBlock."""
        split_sublayer = self.split_config["sublayer"]
        x = activation

        if self.partial_block is None:
            return x

        partial_block = self.partial_block
        if "bn1" in partial_block:
            x = partial_block["bn1"](x)
        if "relu" in partial_block and split_sublayer in ["conv1", "bn1"]:
            x = partial_block["relu"](x)
        if "conv2" in partial_block:
            x = partial_block["conv2"](x)
        if "bn2" in partial_block and split_sublayer != "bn2":
            x = partial_block["bn2"](x)

        # Add residual connection
        x = x + identity
        x = partial_block["final_relu"](x)
        return x

    def forward(self, x):
        split_layer_num = self.split_config["layer"]
        split_block = self.split_config["block"]
        split_sublayer = self.split_config["sublayer"]

        # Handle tuple input (from mid-block split)
        if isinstance(x, tuple):
            activation, identity = x
            x = self._complete_partial_block(activation, identity)

            # Continue remaining blocks in the same layer if any
            if split_block is not None and split_block < 1:  # There are more blocks
                layer_attr = f"layer{split_layer_num}"
                if hasattr(self, layer_attr):
                    layer = getattr(self, layer_attr)
                    for i, block in enumerate(layer):
                        if i > split_block:
                            x = block(x)

        # Continue with remaining layers
        if split_layer_num <= 1 and hasattr(self, "layer2"):
            x = self.layer2(x)
        if split_layer_num <= 2 and hasattr(self, "layer3"):
            x = self.layer3(x)
        if split_layer_num <= 3 and hasattr(self, "layer4"):
            x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
