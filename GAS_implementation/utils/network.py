import copy
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, TypedDict, Tuple
import torchvision.models as tv_models


def disable_inplace(module: nn.Module) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, nn.ReLU(inplace=False))
        elif isinstance(child, nn.ReLU6):
            setattr(module, name, nn.ReLU6(inplace=False))
        else:
            disable_inplace(child)


class AlexNetSplitClient(nn.Module):
    def __init__(self, conv_layers):
        super().__init__()
        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x):
        return self.conv(x)


class AlexNetSplitServer(nn.Module):
    def __init__(self, conv_layers, fc_layers):
        super().__init__()
        self.conv = nn.Sequential(*conv_layers) if conv_layers else nn.Identity()
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def _resolve_alexnet_split_index(split_layer, default_index, total_layers):
    if not split_layer:
        return default_index

    normalized = str(split_layer).strip().lower()
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


def _build_alexnet_split_models(full_model, split_index):
    conv_layers = list(full_model.conv.children())
    if not (0 <= split_index < len(conv_layers)):
        raise ValueError(
            f"AlexNet split index must be within [0, {len(conv_layers) - 1}]"
        )

    client_layers = [copy.deepcopy(layer) for layer in conv_layers[: split_index + 1]]
    server_layers = [copy.deepcopy(layer) for layer in conv_layers[split_index + 1 :]]
    fc_layers = [copy.deepcopy(layer) for layer in full_model.fc.children()]

    return AlexNetSplitClient(client_layers), AlexNetSplitServer(
        server_layers, fc_layers
    )


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
        split_alexnet: AlexNet split point ('default', 'light', or 'conv.N')
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
                base_model = AlexNetCifar(num_classes=num_classes)
                default_index = 2 if split_alexnet == "light" else 5
                split_index = _resolve_alexnet_split_index(
                    split_layer, default_index, len(list(base_model.conv.children()))
                )
                user_model, server_model = _build_alexnet_split_models(
                    base_model, split_index
                )
                if twoLogit:
                    server_local_model = _build_alexnet_split_models(
                        AlexNetCifar(num_classes=num_classes), split_index
                    )[1]
        elif mnist or fmnist:
            base_model = AlexNet()
            default_index = 2 if split_alexnet == "light" else 5
            split_index = _resolve_alexnet_split_index(
                split_layer, default_index, len(list(base_model.conv.children()))
            )
            user_model, server_model = _build_alexnet_split_models(
                base_model, split_index
            )
            if twoLogit:
                server_local_model = _build_alexnet_split_models(
                    AlexNet(), split_index
                )[1]
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
        self.maxpool = nn.Identity()

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

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


class ResNet18ImageStyle(nn.Module):
    def __init__(self, num_classes=10):
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

    def _complete_partial_block(self, out, identity, stop_at):
        if self.partial_block is None:
            raise ValueError("Partial block is not initialized")

        partial_block = self.partial_block
        if "bn1" in partial_block:
            out = partial_block["bn1"](out)
        if "relu" in partial_block and stop_at in ["conv1", "bn1"]:
            out = partial_block["relu"](out)
        if "conv2" in partial_block:
            out = partial_block["conv2"](out)
        if "bn2" in partial_block and stop_at != "bn2":
            out = partial_block["bn2"](out)

        out += identity
        out = partial_block["final_relu"](out)
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


# =============================================================================
# Torchvision Initialization Helper
# =============================================================================


def load_torchvision_resnet18_init(
    client_model: nn.Module,
    server_model: nn.Module,
    split_layer: str = "layer2",
    image_style: bool = True,
) -> Tuple[nn.Module, nn.Module]:
    """
    Load torchvision ResNet18 state_dict into split client/server models.

    This ensures GAS/MultiSFL models start with the same Kaiming normal
    initialization as SFL's torchvision-based ResNet18.

    Args:
        client_model: Client-side model (e.g., ResNet18ImageStyleFlexibleClient)
        server_model: Server-side model (e.g., ResNet18ImageStyleFlexibleServer)
        split_layer: Split point string (e.g., 'layer2', 'layer2.0.bn1')
        image_style: If True, uses ImageNet-style ResNet (7x7 conv, maxpool).
                     If False, uses CIFAR-style (3x3 conv, no maxpool).

    Returns:
        Tuple of (client_model, server_model) with loaded weights.

    Note:
        - Only works for ResNet18ImageStyle* models (image_style=True)
        - CIFAR-style models have different architecture and cannot use
          torchvision weights directly
    """
    if not image_style:
        print(
            "[Warning] Torchvision init only supports image_style=True. "
            "CIFAR-style models have different architecture."
        )
        return client_model, server_model

    # Load torchvision ResNet18 with random init (no pretrained weights)
    tv_resnet = tv_models.resnet18(weights=None)
    tv_state = tv_resnet.state_dict()

    # Parse split configuration
    split_config = parse_split_layer(split_layer)
    split_layer_num = split_config["layer"]
    split_block = split_config["block"]
    split_sublayer = split_config["sublayer"]

    # Define which layers go to client vs server
    # Client gets: conv1, bn1, layer1, and layers up to split point
    # Server gets: layers after split point, avgpool, fc

    client_keys = set()
    server_keys = set()

    # conv1, bn1 always go to client
    for key in tv_state.keys():
        if key.startswith("conv1") or key.startswith("bn1"):
            client_keys.add(key)

    # Determine layer distribution based on split point
    for layer_num in range(1, 5):  # layer1, layer2, layer3, layer4
        layer_prefix = f"layer{layer_num}"
        layer_keys = [k for k in tv_state.keys() if k.startswith(layer_prefix)]

        if layer_num < split_layer_num:
            # Entire layer goes to client
            client_keys.update(layer_keys)
        elif layer_num > split_layer_num:
            # Entire layer goes to server
            server_keys.update(layer_keys)
        else:
            # Split happens within this layer
            if split_block is None:
                # Split at end of layer → entire layer to client
                client_keys.update(layer_keys)
            else:
                # Split within a specific block
                for key in layer_keys:
                    # Parse block number from key like "layer2.0.conv1.weight"
                    parts = key.split(".")
                    if len(parts) >= 2:
                        try:
                            block_num = int(parts[1])
                        except ValueError:
                            # Handle downsample keys like "layer2.0.downsample.0.weight"
                            block_num = int(parts[1]) if parts[1].isdigit() else 0

                        if block_num < split_block:
                            client_keys.add(key)
                        elif block_num > split_block:
                            server_keys.add(key)
                        else:
                            # Within the split block
                            if split_sublayer is None:
                                # Full block to client
                                client_keys.add(key)
                            else:
                                # Need to determine if sublayer is before or after split
                                sublayer_order = ["conv1", "bn1", "conv2", "bn2"]
                                key_sublayer = parts[2] if len(parts) > 2 else ""

                                # Downsample goes to client (computed at start of block)
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
                                    # Unknown sublayer, default to client
                                    client_keys.add(key)
                    else:
                        client_keys.add(key)

    # fc always goes to server
    for key in tv_state.keys():
        if key.startswith("fc"):
            server_keys.add(key)

    # Load client weights
    client_state = client_model.state_dict()
    loaded_client = 0
    for key in client_keys:
        if key in client_state and key in tv_state:
            if client_state[key].shape == tv_state[key].shape:
                client_state[key] = tv_state[key]
                loaded_client += 1
    client_model.load_state_dict(client_state)

    # Load server weights
    # Server model may have different key names for partial blocks
    server_state = server_model.state_dict()
    loaded_server = 0

    for key in server_keys:
        if key in server_state and key in tv_state:
            if server_state[key].shape == tv_state[key].shape:
                server_state[key] = tv_state[key]
                loaded_server += 1

    # Handle partial block weights mapping
    # When split is mid-block, server's partial_block needs mapping
    if (
        split_sublayer is not None
        and hasattr(server_model, "partial_block")
        and server_model.partial_block is not None
    ):
        # Map remaining sublayers from torchvision to server's partial_block
        block_prefix = f"layer{split_layer_num}.{split_block}"
        sublayer_map = {
            "bn1": f"{block_prefix}.bn1",
            "conv2": f"{block_prefix}.conv2",
            "bn2": f"{block_prefix}.bn2",
        }

        for pb_key, tv_key_prefix in sublayer_map.items():
            if pb_key in server_model.partial_block:
                # Find matching torchvision keys
                for tv_key in tv_state.keys():
                    if tv_key.startswith(tv_key_prefix):
                        # Map to partial_block key
                        param_type = tv_key.split(".")[
                            -1
                        ]  # weight, bias, running_mean, etc.
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
