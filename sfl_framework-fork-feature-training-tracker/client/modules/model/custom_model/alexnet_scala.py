import torch
import torch.nn as nn


class AlexNetSCALA(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNetSCALA, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
        )  # 3x28x28 -> 32x28x28
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 32x28x28 -> 32x14x14

        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )  # 32x14x14 -> 64x14x14
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64x14x14 -> 64x7x7

        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        )  # 64x7x7 -> 128x7x7
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
        )  # 128x7x7 -> 256x7x7
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
        )  # 256x7x7 -> 256x7x7
        self.relu5 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)  # 256x7x7 -> 256x3x3

        self.flatten = nn.Flatten()  # 256x3x3 -> 256*3*3
        self.fc6 = nn.Linear(256 * 3 * 3, 1024)
        self.relu6 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=0.5)

        self.fc7 = nn.Linear(1024, 512)
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=0.5)

        self.fc8 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)

        x = self.relu2(self.conv2(x))
        x = self.pool2(x)

        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.relu5(self.conv5(x))
        x = self.pool5(x)

        x = self.flatten(x)  # (N, 256*3*3)
        x = self.dropout1(self.relu6(self.fc6(x)))
        x = self.dropout2(self.relu7(self.fc7(x)))
        x = self.fc8(x)

        return x
