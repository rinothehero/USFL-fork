import torch
import torch.nn as nn
import torchvision.models as models


class AlexNet1Ch(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet1Ch, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 96, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(96, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.dropout1 = nn.Dropout()
        self.fc1 = nn.Linear(32 * 12 * 12, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, num_classes)

        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)
        self.relu3 = nn.ReLU(inplace=False)
        self.relu4 = nn.ReLU(inplace=False)
        self.relu5 = nn.ReLU(inplace=False)
        self.relu6 = nn.ReLU(inplace=False)
        self.relu7 = nn.ReLU(inplace=False)

        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool1(x)
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.relu5(self.conv5(x))
        x = self.pool2(x)

        x = self.flatten(x)

        x = self.dropout1(x)
        x = self.relu6(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu7(self.fc2(x))
        x = self.fc3(x)

        return x
