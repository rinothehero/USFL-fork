import torch.nn as nn

class AlexNetMnist(nn.Module):
    """AlexNet for MNIST/FMNIST (1-channel, 28x28 images)
    
    Structure matches the reference project's AlexNetDown/Up split architecture.
    """
    def __init__(self, num_classes=10):
        super(AlexNetMnist, self).__init__()
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
            nn.Flatten(),
        )

        self.fc = nn.Sequential(
            nn.Linear(256 * 3 * 3, 1024),  # 28x28 → 14x14 → 7x7 → 3x3 after MaxPool(k=3,s=2)
            nn.Linear(1024, 512),
            nn.Linear(512, num_classes),
        )

    def forward(self, x, return_features=False):
        out = self.conv(x)
        features = out
        out = self.fc(features)
        if return_features:
            return out, features
        else:
            return out
