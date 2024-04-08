
import torch.nn as nn

from preprocess import MAX_WIDTH, MAX_HEIGHT

class GIFClassifier(nn.Module):
    def __init__(self, num_classes) -> None:
        super(GIFClassifier, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=11, padding=5)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv3d(256, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=3, stride=2)

        self.fc1 = nn.Linear('TODO: SOME SIZE', 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, num_classes)

    def forward(self, x):
        b = x.shape[0]

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(b, -1)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        logits = self.fc3(x)

        return logits
    