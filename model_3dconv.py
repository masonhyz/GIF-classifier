
import torch.nn as nn

class GIFClassifier(nn.Module):
    def __init__(self, num_classes) -> None:
        super(GIFClassifier, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv3d(3, 16, kernel_size=(3, 5, 5), padding=(1, 2, 2))
        self.pool1 = nn.MaxPool3d(kernel_size=(4, 4, 4), stride=(4, 4, 4))
        self.conv2 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv3 = nn.Conv3d(32, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.fc1 = nn.Linear(4096, 2048)
        self.out = nn.Linear(2048, num_classes)

    def forward(self, x):
        # input shape [b, 40, 3, 128, 128]
        x = x.permute(0, 2, 1, 3, 4)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool3(x)
        
        x = x.flatten(1)
        x = self.fc1(x)
        x = self.relu(x)
        logits = self.out(x)

        return logits
    