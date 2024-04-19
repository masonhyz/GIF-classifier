import torch.nn as nn

class FrameClassifier(nn.Module):
    def __init__(self, num_classes, dropout_prob=0.5):
        super(FrameClassifier, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(5, 5), padding=(2, 2))
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(dropout_prob)

        self.fc1 = nn.Linear(16 * 8 * 8, 1024)  
        self.dropout4 = nn.Dropout(dropout_prob)
        self.out = nn.Linear(1024, num_classes)

    def forward(self, x):
        if x.dim() == 5 and x.size(1) == 1:
            x = x.squeeze(1)
        
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        x = x.flatten(1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout4(x)
        logits = self.out(x)

        return logits

