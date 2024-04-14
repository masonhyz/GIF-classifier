import torch
import torch.nn as nn
from model_2_1dconv import Conv2_1d
import time
from train import train
from dataset import GIFDataset
from torch.utils.data import Subset, DataLoader

class GIFClassifier(nn.Module):
    """The Gif Classifier Model with stacked 2+1d conv layers."""

    def __init__(self, num_classes) -> None:
        super(GIFClassifier, self).__init__()
        self.num_classes = num_classes

        # general shape [b, t, c, h, w]
        # input shape [b, 50, 3, 500, 500]

        # shape [b, 2, 16, 2, 2]
        self.conv = Conv2_1d(3, 16, kernel_size=(3, 3), stride=(497, 47))

        self.relu = nn.ReLU()

        self.out = nn.Linear(2 * 16 * 2 * 2, num_classes)


    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        logits = self.out(x)
        return logits

dataset = GIFDataset()
train_dataset = Subset(dataset, range(2))
val_dataset = Subset(dataset, [3])
model = GIFClassifier(17)
train(model, train_dataset, val_dataset, batch_size=2, num_epochs=100, plot=False)