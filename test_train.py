import torch
import torch.nn as nn
from model_2_1dconv import Conv2_1d
import time
from train import train
from dataset import GIFDataset
from torch.utils.data import Subset

num_classes = 10

class GIFClassifier(nn.Module):
    """The Gif Classifier Model with stacked 2+1d conv layers."""

    def __init__(self, num_classes) -> None:
        super(GIFClassifier, self).__init__()
        self.num_classes = num_classes

        # general shape [b, t, c, h, w]
        # input shape [b, 40, 3, 128, 128]

        # shape [b, 2, 8, 8, 8]
        self.conv = Conv2_1d(3, 8, kernel_size=(3, 3, 16, 20), stride=(1, 1, 16, 20), padding=(1, 1, 0, 0))
        self.relu = nn.ReLU()
        self.out = nn.Linear(2 * 8 * 8 * 8, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        logits = self.out(x)
        return logits

dataset = GIFDataset()
train_dataset = Subset(dataset, range(4))
val_dataset = Subset(dataset, [7])
model = GIFClassifier(num_classes)
train(model, train_dataset, val_dataset, batch_size=4, num_epochs=100, eval_every=1, plot=False)