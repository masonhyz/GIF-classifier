import torch
import torch.nn as nn

from typing import Tuple

class Conv2_1d(nn.Module):
    """2+1d Convolutional Layer"""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 hidden_size: int | None,
                 kernel_size: Tuple[int, int],
                 stride: Tuple[int, int]=(1, 1),
                 padding: Tuple[int, int]=(0, 0)) -> None:
        """
        Note: kernel_sizes, strides, and paddings all take a tuple of 2 integers,
        where the first element is the value for the 2d conv layer, and the
        second element is for the 1d conv layer.
        """
        super(Conv2_1d, self).__init__()

        if hidden_size:
            self.hidden_size = hidden_size
        else:
            t = kernel_size[1]
            d = kernel_size[0]
            self.hidden_size = int(t * (d ** 2) * in_channels * out_channels /
                                   (d ** 2) * in_channels + t * out_channels)
            
        self.relu = nn.ReLU()
        self.conv2d = nn.Conv2d(in_channels,
                                self.hidden_size,
                                kernel_size[0],
                                stride[0],
                                padding[0])
        self.conv1d = nn.Conv1d(self.hidden_size,
                                out_channels,
                                kernel_size[1],
                                stride[1],
                                padding[1])
        
    def forward(self, x):
        # shape [b, t, c, h, w]
        # 2d conv layer
        b, t, c, h, w = x.size()
        x = x.view(b * t, c, h, w)
        x = self.conv2d(x)
        x = self.relu(x)

        # 1d conv layer
        c, h, w = x.size(1), x.size(2), x.size(3)
        x = x.view(b, t, c, h, w)
        x = x.permute(0, 3, 4, 2, 1) # shape [b, h, w, c, t]
        x = x.view(b * h * w, c, t)
        x = self.conv1d(x)
        output = self.relu(x)

        # Output
        c, t = output.size(1), output.size(2)
        output = output.view(b, h, w, c, t)
        output = output.permute(0, 4, 3, 1, 2) # shape [b, t, c, h, w]
        
        return output
    

class GIFClassifier(nn.Module):
    """The Gif Classifier Model with stacked 2+1d conv layers."""

    def __init__(self, num_classes) -> None:
        super(GIFClassifier, self).__init__()

        # general shape [b, t, c, h, w]
        # input shape [b, 50, 3, 500, 500]

        # shape [b, 24, 64, 164, 164]
        self.conv1 = Conv2_1d(3, 64, kernel_size=(9, 3), stride=(3, 2))
        # shape [b, 12, 128, 80, 80]
        self.conv2 = Conv2_1d(64, 128, kernel_size=(5, 2), stride=(2, 2))
        # shape [b, 6, 256, 38, 38]
        self.conv3 = Conv2_1d(128, 256, kernel_size=(5, 2), stride=(2, 2))
        # shape [b, 2, 256, 18, 18]
        self.conv4 = Conv2_1d(256, 256, kernel_size=(3, 3), stride=(2, 2))
        # shape [b, 1, 128, 8, 8]
        self.conv5 = Conv2_1d(256, 128, kernel_size=(3, 2), stride=(2, 1))

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(128 * 8 * 8, 8192)
        self.fc2 = nn.Linear(8192, 4096)
        self.fc3 = nn.Linear(4096, 2048)
        self.out = nn.Linear(2048, num_classes)

    def forward(self, x):
        # 2+1d conv layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        # fc layers
        b = x.size(1)
        x = x.view(b, -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        logits = self.out(x)

        return logits
