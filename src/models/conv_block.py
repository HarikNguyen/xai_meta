import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, device, indim=3, pool=True, pools1=False):
        super(ConvBlock, self).__init__()
        self.device = device
        self.conv = nn.Conv2d(
            in_channels=indim, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.batchnorm = nn.BatchNorm2d(
            32,
            track_running_stats=False,
        )
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=2, stride=2 if not pools1 else 1, padding=0
        )
        self.pool = pool

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        if self.pool:
            x = self.maxpool(x)
        return x

    def forward_weights(self, x, weights):
        # conv2d
        x = F.conv2d(x, weights[0], weights[1], padding=1)

        # batchnrom
        x = F.batch_norm(
            x,
            None,
            None,
            weights[2],
            weights[3],
            training=True,
            eps=self.batchnorm.eps,
        )

        # activation
        x = F.relu(x, inplace=True)

        # pooling
        if self.pool:
            x = F.max_pool2d(x, kernel_size=2, stride=2)

        # return
        return x
