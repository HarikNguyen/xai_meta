import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, device, indim=3, pool=True, pools1=False):
        super(ConvBlock, self).__init__()
        self.device = device
        self.conv = nn.Conv2d(
            in_channels=indim, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.batchnorm = nn.BatchNorm2d(64, track_running_stats=False, momentum=1.0)
        self.relu = nn.ReLU()
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
        running_mean = self.batchnorm.running_mean
        running_var = self.batchnorm.running_var
        momentum = self.batchnorm.momentum
       
        x = F.batch_norm(
            x,
            running_mean,
            running_var,
            weights[2],
            weights[3],
            momentum=self.batchnorm.momentum,
            training=True,
        )
        
        # activation
        x = self.relu(x)

        # pooling
        if self.pool:
            x = F.max_pool2d(F.relu(x), kernel_size=2, stride=2)


        #return
        return x
