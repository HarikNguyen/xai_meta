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
        self.batchnorm = nn.BatchNorm2d(32)
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
        x = F.conv2d(x, weights[0], weights[1], padding=1)

        # Manual batch normalization followed by ReLU
        # running_mean = torch.zeros(32).to(self.device)
        # running_var = torch.ones(32).to(self.device)
        running_mean = self.batchnorm.running_mean
        running_var = self.batchnorm.running_var
        momentum = self.batchnorm.momentum
        
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                print(self.num_batches_tracked)
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        
        x = F.batch_norm(
            x,
            running_mean,
            running_var,
            weights[2],
            weights[3],
            momentum=exponential_average_factor,
            training=True,
        )
        x = self.relu(x)
        if self.pool:
            x = F.max_pool2d(F.relu(x), kernel_size=2, stride=2)
        return x
