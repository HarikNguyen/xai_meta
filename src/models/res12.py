import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class Res12Block(nn.Module):
    """One residual stage of ResNet-12: 3x(conv3x3 + BN) on the main path,
    a 1x1 conv+BN projection shortcut, then LeakyReLU + 2x2 maxpool.
    """

    def __init__(self, device, indim, outdim, pool=True):
        super().__init__()
        self.device = device

        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(outdim, track_running_stats=False)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(outdim, track_running_stats=False)
        self.conv3 = nn.Conv2d(outdim, outdim, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(outdim, track_running_stats=False)

        self.shortcut_conv = nn.Conv2d(indim, outdim, kernel_size=1, stride=1, padding=0)
        self.shortcut_bn = nn.BatchNorm2d(outdim, track_running_stats=False)

        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool = pool

    def forward(self, x, weights=None):
        if weights is None:
            residual = self.shortcut_bn(self.shortcut_conv(x))

            out = self.relu(self.bn1(self.conv1(x)))
            out = self.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
        else:
            residual = F.conv2d(x, weights[12], weights[13], padding=0)
            residual = F.batch_norm(
                residual, None, None, weights[14], weights[15],
                training=True, eps=self.shortcut_bn.eps,
            )

            out = F.conv2d(x, weights[0], weights[1], padding=1)
            out = F.batch_norm(
                out, None, None, weights[2], weights[3],
                training=True, eps=self.bn1.eps,
            )
            out = self.relu(out)

            out = F.conv2d(out, weights[4], weights[5], padding=1)
            out = F.batch_norm(
                out, None, None, weights[6], weights[7],
                training=True, eps=self.bn2.eps,
            )
            out = self.relu(out)

            out = F.conv2d(out, weights[8], weights[9], padding=1)
            out = F.batch_norm(
                out, None, None, weights[10], weights[11],
                training=True, eps=self.bn3.eps,
            )

        out = self.relu(out + residual)
        if self.pool:
            out = self.maxpool(out)
        return out


class Res12(nn.Module):
    """ResNet-12 backbone (4 residual blocks, channels 64-160-320-640) commonly
    used as a few-shot learning embedding (TADAM/MetaOptNet-style).
    """

    def __init__(self, device, criterion, train_classes, channels=(64, 160, 320, 640)):
        super().__init__()
        self.device = device
        self.train_classes = train_classes
        self.criterion = criterion
        c1, c2, c3, c4 = channels
        self.model = nn.ModuleDict(
            {
                "features": nn.Sequential(
                    OrderedDict(
                        [
                            ("res_block1", Res12Block(device=device, indim=3, outdim=c1)),
                            ("res_block2", Res12Block(device=device, indim=c1, outdim=c2)),
                            ("res_block3", Res12Block(device=device, indim=c2, outdim=c3)),
                            ("res_block4", Res12Block(device=device, indim=c3, outdim=c4)),
                        ]
                    )
                ),
                "avgpool": nn.AdaptiveAvgPool2d(1),
                "flatten": nn.Flatten(),
                "out": nn.Linear(in_features=c4, out_features=self.train_classes).to(device),
            }
        )

    def forward(self, x, weights=None, only_features=False):
        # Normal forward
        if weights is None:
            features = self.model.features(x)
            if only_features:
                return features
            pooled = self.model.flatten(self.model.avgpool(features))
            out = self.model.out(pooled)
            return out

        # Functional forward (meta-learning), 16 params per residual block
        x = self.model.features.res_block1(x, weights[0:16])
        x = self.model.features.res_block2(x, weights[16:32])
        x = self.model.features.res_block3(x, weights[32:48])
        x = self.model.features.res_block4(x, weights[48:64])

        # Return features (A)
        if only_features:
            return x

        pooled = self.model.flatten(self.model.avgpool(x))
        out = F.linear(pooled, weights[64], weights[65])
        return out

    def forward_features(self, features, weights):
        pooled = self.model.flatten(self.model.avgpool(features))
        out = F.linear(pooled, weights[64], weights[65])
        return out
