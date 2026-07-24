import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class BasicBlock(nn.Module):
    """Standard ResNet BasicBlock (He et al. 2015): 2x(conv3x3 + BN) on the
    main path, a 1x1 conv+BN projection shortcut, ReLU. Downsampling is done
    via stride=2 in the first conv (and the shortcut), not a separate pooling
    layer -- so only the first conv reads the full-resolution input; the
    second conv already operates on the halved spatial size.
    """

    def __init__(self, device, indim, outdim, stride):
        super().__init__()
        self.device = device

        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(outdim, track_running_stats=False)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(outdim, track_running_stats=False)

        self.shortcut_conv = nn.Conv2d(indim, outdim, kernel_size=1, stride=stride, padding=0)
        self.shortcut_bn = nn.BatchNorm2d(outdim, track_running_stats=False)

        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, weights=None):
        if weights is None:
            residual = self.shortcut_bn(self.shortcut_conv(x))

            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
        else:
            residual = F.conv2d(x, weights[8], weights[9], stride=self.stride, padding=0)
            residual = F.batch_norm(
                residual, None, None, weights[10], weights[11],
                training=True, eps=self.shortcut_bn.eps,
            )

            out = F.conv2d(x, weights[0], weights[1], stride=self.stride, padding=1)
            out = F.batch_norm(
                out, None, None, weights[2], weights[3],
                training=True, eps=self.bn1.eps,
            )
            out = self.relu(out)

            out = F.conv2d(out, weights[4], weights[5], stride=1, padding=1)
            out = F.batch_norm(
                out, None, None, weights[6], weights[7],
                training=True, eps=self.bn2.eps,
            )

        return self.relu(out + residual)


class ResNet10(nn.Module):
    """ResNet-10 backbone (4 BasicBlocks, channels 64-128-256-512), the
    shallowest member of the ResNet family used as a "deep" contrast to
    Conv4 in few-shot learning literature (Chen et al. 2019, "A Closer Look
    at Few-Shot Classification").
    """

    def __init__(self, device, criterion, train_classes, channels=(64, 128, 256, 512)):
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
                            ("block1", BasicBlock(device=device, indim=3, outdim=c1, stride=2)),
                            ("block2", BasicBlock(device=device, indim=c1, outdim=c2, stride=2)),
                            ("block3", BasicBlock(device=device, indim=c2, outdim=c3, stride=2)),
                            ("block4", BasicBlock(device=device, indim=c3, outdim=c4, stride=2)),
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

        # Functional forward (meta-learning), 12 params per block
        x = self.model.features.block1(x, weights[0:12])
        x = self.model.features.block2(x, weights[12:24])
        x = self.model.features.block3(x, weights[24:36])
        x = self.model.features.block4(x, weights[36:48])

        # Return features (A)
        if only_features:
            return x

        pooled = self.model.flatten(self.model.avgpool(x))
        out = F.linear(pooled, weights[48], weights[49])
        return out

    def forward_features(self, features, weights):
        pooled = self.model.flatten(self.model.avgpool(features))
        out = F.linear(pooled, weights[48], weights[49])
        return out
