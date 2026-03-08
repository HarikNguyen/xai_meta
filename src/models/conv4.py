import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from .conv_block import ConvBlock


class Conv4(nn.Module):
    def __init__(self, device, criterion, train_classes):
        super().__init__()
        self.device = device
        self.train_classes = train_classes
        self.criterion = criterion
        self.model = nn.ModuleDict(
            {
                "features": nn.Sequential(
                    OrderedDict(
                        [
                            ("conv_block1", ConvBlock(device=device, indim=3)),
                            ("conv_block2", ConvBlock(device=device, indim=64)),
                            ("conv_block3", ConvBlock(device=device, indim=64)),
                            (
                                "conv_block4",
                                ConvBlock(device=device, indim=32, pools1=False),
                            ),
                            ("flatten", nn.Flatten()),
                        ]
                    )
                ),
                "out": nn.Linear(
                    in_features=32 * 5 * 5, out_features=self.train_classes
                ).to(device),
            }
        )

    def forward(self, x):
        features = self.model.features(x)
        out = self.model.out(features)
        return out

    def forward_weights(self, x, weights):
        x = self.model.features.conv_block1.forward_weights(x, weights[0:4])
        x = self.model.features.conv_block2.forward_weights(x, weights[4:8])
        x = self.model.features.conv_block3.forward_weights(x, weights[8:12])
        x = self.model.features.conv_block4.forward_weights(x, weights[12:16])
        x = self.model.features.flatten(x)
        x = F.linear(x, weights[16], weights[17])
        return x
