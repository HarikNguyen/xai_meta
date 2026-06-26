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
                            ("conv_block2", ConvBlock(device=device, indim=32)),
                            ("conv_block3", ConvBlock(device=device, indim=32)),
                            ("conv_block4", ConvBlock(device=device, indim=32)),
                            ("flatten", nn.Flatten()),
                        ]
                    )
                ),
                "out": nn.Linear(
                    in_features=32 * 5 * 5, out_features=self.train_classes
                ).to(device),
            }
        )

    def forward(self, x, weights=None, only_features=False):
        # Normal forward
        if weights is None:
            features = self.model.features(x)
            out = self.model.out(features)
            return out

        # Functional forward (meta-learning)
        x = self.model.features.conv_block1(x, weights[0:4])
        x = self.model.features.conv_block2(x, weights[4:8])
        x = self.model.features.conv_block3(x, weights[8:12])
        x = self.model.features.conv_block4(x, weights[12:16])
        
        # Return features (A)
        if only_features:
            return x

        x = self.model.features.flatten(x)
        x = F.linear(x, weights[16], weights[17])
        return x

    def forward_features(features, weights):
        x = self.model.features.flatten(features)
        x = F.linear(x, weights[16], weights[17])
        return x

