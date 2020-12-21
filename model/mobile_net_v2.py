import torch.nn as nn
from .bottleneck import Bottleneck
from .helper import make_divisible


class MobileNet(nn.Module):
    def __init__(self, n_class=1000, width_multiplier=1.):
        super(MobileNet, self).__init__()
        self.n_class = n_class
        self.width_multiplier = width_multiplier
        self.round_to = 8

        self.bottleneck_cfg = [
            # t, c, n, s
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
        ]

        self.features = None
        self.classifier = None

        self._build()
        self._init_weights()

    def _build(self):
        in_channels = 3
        out_channels = 32
        out_channels = make_divisible(out_channels * self.width_multiplier, divisor=self.round_to)

        # First layer - conv2d
        self.features = [nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )]

        in_channels = out_channels

        # Bottleneck layers
        for exp_factor, c, n, stride in self.bottleneck_cfg:
            out_channels = make_divisible(c * self.width_multiplier, divisor=self.round_to)

            for i in range(n):
                stride = 1 if i > 0 else stride
                self.features.append(
                    Bottleneck(in_channels, out_channels, stride, exp_factor)
                )
                in_channels = out_channels

        # Last layers
        out_channels = 1280
        out_channels = make_divisible(out_channels * self.width_multiplier, divisor=self.round_to)
        self.features.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU6(inplace=True),
            )
        )

        self.features = nn.Sequential(*self.features)

        # Init classifier
        in_channels = out_channels
        self.classifier = nn.Linear(in_channels, self.n_class)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1)).reshape(x.shape[0], -1)
        return self.classifier(x)
