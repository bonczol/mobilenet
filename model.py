import torch.nn as nn
import math




class Bottleneck(nn.Module):
    def __init__(self, in_, out, stride, expand_ratio):
        super(Bottleneck, self).__init__()
        self.stride = stride
        hidden = int(in_ * expand_ratio)

        self.model = nn.Sequential(
            # expanding
            nn.Conv2d(in_, hidden, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(True),
            # depth-wise conv
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=stride, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(True),
            # compressing
            nn.Conv2d(hidden, out, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out)
        )

    def forward(self, x):
        # Residual connection
        return x + self.model(x)


class MobileNet:
    def __init__(self, n_class=1000, input_size=224, width_multiplier=1.):
        super(MobileNet, self).__init__()
        self.width_multiplier = width_multiplier
        in_ = 32

        self.model = nn.Sequential(
            # conv2d
            nn.Conv2d(3, in_, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_),
            nn.ReLU6(True),
            # bottlenecks
            Bottleneck(in_, self.out(16), 1, 1),
        )

    def forward(self, x):
        return self.model(x)

    def out(self, c):
        o = c * self.width_multiplier
        round_to = 8
        return int(math.ceil(o / round_to) * round_to)
