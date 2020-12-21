import torch.nn as nn


class Bottleneck(nn.Module):
    def __init__(self, in_, out, stride, expand_ratio):
        super(Bottleneck, self).__init__()
        self.res = in_ == out and stride == 1
        hidden = int(in_ * expand_ratio)

        if expand_ratio == 1:
            self.model = nn.Sequential(
                # depth-wise conv
                nn.Conv2d(hidden, hidden, kernel_size=3, stride=stride, padding=1, groups=hidden, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU6(True),
                # compressing
                nn.Conv2d(hidden, out, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out)
            )
        else:
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
        if self.res:
            return x + self.model(x)
        else:
            return self.model(x)
