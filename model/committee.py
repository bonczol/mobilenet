import torch.nn.functional as F
import torch
import os
from model.mobile_net_v2 import MobileNet


class Committee:
    def __init__(self, nets):
        self.nets = nets

    def __call__(self, x):
        y_pred = torch.stack([F.softmax(net(x)) for net in self.nets[:3]], dim=2)
        return torch.sum(y_pred, 2)

    def save(self, directory, filename):
        for i, net in enumerate(self.nets):
            torch.save(net.state_dict(), f'{directory}/{filename}_{i}.pth')

    @classmethod
    def load(cls, directory, device, n_class, with_mul):
        nets = []
        for file in os.listdir(directory):
            net = MobileNet(n_class=n_class, width_multiplier=with_mul)
            net.to(device)
            net.load_state_dict(torch.load(f'{directory}/{file}'))
            net.eval()
            nets.append(net)

        return cls(nets)

