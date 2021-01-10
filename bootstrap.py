from train_helper import prepare_dataset, train, test
import torch
import argparse
import os
from model.mobile_net_v2 import MobileNet
from model.committee import Committee


N = 5
WIDTH_MUL = 0.5
N_CLASS = 100


def train_bs(train_set, device):
    sampler = torch.utils.data.RandomSampler(train_set, replacement=True)

    nets = []
    for n in range(N):
        train_loader = torch.utils.data.DataLoader(train_set, sampler=sampler, batch_size=64, num_workers=2)

        net = MobileNet(n_class=N_CLASS, width_multiplier=WIDTH_MUL)
        net.to(device)

        train(net, train_loader, device, f'./trained/cifar_{n}.pth', num_epochs=2)
        test(net, test_loader, device)
        nets.append(net.cpu())

    committee = Committee(nets)
    test(committee, test_loader, device)


def test_bs(test_set, device):
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)
    committee = Committee.load("trained/bs", device, N_CLASS, WIDTH_MUL)
    test(committee, test_loader, device)


def main():
    train_set, test_set = prepare_dataset()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    test_bs(test_set, device)


if __name__ == '__main__':
    main()
