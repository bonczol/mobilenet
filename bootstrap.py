from train_helper import prepare_dataset, train, test
import torch
from model.mobile_net_v2 import MobileNet


def main():
    N = 5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    train_set, test_set = prepare_dataset()
    sampler = torch.utils.data.RandomSampler(train_set, replacement=True)

    nets = []
    for n in range(N):
        train_loader = torch.utils.data.DataLoader(train_set, sampler=sampler, batch_size=64, num_workers=2)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)

        net = MobileNet(n_class=100, width_multiplier=0.5)
        net.to(device)

        train(net, train_loader, device, f'./trained/cifar_{n}.pth', num_epochs=2)
        test(net, test_loader, device)
        nets.append(net.cpu())


if __name__ == '__main__':
    main()
