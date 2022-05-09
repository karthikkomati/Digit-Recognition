# karthik komati
# this file is for an extension task

import sys

import torchvision.models as models

import torch
import matplotlib.pyplot as plt

import torchvision.models

import cv2 as cv

#main function to analyze and show effect of the filters in a pre trained network
def main(argv):
    batch_size_train = 64
    batch_size_test = 1000

    net = models.resnet18(pretrained=True)

    # print(net)

    fig = plt.figure()
    for i in range(64):
        plt.subplot(8, 8, i + 1)
        plt.tight_layout()
        plt.imshow(net.conv1.weight[i][0].detach(), interpolation='none')
        plt.title("Filter: {}".format(i + (64 * 0)))
        plt.xticks([])
        plt.yticks([])
    # fig
    fig = plt.figure()
    for i in range(64):
        plt.subplot(8, 8, i + 1)
        plt.tight_layout()
        plt.imshow(net.conv1.weight[i][1].detach(), interpolation='none')
        plt.title("Filter: {}".format(i + (64 * 1)))
        plt.xticks([])
        plt.yticks([])

    fig = plt.figure()
    for i in range(64):
        plt.subplot(8, 8, i + 1)
        plt.tight_layout()
        plt.imshow(net.conv1.weight[i][2].detach(), interpolation='none')
        plt.title("Filter: {}".format(i + (64 * 2)))
        plt.xticks([])
        plt.yticks([])

    plt.show()


    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/files/', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/files/', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_test, shuffle=True)

    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    print(type(example_data[0][0].detach().numpy()))
    with torch.no_grad():
        fig = plt.figure()
        for i in range(64):
            o = cv.filter2D(example_data[0][0].detach().numpy(), -1, net.conv1.weight[i][0].detach().numpy())
            plt.subplot(8, 8, i + 1)
            plt.tight_layout()
            plt.imshow(o, cmap='gray', interpolation='none')
            plt.title("Filter: {}".format(i))
            plt.xticks([])
            plt.yticks([])

        fig = plt.figure()
        for i in range(64):
            o = cv.filter2D(example_data[0][0].detach().numpy(), -1, net.conv1.weight[i][1].detach().numpy())
            plt.subplot(8, 8, i + 1)
            plt.tight_layout()
            plt.imshow(o, cmap='gray', interpolation='none')
            plt.title("Filter: {}".format(i + (64 * 1)))
            plt.xticks([])
            plt.yticks([])

        fig = plt.figure()
        for i in range(64):
            o = cv.filter2D(example_data[0][0].detach().numpy(), -1, net.conv1.weight[i][2].detach().numpy())
            plt.subplot(8, 8, i + 1)
            plt.tight_layout()
            plt.imshow(o, cmap='gray', interpolation='none')
            plt.title("Filter: {}".format(i + (64 * 2)))
            plt.xticks([])
            plt.yticks([])
        # fig
        plt.show()

    return


if __name__ == "__main__":
    main(sys.argv)
