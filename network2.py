# karthik komati
# this file is for task 2 of the project

import sys

import torch
import matplotlib.pyplot as plt

import torchvision

import cv2 as cv

from main import Net

import torch.nn.functional as F

#A network model that is a subclass of the previous class
class Submodel(Net):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # override the forward method
    # computes a forward pass for the network
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # relu on max pooled results of conv1
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))  # relu on max pooled results of dropout of conv2
        return x

#main function to run the program. Analyzes and shows effect of the filters in the network
def main(argv):
    batch_size_train = 64
    batch_size_test = 1000

    net = Net()
    print("---------------------------------------------------------------------------------------")
    net.load_state_dict(torch.load('model.pth'))
    print(net)

    print(net.conv1.weight)

    print(net.conv1.weight[1][0])

    print(net.conv1.weight[1][0].detach())

    # fig = plt.figure()
    for i in range(10):
        plt.subplot(3, 4, i + 1)
        plt.tight_layout()
        plt.imshow(net.conv1.weight[i][0].detach(), interpolation='none')
        plt.title("Filter: {}".format(i))
        plt.xticks([])
        plt.yticks([])
    # fig
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
        # fig = plt.figure()
        for i in range(10):
            o = cv.filter2D(example_data[0][0].detach().numpy(), -1, net.conv1.weight[i][0].detach().numpy())
            plt.subplot(3, 4, i + 1)
            plt.tight_layout()
            plt.imshow(o, cmap='gray', interpolation='none')
            plt.title("Filter: {}".format(i))
            plt.xticks([])
            plt.yticks([])
        # fig
        plt.show()

    sub = Submodel()
    sub.eval()
    sub.load_state_dict(torch.load('model.pth'))
    # print(type(example_data))

    a = example_data[0][0]
    plt.imshow(a,cmap='gray')
    plt.show()
    a.unsqueeze_(0)
    # print(example_data[0][0])

    # print(example_data[0].size())
    out = sub(a)

    for i in range(20):
        plt.subplot(4, 5, i + 1)
        plt.tight_layout()
        plt.imshow(out[i].detach(), cmap='gray', interpolation='none')
        plt.title("Channel: {}".format(i))
        plt.xticks([])
        plt.yticks([])
    # fig
    plt.show()

    return


if __name__ == "__main__":
    main(sys.argv)
