# karthik komati
# this file is for task 3 of the project

import sys

import cv2
import os

import csv
from main import Net
import torch.nn.functional as F
import torch

import pandas as pd
import numpy as np

from torchvision import transforms

#A network model that is a subclass of the previous class
class Submodel(Net):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # override the forward method
    # computes a forward pass for the network
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # relu on max pooled results of conv1
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))  # relu on max pooled results of dropout of conv2

        x = x.view(-1, 320)

        return self.fc1(x)

#main function to create a digit embedding space
def main(argv):
    header = ['intensity']

    with open('intensities.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)


        writer.writerow(header)

        filenames = []
        for filename in os.listdir("greek-1"):
            img = cv2.imread(os.path.join("greek-1", filename))
            dim = (28, 28)
            rimg = cv2.resize(img, dim)

            filenames.append(filename)

            gray = cv2.cvtColor(rimg, cv2.COLOR_BGR2GRAY)
            inv = (255 - gray)
            flat = inv.flatten()

            if flat is not None:

                writer.writerow(flat)

    he = ['category']

    with open('categories.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(he)
        for fi in filenames:
            # print(fi)
            if "alpha" in fi:
                writer.writerow([1])

            elif "beta" in fi:
                writer.writerow([2])

            elif "gamma" in fi:
                writer.writerow([3])

    sub = Submodel()

    sub.eval()

    sub.load_state_dict(torch.load('model.pth'))

    data = pd.read_csv('intensities.csv', skiprows=[0], header=None)
    data_list = []
    for index, row in data.iterrows():
        r = row.to_numpy()
        r1 = np.reshape(r, (-1, 28))

        t = torch.from_numpy(r1)

        data_list.append(t)

    tt = torch.stack((data_list))

    a = tt[0]

    a.unsqueeze_(0)
    a = a.float()

    with torch.no_grad():
        out = sub(a)


    print(out.shape)

    out_list = []

    for d in data_list:
        d.unsqueeze_(0)
        d = d.float()
        out1 = sub(d)
        out_list.append(out1.detach().numpy())

    for o in out_list:
        print(np.linalg.norm(out_list[0] - o))

    print("_________________________________________________")

    for o in out_list:
        print(np.linalg.norm(out_list[9] - o))

    print("_________________________________________________")

    for o in out_list:
        print(np.linalg.norm(out_list[18] - o))
    print("_________________________________________________")
    print("_________________________________________________")
    print("_________________________________________________")

    imgs = []
    outs = []
    fnames = []
    for filename in os.listdir("myLetters"):
        # print(filename)
        fnames.append(filename)
        img = cv2.imread(os.path.join("myLetters", filename))
        dim = (28, 28)
        rimg = cv2.resize(img, dim)
        gray = cv2.cvtColor(rimg, cv2.COLOR_BGR2GRAY)
        imgs.append(gray)

        convert_tensor = transforms.ToTensor()

        t = convert_tensor(gray)

        out = sub(t)
        outs.append(out)

    print(fnames[0])
    for o in out_list:
        print(np.linalg.norm(outs[0].detach().numpy() - o))

    print("_________________________________________________")

    print(fnames[1])
    for o in out_list:
        print(np.linalg.norm(outs[1].detach().numpy() - o))

    print("_________________________________________________")

    print(fnames[2])
    for o in out_list:
        print(np.linalg.norm(outs[2].detach().numpy() - o))
    print("_________________________________________________")


if __name__ == "__main__":
    main(sys.argv)
