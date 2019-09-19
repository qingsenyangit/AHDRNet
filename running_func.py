import os
import random
import numpy as np
import torch
import h5py

import torch.nn as nn
from torch.nn import init
import torchvision as tv
import torch.utils.data as data
import torch.nn.functional as F
from torch.autograd import Variable


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal(m.weight.data)


def testing_fun(model, test_loaders, args):
    model.eval()
    test_loss = 0
    num = 0
    for data, target in test_loaders:
        Test_Data_name = test_loaders.dataset.list_txt[num].split('.h5')[0].split('/')[-1]
        if args.use_cuda:
            data, target = data.cuda(), target.cuda()

        data1 = torch.cat((data[:, 0:3, :], data[:, 9:12, :]), dim=1)
        data2 = torch.cat((data[:, 3:6, :], data[:, 12:15, :]), dim=1)
        data3 = torch.cat((data[:, 6:9, :], data[:, 15:18, :]), dim=1)
        data1 = Variable(data1, volatile=True)
        data2 = Variable(data2, volatile=True)
        data3 = Variable(data3, volatile=True)
        target = Variable(target, volatile=True)
        output = model(data1, data2, data3)

        # save the result to .H5 files
        hdrfile = h5py.File(args.result_dir + Test_Data_name + '_hdr.h5', 'w')
        img = output[0, :, :, :]
        img = tv.utils.make_grid(img.data.cpu()).numpy()
        hdrfile.create_dataset('data', data=img)
        hdrfile.close()

        hdr = torch.log(1 + 5000 * output.cpu()) / torch.log(
            Variable(torch.from_numpy(np.array([1 + 5000])).float()))
        target = torch.log(1 + 5000 * target).cpu() / torch.log(
            Variable(torch.from_numpy(np.array([1 + 5000])).float()))

        test_loss += F.mse_loss(hdr, target)
        num = num + 1

    test_loss = test_loss / len(test_loaders.dataset)
    print('\n Test set: Average Loss: {:.4f}'.format(test_loss.data[0]))

    return test_loss


class testimage_dataloader(data.Dataset):
    def __init__(self, list_dir):
        f = open(list_dir)
        self.list_txt = f.readlines()
        self.length = len(self.list_txt)

    def __getitem__(self, index):
        sample_path = self.list_txt[index][:-1]
        if os.path.exists(sample_path):
            f = h5py.File(sample_path, 'r')
            data = f['IN'][:]
            label = f['GT'][:]
            f.close()
        # print(sample_path)
        return torch.from_numpy(data).float(), torch.from_numpy(label).float()

    def __len__(self):
        return self.length

    def random_number(self, num):
        return random.randint(1, num)