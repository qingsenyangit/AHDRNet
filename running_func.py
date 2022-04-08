import os
import random
import numpy as np
import torch
import h5py
import time

import torch.nn as nn
from torch.nn import init
import torchvision as tv
import torch.utils.data as data
import torch.nn.functional as F
from torch.autograd import Variable

def mk_trained_dir_if_not(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)



def model_restore(model, trained_model_dir):
    model_list = glob.glob((trained_model_dir + "/*.pkl"))
    a = []
    for i in range(len(model_list)):
        index = int(model_list[i].split('model')[-1].split('.')[0])
        a.append(index)
    epoch = np.sort(a)[-1]
    model_path = trained_model_dir + 'trained_model{}.pkl'.format(epoch)
    model.load_state_dict(torch.load(model_path))
    return model, epoch


class data_loader(data.Dataset):
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
            crop_size = 256
            data, label = self.imageCrop(data, label, crop_size)
            data, label = self.image_Geometry_Aug(data, label)


        # print(sample_path)
        return torch.from_numpy(data).float(), torch.from_numpy(label).float()

    def __len__(self):
        return self.length

    def random_number(self, num):
        return random.randint(1, num)

    def imageCrop(self, data, label, crop_size):
        c, w, h = data.shape
        w_boder = w - crop_size  # sample point y
        h_boder = h - crop_size  # sample point x ...

        start_w = self.random_number(w_boder - 1)
        start_h = self.random_number(h_boder - 1)

        crop_data = data[:, start_w:start_w + crop_size, start_h:start_h + crop_size]
        crop_label = label[:, start_w:start_w + crop_size, start_h:start_h + crop_size]
        return crop_data, crop_label

    def image_Geometry_Aug(self, data, label):
        c, w, h = data.shape
        num = self.random_number(4)

        if num == 1:
            in_data = data
            in_label = label

        if num == 2:  # flip_left_right
            index = np.arange(w, 0, -1) - 1
            in_data = data[:, index, :]
            in_label = label[:, index, :]

        if num == 3:  # flip_up_down
            index = np.arange(h, 0, -1) - 1
            in_data = data[:, :, index]
            in_label = label[:, :, index]

        if num == 4:  # rotate 180
            index = np.arange(w, 0, -1) - 1
            in_data = data[:, index, :]
            in_label = label[:, index, :]
            index = np.arange(h, 0, -1) - 1
            in_data = in_data[:, :, index]
            in_label = in_label[:, :, index]

        return in_data, in_label

def get_lr(epoch, lr, max_epochs):
    if epoch <= max_epochs * 0.8:
        lr = lr
    else:
        lr = 0.1 * lr
    return lr

def train(epoch, model, train_loaders, optimizer, args):
    lr = get_lr(epoch, args.lr, args.epochs)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('lr: {}'.format(optimizer.param_groups[0]['lr']))
    model.train()
    num = 0
    trainloss = 0
    start = time.time()
    for batch_idx, (data, target) in enumerate(train_loaders):
        if args.use_cuda:
            data, target = data.cuda(), target.cuda()
        end = time.time()

############  used for End-to-End code
        data1 = torch.cat((data[:, 0:3, :, :], data[:, 9:12, :, :]), dim=1)
        data2 = torch.cat((data[:, 3:6, :, :], data[:, 12:15, :, :]), dim=1)
        data3 = torch.cat((data[:, 6:9, :, :], data[:, 15:18, :, :]), dim=1)

        data1 = Variable(data1)
        data2 = Variable(data2)
        data3 = Variable(data3)
        target = Variable(target)
        optimizer.zero_grad()
        output = model(data1, data2, data3)

#########  make the loss
        output = torch.log(1 + 5000 * output.cpu()) / torch.log(Variable(torch.from_numpy(np.array([1+5000])).float()))
        target = torch.log(1 + 5000 * target).cpu() / torch.log(Variable(torch.from_numpy(np.array([1+5000])).float()))

        loss = F.l1_loss(output, target)
        loss.backward()
        optimizer.step()
        trainloss = trainloss + loss
        if (batch_idx +1) % 4 == 0:
            trainloss = trainloss / 4
            print('train Epoch {} iteration: {} loss: {:.6f}'.format(epoch, batch_idx, trainloss.data))
            fname = args.trained_model_dir + 'lossTXT.txt'
            try:
                fobj = open(fname, 'a')

            except IOError:
                print('open error')
            else:
                fobj.write('train Epoch {} iteration: {} Loss: {:.6f}\n'.format(epoch, batch_idx, trainloss.data))
                fobj.close()
            trainloss = 0


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