import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
from torch.autograd import Variable

        
class make_dilation_dense(nn.Module):
  def __init__(self, nChannels, growthRate, kernel_size=3):
    super(make_dilation_dense, self).__init__()
    self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size-1)//2+1, bias=True, dilation=2)
  def forward(self, x):
    out = F.relu(self.conv(x))
    out = torch.cat((x, out), 1)
    return out

# Dilation Residual dense block (DRDB)
class DRDB(nn.Module):
  def __init__(self, nChannels, nDenselayer, growthRate):
    super(DRDB, self).__init__()
    nChannels_ = nChannels
    modules = []
    for i in range(nDenselayer):    
        modules.append(make_dilation_dense(nChannels_, growthRate))
        nChannels_ += growthRate 
    self.dense_layers = nn.Sequential(*modules)    
    self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=True)
  def forward(self, x):
    out = self.dense_layers(x)
    out = self.conv_1x1(out)
    out = out + x
    return out

# Attention Guided HDR, AHDR-Net
class AHDR(nn.Module):
    def __init__(self, args):
        super(AHDR, self).__init__()
        nChannel = args.nChannel
        nDenselayer = args.nDenselayer
        nFeat = args.nFeat
        growthRate = args.growthRate
        self.args = args

        # F-1
        self.conv1 = nn.Conv2d(nChannel, nFeat, kernel_size=3, padding=1, bias=True)
        # F0
        self.conv2 = nn.Conv2d(nFeat*3, nFeat, kernel_size=3, padding=1, bias=True)
        self.att11 = nn.Conv2d(nFeat*2, nFeat*2, kernel_size=3, padding=1, bias=True)
        self.att12 = nn.Conv2d(nFeat*2, nFeat, kernel_size=3, padding=1, bias=True)
        self.attConv1 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        self.att31 = nn.Conv2d(nFeat*2, nFeat*2, kernel_size=3, padding=1, bias=True)
        self.att32 = nn.Conv2d(nFeat*2, nFeat, kernel_size=3, padding=1, bias=True)
        self.attConv3 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)

        # DRDBs 3
        self.RDB1 = DRDB(nFeat, nDenselayer, growthRate)
        self.RDB2 = DRDB(nFeat, nDenselayer, growthRate)

        self.RDB3 = DRDB(nFeat, nDenselayer, growthRate)
        # feature fusion (GFF)
        self.GFF_1x1 = nn.Conv2d(nFeat*3, nFeat, kernel_size=1, padding=0, bias=True)
        self.GFF_3x3 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # fusion
        self.conv_up = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)

        # conv 
        self.conv3 = nn.Conv2d(nFeat, 3, kernel_size=3, padding=1, bias=True)
        self.relu = nn.LeakyReLU()


    def forward(self, x1, x2, x3):

        F1_ = self.relu(self.conv1(x1))
        F2_ = self.relu(self.conv1(x2))
        F3_ = self.relu(self.conv1(x3))

        F1_i = torch.cat((F1_, F2_), 1)
        F1_A = self.relu(self.att11(F1_i))
        F1_A = self.att12(F1_A)
        F1_A = nn.functional.sigmoid(F1_A)
        F1_ = F1_ * F1_A


        F3_i = torch.cat((F3_, F2_), 1)
        F3_A = self.relu(self.att31(F3_i))
        F3_A = self.att32(F3_A)
        F3_A = nn.functional.sigmoid(F3_A)
        F3_ = F3_ * F3_A

        F_ = torch.cat((F1_, F2_, F3_), 1)

        F_0 = self.conv2(F_)
        F_1 = self.RDB1(F_0)
        F_2 = self.RDB2(F_1)
        F_3 = self.RDB3(F_2)
        FF = torch.cat((F_1, F_2, F_3), 1)
        FdLF = self.GFF_1x1(FF)         
        FGF = self.GFF_3x3(FdLF)
        FDF = FGF + F2_
        us = self.conv_up(FDF)

        output = self.conv3(us)
        output = nn.functional.sigmoid(output)

        return output
