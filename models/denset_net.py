#!/usr/bin/python
# -*- coding: UTF-8 -*-

# **********************************************************
# * Author        : lingteng qiu
# * Email         : 1259738366@qq.com
# * Create time   : 2018-08-11 19:07
# * Last modified : 2018-08-11 19:07
# * Filename      : model.py
# * Description   : this part we realize the super-version part about the imperial vision parper for iccv2017 about tong tong 
# **********************************************************
import torch
import torchvision
import torch.nn as nn
import numpy as np
#here we must careful ,every block only output 128 so we have some ideal about this part

def get_upsample_filter(size):
    """
    Make a 2D bilinear kernel suitable for upsampling
    """
    factor = (size + 1)//2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filter = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    return torch.from_numpy(filter).float()
class SingerLayer(nn.Module):
    def __init__(self,inChannels,growthRate):
        super(SingerLayer,self).__init__()
        self.relu = nn.ReLU(inplace = True)
        self.conv = nn.Conv2d(inChannels,growthRate,kernel_size = 3,padding =1,stride = 1)
    def forward(self,x):
        out = self.conv(x)
        out = self.relu(out)
        out = torch.cat((x,out),1)
        return out
class SingerBlock(nn.Module):
    def __init__(self,inChannels,growthRate,nDenselayer):
        super(SingerBlock,self).__init__()
        self.relu = nn.ReLU(inplace = True)
        self.conv = nn.Conv2d(inChannels,growthRate,kernel_size = 3,padding = 1,stride = 1)
        self.block = self.__make_dense(growthRate,growthRate,nDenselayer)
    def forward(self,x):
        output = self.relu(self.conv(x))
        output = self.block(output)
        return output
    def __make_dense(self,inChannels,growthRate,nDenselayer):
        layers = []
        for i in range(1,nDenselayer):
            layers.append(SingerLayer(inChannels,growthRate))
            inChannels += growthRate
        return  nn.Sequential(*layers) 
class Net(nn.Module):
    def __init__(self,inChannels = 1,growthRate = 16,nDenselayer = 8,nBlock = 8):
        super(Net,self).__init__()
        self.relu = nn.ReLU(inplace = True)
        #low level part
        self.low_level = nn.Conv2d(inChannels,128,kernel_size = 3,padding = 1,stride = 1)
        inChannels = 128
        #dense block have 8
        self.block1 = SingerBlock(inChannels,growthRate,nDenselayer)
        self.block2 = SingerBlock(inChannels,growthRate,nDenselayer)
        self.block3 = SingerBlock(inChannels,growthRate,nDenselayer)
        self.block4 = SingerBlock(inChannels,growthRate,nDenselayer)
        self.block5 = SingerBlock(inChannels,growthRate,nDenselayer)
        self.block6 = SingerBlock(inChannels,growthRate,nDenselayer)
        self.block7 = SingerBlock(inChannels,growthRate,nDenselayer)
        self.block8 = SingerBlock(inChannels,growthRate,nDenselayer)
        inChannels += growthRate*nDenselayer*nBlock

        #bottle
        self.bottle = nn.Conv2d(inChannels,256,kernel_size = 1,padding = 0,stride = 1)
        #decov
        self.decov1 = nn.ConvTranspose2d(256,256,kernel_size = 2,padding =0 ,stride = 2)
        self.decov2 = nn.ConvTranspose2d(256,256,kernel_size = 2,padding =0,stride = 2)
        #recov
        self.recon = nn.Conv2d(256,1,kernel_size = 3,padding = 1,stride =1)
        #__init__
        for para in self.modules():
            if isinstance(para,nn.Conv2d):
                nn.init.kaiming_normal_(para.weight)
                if para.bias is not None:
                    para.bias.data.zero_()
            if isinstance(para,nn.ConvTranspose2d):
                c1,c2,h,w = para.weight.data.size()
                weight = get_upsample_filter(h)
                para.weight.data=weight.view(1,1,h,w).repeat(c1,c2,1,1)
                if para.bias is not None:
                    para.bias.data.zero_()




    def forward(self,x):
        temp = self.low_level(x)

        #dense
        output1 = self.block1(temp) 
        output2 = self.block2(output1)
        output3 = self.block3(output2)
        output4 = self.block4(output3)
        output5 = self.block5(output4)
        output6 = self.block6(output5)
        output7 = self.block7(output6)
        output8 = self.block8(output7)
        out = torch.cat([output1,output2,output3,output4,output5,output6,output7,output8,temp],1)
        out = self.bottle(out)
        out = self.decov1(out)
        out = self.decov2(out)
        out = self.recon(out)
        return out
class L1_Charboonier_loss(nn.Module):
    def __init__(self):
        super(L1_Charboonier_loss,self).__init__()
        self.eps = 1e-6
    def forward(self,x,y):
        diff = torch.add(x,-y)
        error = torch.sqrt(diff*diff+self.eps)
        loss = torch.sum(error)
        return loss

if __name__ == "__main__":
    inputs = torch.randn(8,1,100,100)
    labels = torch.randn(8,1,100,100)
    diff = inputs-labels
    error = torch.sqrt(diff*diff+1e-6)
    print torch.sum(error)
