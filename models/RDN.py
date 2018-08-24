#!/usr/bin/python
# -*- coding: UTF-8 -*-

# **********************************************************
# * Author        : lingteng qiu
# * Email         : 1259738366@qq.com
# * Create time   : 2018-08-22 09:54
# * Last modified : 2018-08-22 09:54
# * Filename      : RDN.py
# * Description   : this part for us is realize the RDN model from the paper
# * all detail you can see from the paper "Residual Dense Network for Image SR"
# **********************************************************
import BasicModule
import torch.nn as nn 
import torch
import time
class one_conv(nn.Module):
    def __init__(self,inchanels,growth_rate,kernel_size = 3):
        super(one_conv,self).__init__()
        self.conv = nn.Conv2d(inchanels,growth_rate,kernel_size=kernel_size,padding = kernel_size>>1,stride= 1)
        self.relu = nn.ReLU()
    def forward(self,x):
        output = self.relu(self.conv(x))
        return torch.cat((x,output),1)

class RDB(nn.Module):
    def __init__(self,G0,C,G,kernel_size = 3):
        super(RDB,self).__init__()
        convs = []
        for i in range(C):
            convs.append(one_conv(G0+i*G,G))
        self.conv = nn.Sequential(*convs)
        #local_feature_fusion
        self.LFF = nn.Conv2d(G0+C*G,G0,kernel_size = 1,padding = 0,stride =1)
    def forward(self,x):
        out = self.conv(x)
        lff = self.LFF(out)
        #local residual learning
        return lff + x

class rdn(BasicModule.basic):
    def __init__(self,opts):
        '''
        opts: the system para
        '''
        super(rdn,self).__init__()
        '''
        D: RDB number 20
        C: the number of conv layer in RDB 6
        G: the growth rate 32
        G0:local and global feature fusion layers 64filter
        '''
        self.D = opts.D
        self.C = opts.C
        self.G = opts.G
        self.G0 = opts.G0
        print "D:{},C:{},G:{},G0:{}".format(self.D,self.C,self.G,self.G0)
        kernel_size =opts.kernel_size
        input_channels = opts.input_channels
        #shallow feature extraction 
        self.SFE1 = nn.Conv2d(input_channels,self.G0,kernel_size=kernel_size,padding = kernel_size>>1,stride=  1)
        self.SFE2 = nn.Conv2d(self.G0,self.G0,kernel_size=kernel_size,padding = kernel_size>>1,stride =1)
        #RDB for paper we have D RDB block
        self.RDBS = nn.ModuleList()
        for d in range(self.D):
            self.RDBS.append(RDB(self.G0,self.C,self.G,kernel_size))
        #Global feature fusion
        self.GFF = nn.Sequential(
               nn.Conv2d(self.D*self.G0,self.G0,kernel_size = 1,padding = 0 ,stride= 1),
               nn.Conv2d(self.G0,self.G0,kernel_size,padding = kernel_size>>1,stride = 1),
        )
        #upsample net 
        self.up_net = nn.Sequential(
                nn.Conv2d(self.G0,self.G*4,kernel_size=kernel_size,padding = kernel_size>>1,stride = 1),
                nn.PixelShuffle(2),
                nn.Conv2d(self.G,self.G*4,kernel_size = kernel_size,padding =kernel_size>>1,stride = 1),
                nn.PixelShuffle(2),
                nn.Conv2d(self.G,opts.out_channels,kernel_size=kernel_size,padding = kernel_size>>1,stride = 1)
        )
        #init
        for para in self.modules():
            if isinstance(para,nn.Conv2d):
                nn.init.kaiming_normal_(para.weight)
                if para.bias is not None:
                    para.bias.data.zero_()

    def forward(self,x):
        #f-1
        f__1 = self.SFE1(x)
        out  = self.SFE2(f__1)
        RDB_outs = []
        for i in range(self.D):
            out = self.RDBS[i](out)
            RDB_outs.append(out)
        out = torch.cat(RDB_outs,1)
        out = self.GFF(out)
        out = f__1+out
        return self.up_net(out)

if __name__ == "__main__":
    RDN(None)
