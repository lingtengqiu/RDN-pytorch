#!/usr/bin/python
# -*- coding: UTF-8 -*-

# **********************************************************
# * Author        : lingteng qiu
# * Email         : 1259738366@qq.com
# * Create time   : 2018-08-22 11:53
# * Last modified : 2018-08-22 11:53
# * Filename      : loss_function.py
# * Description   : the loss function for sr question 
# **********************************************************
import torch.nn as nn 
import torch
class L1_Loss(nn.Module):
    def __init__(self):
        super(L1_Loss,self).__init__()
        self.criterion = nn.L1Loss()
    def forward(self,x,label):
        return self.criterion(x,label)
