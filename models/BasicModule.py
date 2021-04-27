#!/usr/bin/python
# -*- coding: UTF-8 -*-

# **********************************************************
# * Author        : lingteng qiu
# * Email         : 
# * Create time   : 2018-08-16 10:45
# * Last modified : 2018-08-16 10:45
# * Filename      : BasicModule.py
# * Description   : this part give you a demo for your project 
# **********************************************************
import torch
import torch.optim as optim
import torch.nn as nn
import time

class basic(nn.Module):
    '''
    give you some method
    '''

    def __init__(self,opts = None):
        super(basic,self).__init__()
        self.model_name = str(type(self))
    def load(self,path):
        self.load_state_dict(torch.load(path))
    def save(self,name= None):
        if name == None:
            prefix = './check_point/'+self.model_name+"_"
            name = time.strftime(prefix +"%m%d_%H:%M:%S.path") 
        torch.save(self.state_dict(),name)
        return name
