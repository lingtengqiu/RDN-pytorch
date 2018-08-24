#!/usr/bin/python
# -*- coding: UTF-8 -*-

# **********************************************************
# * Author        : lingteng qiu
# * Email         : 1259738366@qq.com
# * Create time   : 2018-08-16 10:33
# * Last modified : 2018-08-16 10:33
# * Filename      : dataset.py
# * Description   : according to the paper for our every epoch we have 1000 iteration 
# * for every iteration we have 16 batch for every img ,we random crop the 32*32 
# * may be is easy but some time is very complex so we try our best to realize
# **********************************************************
import os
import cv2
from torch.utils.data import  Dataset
import torch
import numpy as np
from torchvision import transforms as T 
import random
import visdom
import torchvision
class ToTensor(object):
    def __call__(self,sample):
        inputs = sample['input']
        labels = sample['label']
        inputs = np.ascontiguousarray(np.transpose(inputs,(2,0,1)))
        labels = np.ascontiguousarray(np.transpose(labels,(2,0,1)))
        return {"input":torch.from_numpy(inputs).float()/255.0,
                "label":torch.from_numpy(labels).float()/255.0}
class Flippe(object):
    def __call__(self,sample):
        '''
        sample:
            @'inputs':32 32 lr_img
            @'labels':128*128 hr_img 
        '''
        is_hor  = random.random()>0.5
        inputs = sample['input']
        labels = sample['label']
        #whether hor flip
        if is_hor:
            inputs = inputs[:,::-1,:]
            labels = labels[:,::-1,:]
        return {"input":inputs,"label":labels}
class Rotation(object):
    def __call__(self,sample):
        is_rot = random.random()>0.5
        inputs = sample['input']
        labels = sample['label']
        if is_rot:
            inputs = np.transpose(inputs,(1,0,2))
            labels = np.transpose(labels,(1,0,2))
        return {"input":inputs,"label":labels}
class data_set(Dataset):
    def __init__(self,train_hr_root, train_lr_root,opts,train = True,scale = 4,transform =None):
        super(Dataset,self).__init__()
        self.batch_size = opts.batch_size
        self.hr_root = train_hr_root
        self.lr_root = train_lr_root
        self.transform = transform
        self.scale = scale
        self.epoch_itr = opts.iteration
        self.train = train
        self.patch_size = opts.patch_size
        if train:
            begin,end = opts.train_range.split('-')
        else:
            begin,end = opts.validation_range.split('-')
        self.begin = int(begin)
        self.end = int(end)
        print begin,end

        self.hr_file_name = [os.path.join(self.hr_root,i) for i in os.listdir(self.hr_root)]
        self.lr_file_name = [os.path.join(self.lr_root,i) for i in os.listdir(self.lr_root)]
        self.hr_file_name = sorted(self.hr_file_name,key = lambda p:int(p.split('/')[-1].split('.')[0]))
        self.lr_file_name = sorted(self.lr_file_name,key = lambda p:int(p.split('/')[-1].split('.')[0]))
        if train == False:
            self.hr_file_name = self.hr_file_name[0:self.end-self.begin+1]
            self.lr_file_name = self.lr_file_name[0:self.end-self.begin+1]

    def __len__(self):
        if self.train:
            return self.epoch_itr*self.batch_size
        else:
            return len(self.hr_file_name)
    def __getitem__(self,index):
        if self.train:
            index = index % len(self.hr_file_name)
        img_hr = cv2.imread(self.hr_file_name[index])
        img_lr = cv2.imread(self.lr_file_name[index])
        img_hr,img_lr = self.get_patch(img_hr,img_lr)
        sample = {'input':img_lr,"label":img_hr}
        if self.transform is not  None:
            sample = self.transform(sample)
        return sample

    def get_patch(self,img_hr,img_lr):
        '''
        when  train we random crop
        but not we don't change everything
        '''
        if self.train:
            ih,iw =  img_lr.shape[0:2]
            lr_patch = self.patch_size // self.scale
            #random choice crop
            lr_x = random.randrange(0,iw-lr_patch+1)
            lr_y = random.randrange(0,ih-lr_patch+1)
            hr_x = lr_x*self.scale
            hr_y = lr_y*self.scale
            return img_hr[hr_y:hr_y+self.patch_size,hr_x:hr_x+self.patch_size,:],img_lr[lr_y:lr_y+lr_patch,lr_x:lr_x+lr_patch]
        else:
            return img_hr,img_lr
def data_loader(opts):
    transforms = T.Compose([Flippe(),Rotation(),ToTensor()])
    train_set = data_set(opts.train_hr,opts.train_lr,opts,train = True,transform = transforms)
    test_set = data_set(opts.valid_hr,opts.valid_lr,opts,train=False,transform = ToTensor() )
    data_size = {"train":len(train_set),"val":len(test_set)}
    data_loader = {"train":torch.utils.data.DataLoader(train_set,opts.batch_size,shuffle = True,num_workers = opts.num_workers),
            "val":torch.utils.data.DataLoader(test_set,batch_size = 1,shuffle = False)
            }
    return data_size,data_loader
if __name__ =="__main__":
    pass

