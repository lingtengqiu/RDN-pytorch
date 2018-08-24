#!/usr/bin/python
# -*- coding: UTF-8 -*-

# **********************************************************
# * Author        : lingteng qiu
# * Email         : 1259738366@qq.com
# * Create time   : 2018-08-13 09:41
# * Last modified : 2018-08-13 10:02
# * Filename      : config.py
# * Description   : this part is config  
# **********************************************************
import argparse

def get_args():
    arg = argparse.ArgumentParser()
    arg.add_argument("-r",'--root',help='train root',default = './data/train')
    arg.add_argument("-t",'--test',help = 'test root',default = './data/test')
    arg.add_argument("-b","--batch",type = int,default = 128)
    arg.add_argument("-e","--epoch",type = int,default = 60)
    arg.add_argument("-l","--lr",type = float,default = 1e-3)
    arg.add_argument("-c","--config",type = bool,default = False)
    para = arg.parse_args()
    return para


class DefaultConfig(object):
    env = 'default' #visdom
    model = 'rdn'

    train_hr = './data/DIV2K_train_HR' 
    train_lr = './data/train_lr'
    valid_hr = './data/DIV2K_valid_HR' 
    valid_lr = './data/valid_lr'
    loss_function = 'L1_Loss'
    num_workers = 4

    
    load_model_path = './checkpoints/best_model.pth'
    use_gpu = True
    #according your number of cpu  
    print_freq = 20

    result = './result'
    max_epoch = 400
    stone =[200]
    optimer = 'Adam'
    train_range='1-800'
    validation_range ='801-805'
    lr = 1e-4
    lr_decay = 0.1
    iteration = 1000
    weight_decay = 1e-4 
    G0 =64
    batch_size = 16
    patch_size = 128
    scale = 4
    D = 20
    C = 6
    G = 32
    kernel_size = 3
    input_channels = 3
    out_channels = 3
    def __init__(self,args = None):
        if args == None:
            return 
        self.parse(args)
    def parse(self,args):
        '''
        re config the para 
        '''
        if args.config:
            for k,v in args.__get_kwargs():
                if not hasattr(self,k):
                    print("warning: opt has nor attribute {}".format(k))
                setattr(self,k,v)

        else:
            pass
        # print config
        print('user config:')
        for k,v in self.__class__.__dict__.iteritems():
            if not k.startswith("__") and k !='parse':
                print("{} : {}".format(k,getattr(self,k)))

    def parse_kwargs(self,**kwargs):
        for k,v in kwargs.iteritems():
            if not hasattr(self,k):
                print("warning opt has nor attribute{}".format(k))
            setattr(self,k,v)

        print('here user config:')
        for k,v in self.__class__.__dict__.iteritems():
            if not k.startswith("__") and k !='parse' and k!='parse_kwargs':
                print("{} : {}".format(k,getattr(self,k)))

def config_get():
    arg = get_args()
    config =DefaultConfig(arg)
    return config,arg
