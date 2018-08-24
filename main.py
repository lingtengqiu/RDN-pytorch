#!/usr/bin/python
# -*- coding: UTF-8 -*-

# **********************************************************
# * Author        : lingteng qiu
# * Email         : 1259738366@qq.com
# * Create time   : 2018-08-16 16:40
# * Last modified : 2018-08-16 16:40
# * Filename      : main.py
# * Description   : this is a demo about main.py for my back project 
# **********************************************************
import torch
import torchvision
import numpy as np
import fire
import config
import models
import utils
import os
import cv2
import visdom
import data
import time
import random
random.seed(time.time())
import copy
opts = config.DefaultConfig()
def train(**kwargs):
    '''
    para : 
        opts:the para from your 
    return:
        the train model 
    '''
    opts.parse_kwargs(**kwargs)
    print "train begin!"
    viz = utils.Visualizer(opts.env)
    #model
    our_model = getattr(models,opts.model)(opts)
    our_model.load_state_dict(torch.load("./check_point/<class 'models.RDN.rdn'>_0823_17:59:06.path"))
    #step2 data_set
    data_size,data_loader = data.dataset.data_loader(opts)
    #step 3 criterion optimer
    #l1
    criterion = getattr(models,opts.loss_function)()
    print data_size
    optimer = torch.optim.Adam(our_model.parameters(),lr =opts.lr)
    sche = torch.optim.lr_scheduler.MultiStepLR(optimer,milestones = opts.stone,gamma = 0.5)
    #step 4 device 
    best_loss =1e11
    device = torch.device("cuda:0" if opts.use_gpu else "cpu")
    our_model.to(device)
    since = time.time()
    plot_line_win = None
    plot_img_win = None
    plot_label_win = None
    plot_test_img_win = None 
    plot_test_label_win = None
    #step 5 trainning 
    for epoch in range(10,opts.max_epoch):
        print ("*****************"*10)
        print ("epoch {}/{}".format(epoch,opts.max_epoch))
        our_model.train()
        epoch_loss = 0.0
        for _,datas in enumerate(data_loader['train'],1):
            inputs = datas['input'].to(device)
            labels = datas['label'].to(device)
            optimer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs  = our_model(inputs)
                loss = criterion(outputs,labels)
                loss.backward()
                optimer.step()
                show_outputs = copy.deepcopy(outputs.detach())
                show_outputs[show_outputs>1.0] = 1.0
                show_outputs[show_outputs<0.0] = 0.0
                plot_img_win = viz.images(show_outputs,win = plot_img_win,title = 'img_test')
                plot_label_win = viz.images(labels,win = plot_label_win,title = 'img_label')
            epoch_loss += loss.item()*inputs.shape[0]
            if _% 40 == 0:
                #draw loss_line
                print "iteration :{}".format(_)
                times = time.time() -since
                times = 1.0*times/60
                x = times
                train_loss = (epoch_loss*1.0/(opts.batch_size*_))
                plot_test_img_win,plot_test_label_win,test_loss = val(our_model,data_loader['val'],data_size['val'],device,criterion,plot_test_img_win,plot_test_label_win,viz)
                x = np.column_stack((np.asarray(x),np.asarray(x)))
                y = np.column_stack((np.asarray(train_loss),np.asarray(test_loss)))
                plot_line_win = viz.plot(x,y,plot_line_win,'Loss','train_loss','val_loss')
                if (test_loss < best_loss):
                    best_loss =test_loss
                    best_model = our_model.state_dict()
                    our_model.save()
        epoch_loss  = epoch_loss / data_size['train']
        plot_test_img_win,plot_test_label_win,test_loss = val(our_model,data_loader['val'],data_size['val'],device,criterion,plot_test_img_win,plot_test_label_win,viz)
        print("{} :train_loss{:.8f},val_loss{:.8f}".format('lossing',epoch_loss,test_loss))
        sche.step()
    print("best_loss for val{:.8f}".format(best_loss))
    torch.save(opts.load_model_path,best_model)


def val(model,dataloader,data_size,device,criterion,plot_test_img_win,plot_test_label_win,viz):
    '''
    validation our model is well?
    para:
        @ model     : the net
        @ dataloader: the validation data
    return:
        @ the loss
    '''
    model.eval()
    loss = 0.0
    for datas in dataloader:
        inputs = datas['input'].to(device)
        labels = datas['label'].to(device)
        with torch.set_grad_enabled(False):
            output = model(inputs)
            loss += criterion(output,labels)
            show_outputs = copy.deepcopy(output.detach())
            show_outputs[show_outputs>1.0] = 1.0
            show_outputs[show_outputs<0.0] = 0.0
        plot_test_img_win = viz.images(show_outputs,win=plot_test_img_win,title='test_img_test')
        plot_test_label_win = viz.images(labels,win=plot_test_label_win,title = 'test_img_label')
    loss = loss.item()/data_size
    model.train()
    return plot_test_img_win,plot_test_label_win,loss

def test():
    print("test begin")
    pass
def help():
    '''
    print help imformation
    in here we have more ideal
    train_data_root
    test_data_root
    load_model_path
    bath_size
    use_gpu
    num_worker
    print_freq
    max_epoch
    lr
    lr_decay
    weight_decay
    '''
    from inspect import getsource
    getsource = getsource(opts.__class__)
    print getsource
def process(**kwargs):
    scan_root = kwargs['root']
    save_root = kwargs['save_root']
    scale = kwargs['scale']

    scan_file = [os.path.join(scan_root,i) for i in os.listdir(scan_root)]
    for _,name in enumerate(scan_file):
        print "process {:.4f}".format(_*1.0/len(scan_file))
        print name
        img = cv2.imread(name)
        h,w = img.shape[0:2]
        h = h- np.mod(h,4)
        w = w - np.mod(w,4)
        img_hr = img[0:h,0:w,:]
        img_lr = cv2.resize(img_hr,(int(w/4),int(h/4)),cv2.INTER_CUBIC)
        ext = name.split('/')[-1]
        cv2.imwrite(os.path.join(scan_root,ext),img_hr)
        cv2.imwrite(os.path.join(save_root,ext),img_lr)
def model_test(**kwargs):
    '''
    use to test the net is right ?
    '''
    '''
    viz = utils.Visualizer()
    print viz.vis.env
    data_size,data_loader = data.dataset.data_loader(opts)
    for i in data_loader['train']:
        labels = i['label'].to(torch.device("cuda:0"))
        print labels.shape
        viz.images(labels,win = "label",title='loss')
    '''
    data_size,data_loaders,test_set = data.dataset.data_loader(opts)
    print len(test_set)
    for i in data_loaders['val']:
        print i['label'].shape
if __name__ == "__main__":
   fire.Fire() 
