#!/usr/bin/python
# -*- coding: UTF-8 -*-

# **********************************************************
# * Author        : lingteng qiu
# * Email         : 
# * Create time   : 2018-08-16 10:57
# * Last modified : 2018-08-16 10:57
# * Filename      : visualizer.py
# * Description   : this part is a tool to visual our objects 
# **********************************************************
import visdom
import time
import numpy as np
import torch

class Visualizer(object):
    '''
    for us we only use the 
    text
    line
    images
    the other maybe not use 
    '''
    def __init__(self,env='default',**kwargs):
        self.vis = visdom.Visdom(env = env,**kwargs)
        self.loss_win = self.vis.line(np.arange(10))
        self.accurate_win = self.vis.line(np.arange(10))
        self.text_win = self.vis.text("<h1>convolution Nueral NetWork<h1>")
    def reinit(self,env,**kwargs):
        self.vis =visdom.Visdom(env = env,**kwargs)
        return self
    def plot(self,x,y,wins = None,title = None,*args,**kwargs):
        '''
        para:
            x:the axis of x-axis
            y: the axis of y-axis,some time have something changes
        return:
            the window name for next
        '''
        args = list(args)
        print args
        if wins == None:
            wins = self.loss_win
            win = self.vis.line(
                X= x,
                Y=y,
                win= wins,
                opts = dict(legend=args,title = title,xlabel = 'time',webgl = True),
                **kwargs
                )
        else:
            win = self.vis.line(
                X= x,
                Y=y,
                win= wins,
                opts = dict(legend=args,title = title,xlabel = 'time',webgl = True),
                update ='append',
                **kwargs
           ) 

        return win


    def text(self,win_name= None,**kwargs):
        '''
        kwargs: epoch,
                loss,
                trainacc,
                test_acc
                time,
        '''
        # visdom support html 
        #print kwargs['epoch']
        if win_name == None:
            wins=  self.text_win


	win = self.vis.text("<p style='color:red'>epoch:{}</p><br><p style='color:blue'>Loss:{:.4f}</p><br>"
                     "<p style='color:BlueViolet'>TRAIN_acc:{:.4f}</p><br><p style='color:orange'>TEST_acc:{:.4f}</p><br>"
                     "<p style='color:green'>Time:{:.2f}</p>".format(kwargs['epoch'],kwargs['loss'],kwargs['train_acc'],kwargs['test_acc'],kwargs['time']),win=wins)
        return win
    def images(self,img_,win = None,title = None):
        return self.vis.images(img_,win = win,opts=dict(title=title))
if __name__ == "__main__":
    v = Visualizer()
    print v.vis.line(np.arange(100)) 

