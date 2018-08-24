#!/usr/bin/python
# -*- coding: UTF-8 -*-

# **********************************************************
# * Author        : lingteng qiu
# * Email         : 1259738366@qq.com
# * Create time   : 2018-08-12 15:30
# * Last modified : 2018-08-12 15:30
# * Filename      : demo.py
# * Description   : 
# **********************************************************
import torch
from torch import nn, optim   # nn 神经网络模块 optim优化函数模块
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms, datasets
from visdom import Visdom  # 可视化处理模块
import time
import numpy as np
# 可视化app
viz = Visdom()

# 超参数
BATCH_SIZE = 40
LR = 1e-3
EPOCH = 2
# 判断是否使用gpu
USE_GPU = True

if USE_GPU:
    gpu_status = torch.cuda.is_available()
else:
    gpu_status = False

# 数据引入
train_dataset = datasets.MNIST('./mnist', True, transforms.ToTensor(), download=False)
test_dataset = datasets.MNIST('./mnist', False, transforms.ToTensor())

train_loader = DataLoader(train_dataset, BATCH_SIZE, True)
# 为加快测试，把测试数据从10000缩小到2000
test_data = torch.unsqueeze(test_dataset.test_data, 1)[:1500]
test_label = test_dataset.test_labels[:1500]
# visdom可视化部分数据
viz.images(test_data[:100], nrow=10)
# 为防止可视化视窗重叠现象，停顿0.5秒
time.sleep(0.5)
if gpu_status:
    test_data = test_data.cuda()
test_data = Variable(test_data, volatile=True).float()
# 创建线图可视化窗口
line = viz.line(np.arange(10))

# 创建cnn神经网络
class CNN(nn.Module):
    def __init__(self, in_dim, n_class):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            # channel 为信息高度 padding为图片留白 kernel_size 扫描模块size（5x5）
            nn.Conv2d(in_channels=in_dim, out_channels=16,kernel_size=5,stride=1, padding=2),
            nn.ReLU(),
            # 平面缩减 28x28 >> 14*14
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            # 14x14 >> 7x7
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(32*7*7, 120),
            nn.Linear(120, n_class)
        )
    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
net = CNN(1,10)

if gpu_status :
    net = net.cuda()
    print("#"*26, "使用gpu", "#"*26)
else:
    print("#" * 26, "使用cpu", "#" * 26)
# loss、optimizer 函数设置
loss_f = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=LR)
# 起始时间设置
start_time = time.time()
# 可视化所需数据点
time_p, tr_acc, ts_acc, loss_p = [], [], [], []
# 创建可视化数据视窗
text = viz.text("<h1>convolution Nueral Network</h1>")
for epoch in range(20):
    # 由于分批次学习，输出loss为一批平均，需要累积or平均每个batch的loss，acc
    sum_loss, sum_acc, sum_step = 0., 0., 0.
    for i, (tx, ty) in enumerate(train_loader, 1):
        if gpu_status:
            tx, ty = tx.cuda(), ty.cuda()
        tx = Variable(tx)
        ty = Variable(ty)
        out = net(tx)
        loss = loss_f(out, ty)
        sum_loss += loss.data[0]*len(ty)
        pred_tr = torch.max(out,1)[1]
        sum_acc += sum(pred_tr==ty).data[0]
        sum_step += ty.size(0)
        # 学习反馈
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 每40个batch可视化一下数据
        if i % 40 == 0:
            if gpu_status:
                test_data = test_data.cuda()
            test_out = net(test_data)
            # 如果用gpu运行out数据为cuda格式需要.cpu（）转化为cpu数据 在进行比较
            pred_ts = torch.max(test_out, 1)[1].cpu().data.squeeze()
            acc = torch.sum(pred_ts==test_label).item()/float(test_label.size(0))
            print("epoch: [{}/{}] | Loss: {:.4f} | TR_acc: {:.4f} | TS_acc: {:.4f} | Time: {:.1f}".format(epoch+1, EPOCH, sum_loss.item()/(sum_step), sum_acc.item()/(sum_step), acc, time.time()-start_time))
            # 可视化部分
            time_p.append(time.time()-start_time)
            tr_acc.append(sum_acc.item()/sum_step)
            ts_acc.append(acc)
            loss_p.append(sum_loss.item()/sum_step)
            viz.line(X=np.column_stack((np.array(time_p), np.array(time_p), np.array(time_p))),
                     Y=np.column_stack((np.array(loss_p), np.array(tr_acc), np.array(ts_acc))),
                     win=line,
                     opts=dict(legend=["Loss", "TRAIN_acc", "TEST_acc"],
                     width=200,
                     height =800,
                     xlabel = 'time',webgl=True))
            # visdom text 支持html语句
            viz.text("<p style='color:red'>epoch:{}</p><br><p style='color:blue'>Loss:{:.4f}</p><br>"
                     "<p style='color:BlueViolet'>TRAIN_acc:{:.4f}</p><br><p style='color:orange'>TEST_acc:{:.4f}</p><br>"
                     "<p style='color:green'>Time:{:.2f}</p>".format(epoch, sum_loss.item()/sum_step, sum_acc.item()/sum_step, acc,
                                                                       time.time()-start_time),
                     win=text)
            sum_loss, sum_acc, sum_step = 0., 0., 0.
