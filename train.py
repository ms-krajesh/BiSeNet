#!/usr/bin/python
# -*- encoding: utf-8 -*-


from logger import *
from model import BiSeNet
from  cityscapes import CityScapes

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import os
import logging
import time


respth = './res'
if not osp.exists(respth): os.makedirs(respth)

class Optimizer(object):
    def __init__(self, params, lr0, momentum, wd, max_iter, power):
        self.lr0 = lr0
        self.lr = self.lr0
        self.max_iter = float(max_iter)
        self.power = power
        self.it = 0
        self.optim = torch.optim.SGD(
                params,
                lr = lr0,
                momentum = momentum,
                weight_decay = wd)

    def get_lr(self):
        factor = (1 - self.it / self.max_iter) ** self.power
        lr = self.lr0 * factor
        return lr

    def step(self):
        self.lr = self.get_lr()
        for pg in self.optim.param_groups:
            pg['lr'] = self.lr
        self.optim.defaults['lr'] = self.lr
        self.it += 1


def train():
    setup_logger(respth)
    logger = logging.getLogger()

    ## dataset
    batchsize = 16
    n_workers = 8
    ds = CityScapes('./data', 'train')
    dl = DataLoader(ds,
                    batch_size = batchsize,
                    shuffle = True,
                    num_workers = n_workers,
                    drop_last = True)

    ## model
    n_classes = 30
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    net = nn.DataParallel(net)
    Loss = nn.CrossEntropyLoss(ignore_index=255)

    ## optimizer
    momentum = 0.9
    weight_decay = 1e-4
    lr_start = 2.5e-2
    max_iter = 10000
    power = 0.9
    optim = Optimizer(
            net.parameters(),
            lr_start,
            momentum,
            weight_decay,
            max_iter,
            power)

    ## train loop
    msg_iter = 20
    loss_avg = []
    st = time.time()
    diter = iter(dl)
    for it in range(max_iter):
        try:
            im, lb = next(diter)
            if not im.size()[0] == batchsize: continue
        except StopIteration:
            diter = iter(dl)
            im, lb = next(diter)
        im = im.cuda()
        lb = lb.cuda()
        H, W = im.size()[2:]
        lb = torch.squeeze(lb, 1)

        out, out16, out32 = net(im)
        out = F.interpolate(out, (H, W), mode='bilinear')
        loss = Loss(out, lb)
        loss.backward()
        optim.step()

        loss_avg.append(loss.item())

        if it % msg_iter == 0 and not it == 0:
            loss_avg = sum(loss_avg) / len(loss_avg)
            lr = optim.lr
            ed = time.time()
            t_interval = ed - st
            msg = ',  '.join([
                    'it: {it}/{max_it}',
                    'lr: {lr:4f}',
                    'loss: {loss:4f}',
                    'time: {time:2f}',
                ]).format(
                    it = it,
                    max_it = max_iter,
                    lr = lr,
                    loss = loss_avg,
                    time = t_interval,
                )
            logger.info(msg)
            loss_avg = []
            st = ed

    ## dump result
    save_pth = osp.join(respth, 'model_final.pth')
    torch.save(net.module.state_dict(), save_pth)
    logger.info('training done, model saved to: {}'.format(save_pth))



if __name__ == "__main__":
    train()
