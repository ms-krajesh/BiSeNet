#!/usr/bin/python
# -*- encoding: utf-8 -*-


from logger import *
from model import BiSeNet
from cityscapes import CityScapes
from evaluate import eval_model

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import os
import logging
import time
import datetime


respth = './res'
if not osp.exists(respth): os.makedirs(respth)
setup_logger(respth)
logger = logging.getLogger()

class Optimizer(object):
    def __init__(self,
                params,
                lr0,
                momentum,
                wd,
                warmup_steps,
                warmup_start_lr,
                max_iter,
                power):
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr
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
        self.warmup_factor = (self.lr0 / self.warmup_start_lr) ** (1. / self.warmup_steps)

    def get_lr(self):
        if self.it <= self.warmup_steps:
            lr = self.warmup_start_lr * (self.warmup_factor ** self.it)
        else:
            factor = (1 - (self.it - self.warmup_steps) / (self.max_iter - self.warmup_steps)) ** self.power
            lr = self.lr0 * factor
        return lr

    def step(self):
        self.lr = self.get_lr()
        for pg in self.optim.param_groups:
            pg['lr'] = self.lr
        self.optim.defaults['lr'] = self.lr
        self.it += 1
        self.optim.step()
        if self.it == self.warmup_steps:
            logger.info('==> warmup done, start to implement poly lr strategy')

    def zero_grad(self):
        self.optim.zero_grad()


def train():
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
    n_classes = 20
    ignore_idx = 255
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    net.train()
    net = nn.DataParallel(net)
    Loss = nn.CrossEntropyLoss(ignore_index=ignore_idx)

    ## optimizer
    momentum = 0.9
    weight_decay = 1e-4
    lr_start = 2.5e-2
    #  lr_start = 1e-3
    max_iter = 11000
    power = 0.9
    warmup_steps = 1
    warmup_start_lr = 2.5e-2
    optim = Optimizer(
            net.parameters(),
            lr_start,
            momentum,
            weight_decay,
            warmup_steps,
            warmup_start_lr,
            max_iter,
            power)

    ## train loop
    msg_iter = 20
    loss_avg = []
    st = glob_st = time.time()
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

        optim.zero_grad()
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
            t_intv, glob_t_intv = ed - st, ed - glob_st
            eta = int((max_iter - it) * (glob_t_intv / it))
            eta = str(datetime.timedelta(seconds = eta))
            msg = ', '.join([
                    'it: {it}/{max_it}',
                    'lr: {lr:4f}',
                    'loss: {loss:.4f}',
                    'eta: {eta}',
                    'time: {time:.4f}',
                ]).format(
                    it = it,
                    max_it = max_iter,
                    lr = lr,
                    loss = loss_avg,
                    time = t_intv,
                    eta = eta
                )
            logger.info(msg)
            loss_avg = []
            st = ed

    ## dump result
    save_pth = osp.join(respth, 'model_final.pth')
    torch.save(net.module.state_dict(), save_pth)
    logger.info('training done, model saved to: {}'.format(save_pth))

    ## evaluate
    logger.info('start evaluating the model')
    net.eval()
    mIOU = eval_model(net)
    logger.info('mIOU is: {}'.format(mIOU))



if __name__ == "__main__":
    train()
