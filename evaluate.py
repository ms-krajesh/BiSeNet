#!/usr/bin/python
# -*- encoding: utf-8 -*-

from logger import *
from model import BiSeNet
from cityscapes import CityScapes

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import os
import logging
import time
import numpy as np
from tqdm import tqdm


def compute_iou(pred, lb, lb_ignore=255):
    assert pred.shape == lb.shape

    iou_batch = []
    for i, (p, l) in enumerate(zip(pred, lb)):
        clses = set(np.unique(lb).tolist())
        clses.remove(lb_ignore)
        ious = []
        for cls in clses:
            prone = p == cls
            lbone = l == cls
            cross = np.logical_and(prone, lbone)
            union = np.logical_or(prone, lbone)
            iou = float(np.sum(cross)) / float(np.sum(union) + 1e-4)
            ious.append(iou)
        iou_batch.append(sum(ious) / len(ious))
    return iou_batch

def eval_model(net):
    ## dataloader
    ds = CityScapes('./data', 'val')
    dl = DataLoader(ds,
                    batch_size = 4,
                    shuffle = False,
                    num_workers = 4,
                    drop_last = False)

    ## evaluate
    ious = []
    for i, (im, lb) in enumerate(tqdm(dl)):
        im = im.cuda()
        lb = np.squeeze(lb.numpy(), 1)
        _, H, W = lb.shape
        out, out16, out32 = net(im)
        with torch.no_grad():
            scores = F.interpolate(out, (H, W), mode='bilinear')
            probs = F.softmax(scores, 1)
        probs = probs.detach().cpu().numpy()
        pred = np.argmax(probs, axis=1)

        IOUB = compute_iou(pred, lb)
        ious.extend(IOUB)
        #  lb[lb == 255] = 2
        #  lbmax = lbmax if lbmax > np.max(lb) else np.max(lb)
        #  lbmin = lbmin if lbmin < np.min(lb) else np.min(lb)
    mIOU = sum(ious) / len(ious)
    return mIOU


def evaluate():
    respth = './res'
    logger = logging.getLogger()

    ## model
    logger.info('setup and restore model')
    n_classes = 20
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    net.eval()
    save_pth = osp.join(respth, 'model_final.pth.aspaper11000')
    net.load_state_dict(torch.load(save_pth))
    net = nn.DataParallel(net)

    ## dataset
    logger.info('compute the mIOU')
    mIOU = eval_model(net)
    logger.info('mIOU is: {:.6f}'.format(mIOU))




if __name__ == "__main__":
    import sys
    FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
    logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
    evaluate()
