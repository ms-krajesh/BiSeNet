#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os.path as osp
import os
from PIL import Image
import numpy as np

from transform import *
from preprocess_data import labels_info



class CityScapes(Dataset):
    def __init__(self, rootpth, mode = 'train', *args, **kwargs):
        super(CityScapes, self).__init__(*args, **kwargs)
        assert mode in ('train', 'val', 'test')
        self.mode = mode
        self.n_classes = 19
        self.ignore_lb = 255
        self.lb_map = {el['id']: el['trainId'] for el in labels_info}
        self.lb_ignore_eval = [el['id'] for el in labels_info if el['ignoreInEval']]

        ## parse img directory
        self.imgs = {}
        imgnames = []
        impth = osp.join(rootpth, 'leftImg8bit', mode)
        folders = os.listdir(impth)
        for fd in folders:
            fdpth = osp.join(impth, fd)
            im_names = os.listdir(fdpth)
            names = [el.replace('_leftImg8bit.png', '') for el in im_names]
            impths = [osp.join(fdpth, el) for el in im_names]
            imgnames.extend(names)
            self.imgs.update(dict(zip(names, impths)))

        ## parse gt directory
        self.labels = {}
        gtnames = []
        gtpth = osp.join(rootpth, 'gtFine', mode)
        folders = os.listdir(gtpth)
        for fd in folders:
            fdpth = osp.join(gtpth, fd)
            lbnames = os.listdir(fdpth)
            lbnames = [el for el in lbnames if 'labelIds' in el]
            names = [el.replace('_gtFine_labelIds.png', '') for el in lbnames]
            lbpths = [osp.join(fdpth, el) for el in lbnames]
            gtnames.extend(names)
            self.labels.update(dict(zip(names, lbpths)))

        self.imnames = imgnames
        self.len = len(self.imnames)
        assert set(imgnames) == set(gtnames)
        assert set(self.imnames) == set(self.imgs.keys())
        assert set(self.imnames) == set(self.labels.keys())

        ## pre-processing
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        if self.mode == 'train':
            self.trans = Compose([
                HorizontalFlip(),
                RandomScale((0.75, 1.0, 1.5, 1.75, 2.0)),
                RandomCrop((640, 360))
                ])


    def __getitem__(self, idx):
        fn  = self.imnames[idx]
        impth = self.imgs[fn]
        lbpth = self.labels[fn]
        img = Image.open(impth)
        label = Image.open(lbpth)
        if self.mode == 'train':
            im_lb = dict(im = img, lb = label)
            im_lb = self.trans(im_lb)
            img, label = im_lb['im'], im_lb['lb']
        img = self.to_tensor(img)
        label = np.array(label).astype(np.int64)[np.newaxis, :]
        label = self.convert_labels(label)
        return img, label


    def __len__(self):
        return self.len


    def convert_labels(self, label):
        if self.mode in ('val', 'test'):
            label[np.isin(label, self.lb_ignore_eval)] = self.ignore_lb
        for k, v in self.lb_map.items():
            label[label == k] = v
        return label



if __name__ == "__main__":
    ds = CityScapes('./data/', mode = 'val')
    im, label = ds[10]
    #  print(label)
    print(type(label))
    #  print(type(im))
    #  print(im.size())
    #  print(label.size())
    #  print(type(label))
    from torch.utils.data import DataLoader
    dl = DataLoader(ds,
                    batch_size = 30,
                    shuffle = False,
                    num_workers = 6,
                    drop_last = True)
    im, lb = next(iter(dl))
    lb = lb.numpy()
    #  print(lb)
    print(type(lb))
    print(lb.shape)
    print(label.shape)
    #  print(im)
    print(im.size())
    print(np.max(lb))

    from tqdm import tqdm

    diter = iter(dl)
    lmax, lmin = -1, 1000
    for i, (im, lb) in enumerate(tqdm(diter)):
        lb = lb.numpy()
        lb[lb == 255] = 3
        lmax = np.max(lb) if lmax < np.max(lb) else lmax
        lmin = np.min(lb) if lmin > np.min(lb) else lmin

    print(lmax, lmin)
