#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.utils.model_zoo as modelzoo
import torch.nn.functional as F
import torchvision


resnet18_url = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks = 3, stride=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride=stride,
                padding = 1,
                bias=True)
        self.bn = nn.BatchNorm2d(out_chan)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace = True)
        return x



class UpSampleOut(nn.Module):
    def __init__(self, in_chan, out_chan, ks = 3, stride=1, *args, **kwargs):
        super(UpSampleOut, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride=stride,
                padding = 1,
                bias=True)
        self.bn = nn.BatchNorm2d(out_chan)
        self.conv_out = nn.Conv2d(out_chan,
                out_chan,
                kernel_size = 1,
                bias=True)

    def forward(self, x, size):
        x = F.interpolate(x, size, mode='bilinear')
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace = True)
        x = self.conv_out(x)
        return x


class SpatialPath(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SpatialPath, self).__init__()
        self.conv1 = ConvBNReLU(3, 64, stride=2)
        self.conv2 = ConvBNReLU(64, 256, stride=2)
        self.conv3 = ConvBNReLU(256, 512, stride=2)
        #  self.conv1 = ConvBNReLU(3, 64, ks=3, stride=2)
        #  self.conv2 = ConvBNReLU(32, 128, ks=3, stride=2)
        #  self.conv3 = ConvBNReLU(64, 512, ks=3, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, *args, **kwargs):
        super(AttentionRefinementModule, self).__init__()
        self.in_chan = in_chan
        self.conv = nn.Conv2d(in_chan,
                in_chan,
                kernel_size = 1,
                bias=True)
        self.bn = nn.BatchNorm2d(in_chan)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        assert self.in_chan == x.size()[1]
        in_ten = x
        x = F.avg_pool2d(x, x.size()[2:])
        x = self.conv(x)
        x = self.bn(x)
        x = self.sigmoid(x)
        x = torch.mul(in_ten, x)
        return x



class ContextPath(nn.Module):
    def __init__(self, n_classes = 10, *args, **kwargs):
        super(ContextPath, self).__init__()
        resnet = torchvision.models.resnet18()
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.arm16 = AttentionRefinementModule(256)
        self.arm32 = AttentionRefinementModule(512)

        self.conv_feat16 = nn.Conv2d(256,
                n_classes,
                kernel_size = 3,
                bias=True)
        self.conv_feat32 = nn.Conv2d(512,
                n_classes,
                kernel_size = 3,
                bias=True)
        #  self.conv_feat16_out = UpSampleOut(256, n_classes)
        #  self.conv_feat32_out = UpSampleOut(512, n_classes)
        self.conv16_inner = nn.Conv2d(256,
                512,
                kernel_size = 1,
                bias=True)
        self.conv16_outer = nn.Conv2d(512,
                512,
                kernel_size = 3,
                padding = 1,
                bias=True)
        self.conv32_avg = nn.Conv2d(512,
                512,
                kernel_size = 1,
                bias=True)
        self.bn32_avg = nn.BatchNorm2d(512)
        self.sig_avg = nn.Sigmoid()

        self.init_weight()


    def forward(self, x):
        H0, W0 = x.size()[2:]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        feat16 = self.layer3(x)
        feat32 = self.layer4(feat16)
        #  H, W = x.size()[2:]

        H8, W8 = x.size()[2:]
        H16, W16 = feat16.size()[2:]
        H32, W32 = feat32.size()[2:]

        avg = F.avg_pool2d(feat32, feat32.size()[2:])
        avg_conv = self.conv32_avg(avg)
        avg_bn = self.bn32_avg(avg_conv)
        avg_sig = self.sig_avg(avg_bn)

        feat16_arm = self.arm16(feat16)
        feat32_arm = self.arm32(feat32)

        ## TODO: 1. try nearest as FPN
        ## TODO: 2. try first FPN then channel wise multiplication
        feat32_up = F.interpolate(feat32_arm, (H16, W16), mode='bilinear')
        feat16_inner = self.conv16_inner(feat16_arm)
        feat16_inner = feat16_inner + feat32_up
        feat16_out = self.conv16_outer(feat16_inner)

        feat16_out = torch.mul(avg_sig, feat16_out)
        feat_out = F.interpolate(feat16_out, (H8, W8), mode = 'bilinear')
        ## TODO: 3. try to use resnet feature instead of arm feature as output feat
        #  feat_out16 = self.conv_feat16_out(feat16_arm, (H0, W0))
        #  feat_out32 = self.conv_feat32_out(feat32_arm, (H0, W0))

        #  feat32_up = F.interpolate(feat32_with_avg, (H, W), mode = 'bilinear')
        #  feat16_up = F.interpolate(feat16_arm, (H, W), mode = 'bilinear')
        #  feat_out = torch.cat((feat32_up, feat16_up), dim = 1)
        feat_out16 = self.conv_feat16(feat16)
        feat_out32 = self.conv_feat32(feat32)

        return feat_out, feat_out16, feat_out32

    def init_weight(self):
        state_dict = modelzoo.load_url(resnet18_url)
        self_state_dict = self.state_dict()
        for k, v in state_dict.items():
            if 'fc' in k: continue
            self_state_dict.update({k: v})
        self.load_state_dict(self_state_dict)




class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, n_classes, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, n_classes, ks=3, stride=2)
        self.conv1 = nn.Conv2d(n_classes, n_classes, 1)
        self.conv2 = nn.Conv2d(n_classes, n_classes, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fsp, fcp):
        fcat = torch.cat((fsp, fcp), dim = 1)
        feat = self.convblk(fcat)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv1(atten)
        atten = F.relu(atten, inplace = True)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out



class BiSeNet(nn.Module):
    def __init__(self, n_classes, *args, **kwargs):
        super(BiSeNet, self).__init__()
        self.sp = SpatialPath()
        self.cp = ContextPath(n_classes)
        self.ffm = FeatureFusionModule(1024, n_classes)
        self.conv = nn.Conv2d(n_classes,
                n_classes,
                kernel_size=1,
                stride=1,
                bias=True)


    def forward(self, x):
        H, W = x.size()[2:]
        feat_sp = self.sp(x)
        feat_cp, feat16, feat32 = self.cp(x)
        feat_out = self.ffm(feat_sp, feat_cp)
        feat_out = F.interpolate(feat_out, (H, W), mode='bilinear')
        feat_out = self.conv(feat_out)
        return feat_out, feat16, feat32




if __name__ == "__main__":
    net = BiSeNet(19)
    in_ten = torch.randn(16, 3, 360, 640)
    out, out16, out32 = net(in_ten)
    print(out.shape)
    print(out16.shape)
    print(out32.shape)

    #  convbnrelu = ConvBNReLU(3, 10)
    #  print(convbnrelu(in_ten).shape)
    #  sp = SpatialPath()
    #  out = sp(in_ten)
    #  print(out.shape)
    #  cp = ContextPath(10)
    #  out, out16, out32 = cp(in_ten)
    #  print(out.shape)
    #  print(out16.shape)
    #  print(out32.shape)
    #  #  arm = AttentionRefinementModule(3, 10)
    #  #  out = arm(in_ten)
    #  #  print(out.shape)
    #  #  #  out_x, out_aux = net(in_ten)
    #  #  #  print(out_x.shape)
    #  #  #  print(out_aux.shape)
    #  #  in_ten = torch.randn(1, 2, 3,3)
    #  #  print(in_ten)
    #  #  import numpy as np
    #  #  sig = np.arange(2).reshape(1,2,1,1).astype(np.float32)
    #  #  sig = torch.tensor(sig)
    #  #  print(torch.mul(in_ten, sig))
    #
    #  ffm = FeatureFusionModule(in_chan = 1024, n_classes = 21)
    #  feat1 = torch.randn(10, 768, 32, 32)
    #  feat2 = torch.randn(10, 256, 32, 32)
    #  feat_out = ffm(feat1, feat2)
    #  print(feat_out.shape)
