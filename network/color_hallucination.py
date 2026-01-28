
import torch
import torch.nn as nn
import torch.distributions as tdist
import torch.nn.functional as F
import math
import random


class ColorHallucination(nn.Module):
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.contrast = nn.Sequential(
            nn.InstanceNorm2d(19),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)))
        self.brightness = nn.Sequential(
            nn.InstanceNorm2d(19),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)))
        self.saturation = nn.Sequential(
            nn.InstanceNorm2d(19),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)))
        self.hue = nn.Sequential(
            nn.InstanceNorm2d(19),
            nn.Sigmoid(),
            nn.AdaptiveAvgPool2d((1, 1)))


    def forward(self, x_cls):
        B,C,H,W = x_cls.size()  # B 19 H W

        c_mat = torch.randn(B,C,H,W).cuda()
        b_mat = torch.randn(B,C,H,W).cuda()
        s_mat = torch.randn(B,C,H,W).cuda()
        h_mat = torch.randn(B,C,H,W).cuda()

        c_x_cls = x_cls * c_mat
        b_x_cls = x_cls * b_mat
        s_x_cls = x_cls * s_mat
        h_x_cls = x_cls * h_mat

        c_x_cls = self.contrast(c_x_cls)
        b_x_cls = self.brightness(b_x_cls)
        s_x_cls = self.saturation(s_x_cls)
        h_x_cls = self.hue(h_x_cls)

        x_color = x_cls * c_x_cls * b_x_cls * s_x_cls * h_x_cls

        return x_color



