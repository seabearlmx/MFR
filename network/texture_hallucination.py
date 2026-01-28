
import torch
import torch.nn as nn
import torch.distributions as tdist
import torch.nn.functional as F
import math
import random


class TextureHallucination(nn.Module):
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.TRF = nn.Sequential(
            nn.Conv2d(19, 19, kernel_size=1, groups=19, bias=False),
            nn.InstanceNorm2d(19),
            nn.ReLU(inplace=True))
        self.loss = nn.MSELoss()


    def forward(self, x_cls):
        B,C,H,W = x_cls.size()  # B 19 H W

        HW = H * W

        x_texture = self.TRF(x_cls)

        x_cls_m = x_cls.contiguous().view(B, C, -1)
        x_cls_cov_matrix = torch.bmm(x_cls_m, x_cls_m.transpose(1, 2)).div(HW - 1)
        x_texture_m = x_texture.contiguous().view(B, C, -1)
        x_texture_cov_matrix = torch.bmm(x_texture_m, x_texture_m.transpose(1, 2)).div(HW - 1)
        loss_diff_texture = 1.0 / self.loss(x_cls_cov_matrix, x_texture_cov_matrix)

        return x_texture, loss_diff_texture



