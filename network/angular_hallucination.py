
import torch
import torch.nn as nn
import torch.distributions as tdist
import torch.nn.functional as F
import math
import random
from network.mynn import initialize_weights, Norm2d, Upsample, freeze_weights, unfreeze_weights


class AngularHallucination(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, x_cls):
        B,C,H,W = x_cls.size()  
        x_cls_ori = x_cls

        x_cls = x_cls.unsqueeze(2)

        angle = random.randint(1, 360)

        theta = torch.tensor([[math.cos(angle), 0, math.sin(angle), 0],
                              [0, 1, 0, 0],
                              [-math.sin(angle), 0, math.cos(angle), 0]], dtype=torch.float).cuda()
        theta = theta.unsqueeze(0).repeat(B, 1, 1)

        grid = F.affine_grid(theta, x_cls.size())

        x_angular = F.grid_sample(x_cls, grid)

        x_angular = x_angular.squeeze(2)  # B 19 H W


        x_cls_temp = x_cls_ori.contiguous().view(B, C, -1)

        x_angular_temp = x_angular.contiguous().view(B, C, -1).transpose(1, 2)

        corr_matrix = torch.bmm(x_cls_temp, x_angular_temp)
        pro_corr_matrix = F.softmax(corr_matrix, dim=1)

        x_angular = torch.bmm(pro_corr_matrix, x_cls_temp).contiguous().view(B, C, H, W) #+ x_cls_ori

        return x_angular

