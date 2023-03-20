import torch
import torch.nn as nn
import torch.nn.functional as F
from .submodule import *

class CorrBlock1D_Cost_Volume:
    def __init__(self, init_corr, corr, num_levels=2, radius=4, inverse_depth_min=None, inverse_depth_max=None, num_sample=None):
        self.num_levels = 2
        self.radius = radius
        self.inverse_depth_min = inverse_depth_min
        self.inverse_depth_max = inverse_depth_max
        self.num_sample = num_sample
        self.corr_pyramid = []
        self.init_corr_pyramid = []

        # all pairs correlation

        # batch, h1, w1, dim, w2 = corr.shape
        b, c, d, h, w = corr.shape
        corr = corr.permute(0, 3, 4, 1, 2).reshape(b*h*w, 1, 1, d)
        init_corr = init_corr.permute(0, 3, 4, 1, 2).reshape(b*h*w, 1, 1, d)

        self.corr_pyramid.append(corr)
        self.init_corr_pyramid.append(init_corr)


        for i in range(self.num_levels):
            corr = F.avg_pool2d(corr, [1,2], stride=[1,2])
            self.corr_pyramid.append(corr)

        for i in range(self.num_levels):
            init_corr = F.avg_pool2d(init_corr, [1,2], stride=[1,2])
            self.init_corr_pyramid.append(init_corr)



    def __call__(self, disp):
        r = self.radius
        b, _, h, w = disp.shape
        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            init_corr = self.init_corr_pyramid[i]
            dx = torch.linspace(-r, r, 2*r+1)
            dx = dx.view(1, 1, 2*r+1, 1).to(disp.device)
            x0 = dx + disp.reshape(b*h*w, 1, 1, 1) / 2**i
            y0 = torch.zeros_like(x0)

            disp_lvl = torch.cat([x0,y0], dim=-1)
            corr = bilinear_sampler(corr, disp_lvl)
            corr = corr.view(b, h, w, -1)

            init_corr = bilinear_sampler(init_corr, disp_lvl)
            init_corr = init_corr.view(b, h, w, -1)

            out_pyramid.append(corr)
            out_pyramid.append(init_corr)


        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()