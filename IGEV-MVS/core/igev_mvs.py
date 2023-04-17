import torch
import torch.nn as nn
import torch.nn.functional as F
from .submodule import *
from .corr import *
from .extractor import *
from .update import *

try:
    autocast = torch.cuda.amp.autocast
except:
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

class IGEVMVS(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        context_dims = [128, 128, 128]
        self.n_gru_layers = 3
        self.slow_fast_gru = False
        self.mixed_precision = True
        self.num_sample = 64
        self.G = 1
        self.corr_radius = 4
        self.corr_levels = 2
        self.iters = args.iteration
        self.update_block = BasicMultiUpdateBlock(hidden_dims=context_dims)
        self.conv_hidden_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1)
        self.conv_hidden_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2) 
        self.conv_hidden_4 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2) 
        self.feature = Feature()

        self.stem_2 = nn.Sequential(
            BasicConv_IN(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32), nn.ReLU()
            )
        self.stem_4 = nn.Sequential(
            BasicConv_IN(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(48), nn.ReLU()
            )

        self.conv = BasicConv_IN(96, 48, kernel_size=3, padding=1, stride=1)
        self.desc = nn.Conv2d(48, 48, kernel_size=1, padding=0, stride=1)

        self.spx = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)
        self.spx_2 = Conv2x_IN(32, 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv_IN(96, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32), nn.ReLU()
            )

        self.depth_initialization = DepthInitialization(self.num_sample)
        self.pixel_view_weight = PixelViewWeight(self.G)

        self.corr_stem = BasicConv(1, 8, is_3d=True, kernel_size=3, stride=1, padding=1)
        self.corr_feature_att = FeatureAtt(8, 96)
        self.cost_agg = hourglass(8)

        self.spx_2_gru = Conv2x(32, 32, True)
        self.spx_gru = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)

    def upsample_disp(self, depth, mask_feat_4, stem_2x):
        with autocast(enabled=self.mixed_precision):
            xspx = self.spx_2_gru(mask_feat_4, stem_2x)
            spx_pred = self.spx_gru(xspx)
            spx_pred = F.softmax(spx_pred, 1)

            up_depth = context_upsample(depth, spx_pred).unsqueeze(1)

        return up_depth

    def forward(self, imgs, proj_matrices, depth_min, depth_max, test_mode=False):
        proj_matrices_2 = torch.unbind(proj_matrices['level_2'].float(), 1)
        depth_min = depth_min.float()
        depth_max = depth_max.float()

        ref_proj = proj_matrices_2[0]
        src_projs = proj_matrices_2[1:]

        with autocast(enabled=self.mixed_precision):
            images = torch.unbind(imgs['level_0'], dim=1)
            features = self.feature(imgs['level_0'])
            ref_feature = []
            for fea in features:
                ref_feature.append(torch.unbind(fea, dim=1)[0])
            src_features = [src_fea for src_fea in torch.unbind(features[0], dim=1)[1:]]

            stem_2x = self.stem_2(images[0])
            stem_4x = self.stem_4(stem_2x)
            ref_feature[0] = torch.cat((ref_feature[0], stem_4x), 1)

            for idx, src_fea in enumerate(src_features):
                stem_2y = self.stem_2(images[idx + 1])
                stem_4y = self.stem_4(stem_2y)
                src_features[idx] = torch.cat((src_fea, stem_4y), 1)

            match_left = self.desc(self.conv(ref_feature[0]))
            match_left = match_left / torch.norm(match_left, 2, 1, True)

            match_rights = [self.desc(self.conv(src_fea)) for src_fea in src_features]
            match_rights = [match_right / torch.norm(match_right, 2, 1, True) for match_right in match_rights]

            xspx = self.spx_4(ref_feature[0])
            xspx = self.spx_2(xspx, stem_2x)
            spx_pred = self.spx(xspx)
            spx_pred = F.softmax(spx_pred, 1)

            batch, dim, height, width = match_left.size()
            inverse_depth_min = (1.0 / depth_min).view(batch, 1, 1, 1)
            inverse_depth_max = (1.0 / depth_max).view(batch, 1, 1, 1)

            device = match_left.device
            correlation_sum = 0
            view_weight_sum = 1e-5

        match_left = match_left.float()
        depth_samples = self.depth_initialization(inverse_depth_min, inverse_depth_max, height, width, device)
        for src_feature, src_proj in zip(match_rights, src_projs):
            src_feature = src_feature.float()
            warped_feature = differentiable_warping(src_feature, src_proj, ref_proj, depth_samples)
            warped_feature = warped_feature.view(batch, self.G, dim // self.G, self.num_sample, height, width)
            correlation = torch.mean(warped_feature * match_left.view(batch, self.G, dim // self.G, 1, height, width), dim=2, keepdim=False)

            view_weight = self.pixel_view_weight(correlation)
            del warped_feature, src_feature, src_proj

            correlation_sum += correlation * view_weight.unsqueeze(1)
            view_weight_sum += view_weight.unsqueeze(1) 
            del correlation, view_weight
        del match_left, match_rights, src_projs
                
        with autocast(enabled=self.mixed_precision):
            init_corr_volume = correlation_sum.div_(view_weight_sum)
            corr_volume = self.corr_stem(init_corr_volume)
            corr_volume = self.corr_feature_att(corr_volume, ref_feature[0])
            regularized_cost_volume = self.cost_agg(corr_volume, ref_feature)

            GEV_hidden = self.conv_hidden_1(regularized_cost_volume.squeeze(1))

            GEV_hidden_2 = self.conv_hidden_2(GEV_hidden)

            GEV_hidden_4 = self.conv_hidden_4(GEV_hidden_2)

            net_list = [GEV_hidden, GEV_hidden_2, GEV_hidden_4]

            net_list = [torch.tanh(x) for x in net_list]

        corr_block = CorrBlock1D_Cost_Volume

        init_corr_volume = init_corr_volume.float()
        regularized_cost_volume = regularized_cost_volume.float()
        probability = F.softmax(regularized_cost_volume.squeeze(1), dim=1)
        index = torch.arange(0, self.num_sample, 1, device=probability.device).view(1, self.num_sample, 1, 1).float()
        disp_init = torch.sum(index * probability, dim = 1, keepdim=True)

        corr_fn = corr_block(init_corr_volume, regularized_cost_volume, radius=self.corr_radius, num_levels=self.corr_levels, inverse_depth_min=inverse_depth_min, inverse_depth_max=inverse_depth_max, num_sample=self.num_sample)

        disp_predictions = []
        disp = disp_init

        for itr in range(self.iters):
            disp = disp.detach()
            corr = corr_fn(disp)

            with autocast(enabled=self.mixed_precision):
                if self.n_gru_layers == 3 and self.slow_fast_gru: # Update low-res GRU
                    net_list = self.update_block(net_list, iter16=True, iter08=False, iter04=False, update=False)
                if self.n_gru_layers >= 2 and self.slow_fast_gru:# Update low-res GRU and mid-res GRU
                    net_list = self.update_block(net_list, iter16=self.n_gru_layers==3, iter08=True, iter04=False, update=False)
                net_list, mask_feat_4, delta_disp = self.update_block(net_list, corr, disp, iter16=self.n_gru_layers==3, iter08=self.n_gru_layers>=2)

            disp = disp + delta_disp

            if test_mode and itr < self.iters-1:
                continue

            disp_up = self.upsample_disp(disp, mask_feat_4, stem_2x)  / (self.num_sample-1)
            disp_predictions.append(disp_up)

        disp_init = context_upsample(disp_init, spx_pred.float()).unsqueeze(1)  / (self.num_sample-1)

        if test_mode:
            return disp_up
 

        return disp_init, disp_predictions
