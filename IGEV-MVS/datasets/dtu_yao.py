from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from datasets.data_io import *
import cv2
import random
from torchvision import transforms


class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, robust_train = False):
        super(MVSDataset, self).__init__()

        self.levels = 4
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.img_wh = (640, 512)
        # self.img_wh = (1440, 1056)
        self.robust_train = robust_train
        

        assert self.mode in ["train", "val", "test"]
        self.metas = self.build_list()
        self.color_augment = transforms.ColorJitter(brightness=0.5, contrast=0.5)

    def build_list(self):
        metas = []
        with open(self.listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        for scan in scans:
            pair_file = "Cameras_1/pair.txt"
            
            with open(os.path.join(self.datapath, pair_file)) as f:
                self.num_viewpoint = int(f.readline())
                # viewpoints (49)
                for view_idx in range(self.num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    # light conditions 0-6
                    for light_idx in range(7):
                        metas.append((scan, light_idx, ref_view, src_views))
        print("dataset", self.mode, "metas:", len(metas))
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        depth_min = float(lines[11].split()[0])
        depth_max = float(lines[11].split()[-1])
        return intrinsics, extrinsics, depth_min, depth_max

    def read_img(self, filename):
        img = Image.open(filename)
        if self.mode=='train':
            img = self.color_augment(img)
        # scale 0~255 to -1~1
        np_img = 2*np.array(img, dtype=np.float32) / 255. - 1
        h, w, _ = np_img.shape
        np_img_ms = {
            "level_3": cv2.resize(np_img, (w//8, h//8), interpolation=cv2.INTER_LINEAR), 
            "level_2": cv2.resize(np_img, (w//4, h//4), interpolation=cv2.INTER_LINEAR),
            "level_1": cv2.resize(np_img, (w//2, h//2), interpolation=cv2.INTER_LINEAR),
            "level_0": np_img
        }
        return np_img_ms


    def prepare_img(self, hr_img):
        #downsample
        h, w = hr_img.shape
        # original w,h: 1600, 1200; downsample -> 800, 600 ; crop -> 640, 512
        hr_img = cv2.resize(hr_img, (w//2, h//2), interpolation=cv2.INTER_NEAREST)
        #crop
        h, w = hr_img.shape
        target_h, target_w = self.img_wh[1], self.img_wh[0]
        start_h, start_w = (h - target_h)//2, (w - target_w)//2
        hr_img_crop = hr_img[start_h: start_h + target_h, start_w: start_w + target_w]

        return hr_img_crop

    def read_mask(self, filename):
        img = Image.open(filename)
        np_img = np.array(img, dtype=np.float32)
        np_img = (np_img > 10).astype(np.float32)
        return np_img


    def read_depth_mask(self, filename, mask_filename, scale):
        depth_hr = np.array(read_pfm(filename)[0], dtype=np.float32) * scale
        depth_hr = np.squeeze(depth_hr,2)
        depth_lr = self.prepare_img(depth_hr)
        mask = self.read_mask(mask_filename)
        mask = self.prepare_img(mask)
        mask = mask.astype(np.bool_)
        mask = mask.astype(np.float32)
        
        h, w = depth_lr.shape
        depth_lr_ms = {}
        mask_ms = {}

        for i in range(self.levels):
            depth_cur = cv2.resize(depth_lr, (w//(2**i), h//(2**i)), interpolation=cv2.INTER_NEAREST)
            mask_cur = cv2.resize(mask, (w//(2**i), h//(2**i)), interpolation=cv2.INTER_NEAREST)
            depth_lr_ms[f"level_{i}"] = depth_cur
            mask_ms[f"level_{i}"] = mask_cur

        return depth_lr_ms, mask_ms


    def __getitem__(self, idx):
        meta = self.metas[idx]
        scan, light_idx, ref_view, src_views = meta
        # robust training strategy
        if self.robust_train:
            num_src_views = len(src_views)
            index = random.sample(range(num_src_views), self.nviews - 1)
            view_ids = [ref_view] + [src_views[i] for i in index]
            scale = random.uniform(0.8, 1.25)

        else:
            view_ids = [ref_view] + src_views[:self.nviews - 1]
            scale = 1

        imgs_0 = []
        imgs_1 = []
        imgs_2 = []
        imgs_3 = []

        mask = None
        depth = None
        depth_min = None
        depth_max = None

        proj_matrices_0 = []
        proj_matrices_1 = []
        proj_matrices_2 = []
        proj_matrices_3 = []



        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(self.datapath,
                                    'Rectified/{}_train/rect_{:0>3}_{}_r5000.png'.format(scan, vid + 1, light_idx))
            proj_mat_filename = os.path.join(self.datapath, 'Cameras_1/{}_train/{:0>8}_cam.txt').format(scan, vid)

            mask_filename = os.path.join(self.datapath, 'Depths_raw/{}/depth_visual_{:0>4}.png'.format(scan, vid))
            depth_filename = os.path.join(self.datapath, 'Depths_raw/{}/depth_map_{:0>4}.pfm'.format(scan, vid))

            imgs = self.read_img(img_filename)
            imgs_0.append(imgs['level_0'])
            imgs_1.append(imgs['level_1'])
            imgs_2.append(imgs['level_2'])
            imgs_3.append(imgs['level_3'])

            intrinsics, extrinsics, depth_min_, depth_max_ = self.read_cam_file(proj_mat_filename)
            extrinsics[:3,3] *= scale
            intrinsics[0] *= 4
            intrinsics[1] *= 4

            proj_mat = extrinsics.copy()
            intrinsics[:2,:] *= 0.125
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices_3.append(proj_mat)

            proj_mat = extrinsics.copy()
            intrinsics[:2,:] *= 2
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices_2.append(proj_mat)

            proj_mat = extrinsics.copy()
            intrinsics[:2,:] *= 2
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices_1.append(proj_mat)

            proj_mat = extrinsics.copy()
            intrinsics[:2,:] *= 2
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices_0.append(proj_mat)

            if i == 0:  # reference view
                depth_min = depth_min_ * scale
                depth_max = depth_max_ * scale
                depth, mask = self.read_depth_mask(depth_filename, mask_filename, scale)

                for l in range(self.levels):
                    mask[f'level_{l}'] = np.expand_dims(mask[f'level_{l}'],2)
                    mask[f'level_{l}'] = mask[f'level_{l}'].transpose([2,0,1])
                    depth[f'level_{l}'] = np.expand_dims(depth[f'level_{l}'],2)
                    depth[f'level_{l}'] = depth[f'level_{l}'].transpose([2,0,1])

        # imgs: N*3*H0*W0, N is number of images
        imgs_0 = np.stack(imgs_0).transpose([0, 3, 1, 2])
        imgs_1 = np.stack(imgs_1).transpose([0, 3, 1, 2])
        imgs_2 = np.stack(imgs_2).transpose([0, 3, 1, 2])
        imgs_3 = np.stack(imgs_3).transpose([0, 3, 1, 2])

        imgs = {}
        imgs['level_0'] = imgs_0
        imgs['level_1'] = imgs_1
        imgs['level_2'] = imgs_2
        imgs['level_3'] = imgs_3

        # proj_matrices: N*4*4
        proj_matrices_0 = np.stack(proj_matrices_0)
        proj_matrices_1 = np.stack(proj_matrices_1)
        proj_matrices_2 = np.stack(proj_matrices_2)
        proj_matrices_3 = np.stack(proj_matrices_3)
        
        proj={}
        proj['level_3']=proj_matrices_3
        proj['level_2']=proj_matrices_2
        proj['level_1']=proj_matrices_1
        proj['level_0']=proj_matrices_0


        # data is numpy array
        return {"imgs": imgs,                   # [N, 3, H, W]
                "proj_matrices": proj,          # [N,4,4]
                "depth": depth,                 # [1, H, W]
                "depth_min": depth_min,         # scalar
                "depth_max": depth_max,         # scalar
                "mask": mask}                   # [1, H, W]

