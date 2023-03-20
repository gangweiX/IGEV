from torch.utils.data import Dataset
from datasets.data_io import *
import os
import numpy as np
import cv2
from PIL import Image

class MVSDataset(Dataset):
    def __init__(self, datapath, split='test', n_views=7, img_wh=(1920,1280)):
        self.levels = 4
        self.datapath = datapath
        self.img_wh = img_wh
        self.split = split
        self.build_metas()
        self.n_views = n_views

    def build_metas(self):
        self.metas = []
        if self.split == "test":
            self.scans = ['botanical_garden', 'boulders', 'bridge', 'door',
                'exhibition_hall', 'lecture_room', 'living_room', 'lounge',
                'observatory', 'old_computer', 'statue', 'terrace_2']

        elif self.split == "train":
            self.scans = ['courtyard', 'delivery_area', 'electro', 'facade',
                    'kicker', 'meadow', 'office', 'pipes', 'playground',
                    'relief', 'relief_2', 'terrace', 'terrains']
        

        for scan in self.scans:
            with open(os.path.join(self.datapath, scan, 'pair.txt')) as f:
                num_viewpoint = int(f.readline())
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    if len(src_views) != 0:
                        self.metas += [(scan, -1, ref_view, src_views)]
                    

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
        extrinsics = extrinsics.reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
        intrinsics = intrinsics.reshape((3, 3))
        
        depth_min = float(lines[11].split()[0])
        if depth_min < 0:
            depth_min = 1
        depth_max = float(lines[11].split()[-1])

        return intrinsics, extrinsics, depth_min, depth_max

    def read_img(self, filename, h, w):
        img = Image.open(filename)
        # scale 0~255 to -1~1
        np_img = 2*np.array(img, dtype=np.float32) / 255. - 1
        original_h, original_w, _ = np_img.shape
        np_img = cv2.resize(np_img, self.img_wh, interpolation=cv2.INTER_LINEAR)
        
        np_img_ms = {
            "level_3": cv2.resize(np_img, (w//8, h//8), interpolation=cv2.INTER_LINEAR),
            "level_2": cv2.resize(np_img, (w//4, h//4), interpolation=cv2.INTER_LINEAR),
            "level_1": cv2.resize(np_img, (w//2, h//2), interpolation=cv2.INTER_LINEAR),
            "level_0": np_img
        }
        return np_img_ms, original_h, original_w

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        scan, _, ref_view, src_views = self.metas[idx]
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.n_views-1]
        imgs_0 = []
        imgs_1 = []
        imgs_2 = []
        imgs_3 = []

        # depth = None
        depth_min = None
        depth_max = None

        proj_matrices_0 = []
        proj_matrices_1 = []
        proj_matrices_2 = []
        proj_matrices_3 = []

        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(self.datapath,  scan, f'images/{vid:08d}.jpg')
            proj_mat_filename = os.path.join(self.datapath, scan, f'cams_1/{vid:08d}_cam.txt')

            imgs, original_h, original_w = self.read_img(img_filename,self.img_wh[1], self.img_wh[0])
            imgs_0.append(imgs['level_0'])
            imgs_1.append(imgs['level_1'])
            imgs_2.append(imgs['level_2'])
            imgs_3.append(imgs['level_3'])

            intrinsics, extrinsics, depth_min_, depth_max_ = self.read_cam_file(proj_mat_filename)
            intrinsics[0] *= self.img_wh[0]/original_w
            intrinsics[1] *= self.img_wh[1]/original_h

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
                depth_min = depth_min_
                depth_max = depth_max_

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


        return {"imgs": imgs,                   # N*3*H0*W0
                "proj_matrices": proj, # N*4*4
                "depth_min": depth_min,         # scalar
                "depth_max": depth_max,
                "filename": scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}"
                }  
