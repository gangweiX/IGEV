import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import time
from datasets import find_dataset_def
from core.igev_mvs import IGEVMVS
from utils import *
import sys
import cv2
from datasets.data_io import read_pfm, save_pfm
from core.submodule import depth_unnormalization
from plyfile import PlyData, PlyElement
from tqdm import tqdm
from PIL import Image

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Predict depth, filter, and fuse')
parser.add_argument('--model', default='IterMVS', help='select model')
parser.add_argument('--dataset', default='dtu_yao_eval', help='select dataset')
parser.add_argument('--testpath', default='/data/DTU_data/dtu_test/', help='testing data path')
parser.add_argument('--testlist', default='./lists/dtu/test.txt', help='testing scan list')
parser.add_argument('--maxdisp', default=256)
parser.add_argument('--split', default='intermediate', help='select data')
parser.add_argument('--batch_size', type=int, default=2, help='testing batch size')
parser.add_argument('--n_views', type=int, default=5, help='num of view')
parser.add_argument('--img_wh', nargs='+', type=int, default=[640, 480],
        help='height and width of the image')
parser.add_argument('--loadckpt', default='./pretrained_models/dtu.ckpt', help='load a specific checkpoint')
parser.add_argument('--outdir', default='./output/', help='output dir')
parser.add_argument('--display', action='store_true', help='display depth images and masks')
parser.add_argument('--iteration', type=int, default=32, help='num of iteration of GRU')
parser.add_argument('--geo_pixel_thres', type=float, default=1, help='pixel threshold for geometric consistency filtering')
parser.add_argument('--geo_depth_thres', type=float, default=0.01, help='depth threshold for geometric consistency filtering')
parser.add_argument('--photo_thres', type=float, default=0.3, help='threshold for photometric consistency filtering')

# parse arguments and check
args = parser.parse_args()
print("argv:", sys.argv[1:])
print_args(args)

if args.dataset=="dtu_yao_eval":
    img_wh=(1600, 1152)
elif args.dataset=="tanks":
    img_wh=(1920, 1024)
elif args.dataset=="eth3d":
    img_wh = (1920,1280)
else:
    img_wh = (args.img_wh[0], args.img_wh[1]) # custom dataset

# read intrinsics and extrinsics
def read_camera_parameters(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    
    return intrinsics, extrinsics


# read an image
def read_img(filename, img_wh):
    img = Image.open(filename)
    # scale 0~255 to 0~1
    np_img = np.array(img, dtype=np.float32) / 255.
    original_h, original_w, _ = np_img.shape
    np_img = cv2.resize(np_img, img_wh, interpolation=cv2.INTER_LINEAR)
    return np_img, original_h, original_w


# save a binary mask
def save_mask(filename, mask):
    assert mask.dtype == np.bool_
    mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask).save(filename)

def save_depth_img(filename, depth):
    # assert mask.dtype == np.bool
    depth = depth.astype(np.float32) * 255
    Image.fromarray(depth).save(filename)


def read_pair_file(filename):
    data = []
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        # 49 viewpoints
        for view_idx in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            if len(src_views) != 0:
                data.append((ref_view, src_views))
    return data


# run MVS model to save depth maps
def save_depth():
    # dataset, dataloader
    MVSDataset = find_dataset_def(args.dataset)
    if args.dataset=="dtu_yao_eval":
        test_dataset = MVSDataset(args.testpath, args.testlist, args.n_views, img_wh)
    elif args.dataset=="tanks":
        test_dataset = MVSDataset(args.testpath, args.n_views, img_wh, args.split)
    elif args.dataset=="eth3d":
        test_dataset = MVSDataset(args.testpath, args.split, args.n_views, img_wh)
    else:
        test_dataset = MVSDataset(args.testpath, args.n_views, img_wh)
    TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=4, drop_last=False)

    # model
    model = IGEVMVS(args)
    model = nn.DataParallel(model)
    model.cuda()

    # load checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'])
    model.eval()
    
    with torch.no_grad():
        tbar = tqdm(TestImgLoader)
        for batch_idx, sample in enumerate(tbar):
            start_time = time.time()
            sample_cuda = tocuda(sample)
            disp_prediction = model(sample_cuda["imgs"], sample_cuda["proj_matrices"],
                        sample_cuda["depth_min"], sample_cuda["depth_max"], test_mode=True)

            b = sample_cuda["depth_min"].shape[0]

            inverse_depth_min = (1.0 / sample_cuda["depth_min"]).view(b, 1, 1, 1)
            inverse_depth_max = (1.0 / sample_cuda["depth_max"]).view(b, 1, 1, 1)

            depth_prediction = depth_unnormalization(disp_prediction, inverse_depth_min, inverse_depth_max)
            depth_prediction = tensor2numpy(depth_prediction.float())
            del sample_cuda, disp_prediction
            tbar.set_description('Iter {}/{}, time = {:.3f}'.format(batch_idx, len(TestImgLoader), time.time() - start_time))
            filenames = sample["filename"]

            # save depth maps and confidence maps
            for filename, depth_est in zip(filenames, depth_prediction):
                depth_filename = os.path.join(args.outdir, filename.format('depth_est', '.pfm'))
                os.makedirs(depth_filename.rsplit('/', 1)[0], exist_ok=True)
                # save depth maps
                depth_est = np.squeeze(depth_est, 0)
                save_pfm(depth_filename, depth_est)

# project the reference point cloud into the source view, then project back
def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    ## step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    # reference 3D space
    xyz_ref = np.matmul(np.linalg.inv(intrinsics_ref),
                        np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1]))
    # source 3D space
    xyz_src = np.matmul(np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)),
                        np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]
    # source view x, y
    K_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)
    # mask = sampled_depth_src > 0

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = np.matmul(np.linalg.inv(intrinsics_src),
                        np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))
    # reference 3D space
    xyz_reprojected = np.matmul(np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)),
                                np.vstack((xyz_src, np.ones_like(x_ref))))[:3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / (K_xyz_reprojected[2:3]+1e-6)
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src


def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src, thre1, thre2):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(depth_ref,
                                                                                                 intrinsics_ref,
                                                                                                 extrinsics_ref,
                                                                                                 depth_src,
                                                                                                 intrinsics_src,
                                                                                                 extrinsics_src)
    # check |p_reproj-p_1| < 1
    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| / d_1 < 0.01
    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref
    masks=[]
    for i in range(2,11):
        mask = np.logical_and(dist < i/thre1, relative_depth_diff < i/thre2)
        masks.append(mask)
    depth_reprojected[~mask] = 0

    return masks, mask, depth_reprojected, x2d_src, y2d_src


def filter_depth(scan_folder, out_folder, plyfilename, geo_pixel_thres, geo_depth_thres, photo_thres, img_wh, geo_mask_thres=3):
    # the pair file
    pair_file = os.path.join(scan_folder, "pair.txt")
    # for the final point cloud
    vertexs = []
    vertex_colors = []

    pair_data = read_pair_file(pair_file)
    nviews = len(pair_data)

    thre_left = -2
    thre_right = 2
    total_iter = 10
    for iter in range(total_iter):
        thre = (thre_left + thre_right) / 2
        print(f"{iter} {10 ** thre}")
        depth_est_averaged = []
        geo_mask_all = []
        # for each reference view and the corresponding source views
        for ref_view, src_views in pair_data:
            # load the camera parameters
            ref_intrinsics, ref_extrinsics = read_camera_parameters(
                os.path.join(scan_folder, 'cams_1/{:0>8}_cam.txt'.format(ref_view)))
            ref_img, original_h, original_w = read_img(os.path.join(scan_folder, 'images/{:0>8}.jpg'.format(ref_view)), img_wh)
            ref_intrinsics[0] *= img_wh[0]/original_w
            ref_intrinsics[1] *= img_wh[1]/original_h
            # load the estimated depth of the reference view
            ref_depth_est = read_pfm(os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(ref_view)))[0]
            ref_depth_est = np.squeeze(ref_depth_est, 2)

            all_srcview_depth_ests = []
            # compute the geometric mask
            geo_mask_sum = 0
            geo_mask_sums=[]
            n = 1 + len(src_views)
            ct = 0
            for src_view in src_views:
                ct = ct + 1
                # camera parameters of the source view
                src_intrinsics, src_extrinsics = read_camera_parameters(
                    os.path.join(scan_folder, 'cams_1/{:0>8}_cam.txt'.format(src_view)))
                _, original_h, original_w = read_img(os.path.join(scan_folder, 'images/{:0>8}.jpg'.format(src_view)), img_wh)
                src_intrinsics[0] *= img_wh[0]/original_w
                src_intrinsics[1] *= img_wh[1]/original_h

                # the estimated depth of the source view
                src_depth_est = read_pfm(os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(src_view)))[0]
                

                masks, geo_mask, depth_reprojected, _, _ = check_geometric_consistency(ref_depth_est, ref_intrinsics, ref_extrinsics,
                                                                        src_depth_est,
                                                                        src_intrinsics, src_extrinsics, 10 ** thre * 4, 10 ** thre * 1300)
                if (ct==1):
                    for i in range(2,n):
                        geo_mask_sums.append(masks[i-2].astype(np.int32))
                else:
                    for i in range(2,n):
                        geo_mask_sums[i-2]+=masks[i-2].astype(np.int32)

                geo_mask_sum+=geo_mask.astype(np.int32)
                all_srcview_depth_ests.append(depth_reprojected)

            geo_mask=geo_mask_sum>=n
            for i in range (2,n):
                geo_mask=np.logical_or(geo_mask,geo_mask_sums[i-2]>=i)

            depth_est_averaged.append((sum(all_srcview_depth_ests) + ref_depth_est) / (geo_mask_sum + 1))
            geo_mask_all.append(np.mean(geo_mask))
            final_mask = geo_mask

            if iter == total_iter - 1:
                os.makedirs(os.path.join(out_folder, "mask"), exist_ok=True)
                save_mask(os.path.join(out_folder, "mask/{:0>8}_geo.png".format(ref_view)), geo_mask)
                save_mask(os.path.join(out_folder, "mask/{:0>8}_final.png".format(ref_view)), final_mask)

                print("processing {}, ref-view{:0>2}, geo_mask:{:3f} final_mask: {:3f}".format(scan_folder, ref_view,
                                                                        geo_mask.mean(), final_mask.mean()))

                if args.display:
                    cv2.imshow('ref_img', ref_img[:, :, ::-1])
                    cv2.imshow('ref_depth', ref_depth_est / np.max(ref_depth_est))
                    cv2.imshow('ref_depth * geo_mask', ref_depth_est * geo_mask.astype(np.float32) / np.max(ref_depth_est))
                    cv2.imshow('ref_depth * mask', ref_depth_est * final_mask.astype(np.float32) / np.max(ref_depth_est))
                    cv2.waitKey(0)

                height, width = depth_est_averaged[-1].shape[:2]
                x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))

                valid_points = final_mask
                # print("valid_points", valid_points.mean())
                x, y, depth = x[valid_points], y[valid_points], depth_est_averaged[-1][valid_points]
                
                color = ref_img[valid_points]
                xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics),
                                    np.vstack((x, y, np.ones_like(x))) * depth)
                xyz_world = np.matmul(np.linalg.inv(ref_extrinsics),
                                    np.vstack((xyz_ref, np.ones_like(x))))[:3]
                vertexs.append(xyz_world.transpose((1, 0)))
                vertex_colors.append((color * 255).astype(np.uint8))
        if np.mean(geo_mask_all) >= 0.25:
            thre_left = thre
        else:
            thre_right = thre
    vertexs = np.concatenate(vertexs, axis=0)
    vertex_colors = np.concatenate(vertex_colors, axis=0)
    vertexs = np.array([tuple(v) for v in vertexs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write(plyfilename)
    print("saving the final model to", plyfilename)


if __name__ == '__main__':
    save_depth()
    if args.dataset=="dtu_yao_eval":
        with open(args.testlist) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        for scan in scans:
            scan_id = int(scan[4:])
            scan_folder = os.path.join(args.testpath, scan)
            out_folder = os.path.join(args.outdir, scan)
            filter_depth(scan_folder, out_folder, os.path.join(args.outdir, 'igev_mvs{:0>3}_l3.ply'.format(scan_id)), 
                        args.geo_pixel_thres, args.geo_depth_thres, args.photo_thres, img_wh, 4)
    elif args.dataset=="tanks":
        # intermediate dataset
        if args.split == "intermediate":
            scans = ['Family', 'Francis', 'Horse', 'Lighthouse',
                    'M60', 'Panther', 'Playground', 'Train']
            geo_mask_thres = {'Family': 5,
                                'Francis': 6,
                                'Horse': 5,
                                'Lighthouse': 6,
                                'M60': 5,
                                'Panther': 5,
                                'Playground': 5,
                                'Train': 5}

            for scan in scans:
                scan_folder = os.path.join(args.testpath, args.split, scan)
                out_folder = os.path.join(args.outdir, scan)

                filter_depth(scan_folder, out_folder, os.path.join(args.outdir, scan + '.ply'), 
                    args.geo_pixel_thres, args.geo_depth_thres, args.photo_thres, img_wh, geo_mask_thres[scan])

        # advanced dataset
        elif args.split == "advanced":
            scans = ['Auditorium', 'Ballroom', 'Courtroom',
                    'Museum', 'Palace', 'Temple']
            geo_mask_thres = {'Auditorium': 3,
                                'Ballroom': 4,
                                'Courtroom': 4,
                                'Museum': 4,
                                'Palace': 5,
                                'Temple': 4}

            for scan in scans:
                scan_folder = os.path.join(args.testpath, args.split, scan)
                out_folder = os.path.join(args.outdir, scan)
                filter_depth(scan_folder, out_folder, os.path.join(args.outdir, scan + '.ply'), 
                    args.geo_pixel_thres, args.geo_depth_thres, args.photo_thres, img_wh, geo_mask_thres[scan])

    elif args.dataset=="eth3d":
        if args.split == "test":
            scans = ['botanical_garden', 'boulders', 'bridge', 'door',
                    'exhibition_hall', 'lecture_room', 'living_room', 'lounge',
                    'observatory', 'old_computer', 'statue', 'terrace_2']
            
            geo_mask_thres = {'botanical_garden':1,  # 30 images, outdoor
                    'boulders':1, # 26 images, outdoor
                    'bridge':2,  # 110 images, outdoor
                    'door':2, # 6 images, indoor
                    'exhibition_hall':2,  # 68 images, indoor
                    'lecture_room':2, # 23 images, indoor
                    'living_room':2, # 65 images, indoor
                    'lounge':1,# 10 images, indoor
                    'observatory':2, # 27 images, outdoor
                    'old_computer':2, # 54 images, indoor
                    'statue':2,  # 10 images, indoor
                    'terrace_2':2 # 13 images, outdoor
                    }
            for scan in scans:
                start_time = time.time()
                scan_folder = os.path.join(args.testpath, scan)
                out_folder = os.path.join(args.outdir, scan)
                filter_depth(scan_folder, out_folder, os.path.join(args.outdir, scan + '.ply'), 
                            args.geo_pixel_thres, args.geo_depth_thres, args.photo_thres, img_wh, geo_mask_thres[scan]) 
                print('scan: '+scan+' time = {:3f}'.format(time.time() - start_time))

        elif args.split == "train":
            scans = ['courtyard', 'delivery_area', 'electro', 'facade',
                    'kicker', 'meadow', 'office', 'pipes', 'playground',
                    'relief', 'relief_2', 'terrace', 'terrains']

            geo_mask_thres = {'courtyard':1,  # 38 images, outdoor
                    'delivery_area':2, # 44 images, indoor
                    'electro':1,  # 45 images, outdoor
                    'facade':2, # 76 images, outdoor
                    'kicker':1,  # 31 images, indoor
                    'meadow':1, # 15 images, outdoor
                    'office':1, # 26 images, indoor
                    'pipes':1,# 14 images, indoor
                    'playground':1, # 38 images, outdoor
                    'relief':1, # 31 images, indoor
                    'relief_2':1, # 31 images, indoor
                    'terrace':1,  # 23 images, outdoor
                    'terrains':2 # 42 images, indoor
                    }

            for scan in scans:
                start_time = time.time()
                scan_folder = os.path.join(args.testpath, scan)
                out_folder = os.path.join(args.outdir, scan)
                filter_depth(scan_folder, out_folder, os.path.join(args.outdir, scan + '.ply'), 
                            args.geo_pixel_thres, args.geo_depth_thres, args.photo_thres, img_wh, geo_mask_thres[scan])   
                print('scan: '+scan+' time = {:3f}'.format(time.time() - start_time))
    else:
        filter_depth(args.testpath, args.outdir, os.path.join(args.outdir, 'custom.ply'), 
                    args.geo_pixel_thres, args.geo_depth_thres, args.photo_thres, img_wh, geo_mask_thres=3) 
