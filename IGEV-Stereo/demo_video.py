import sys
sys.path.append('core')
import cv2
import numpy as np
import glob
from pathlib import Path
from tqdm import tqdm
import torch
from PIL import Image
from igev_stereo import IGEVStereo
import os
import argparse
from utils.utils import InputPadder
torch.backends.cudnn.benchmark = True
half_precision = True


DEVICE = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='Iterative Geometry Encoding Volume for Stereo Matching and Multi-View Stereo (IGEV-Stereo)')
parser.add_argument('--restore_ckpt', help="restore checkpoint", default='./pretrained_models/kitti/kitti15.pth')
parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="/data/KITTI_raw/2011_09_26/2011_09_26_drive_0005_sync/image_02/data/*.png")
parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="/data/KITTI_raw/2011_09_26/2011_09_26_drive_0005_sync/image_03/data/*.png")
parser.add_argument('--mixed_precision', default=True, action='store_true', help='use mixed precision')
parser.add_argument('--valid_iters', type=int, default=16, help='number of flow-field updates during forward pass')
parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")

args = parser.parse_args()
model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])
model.load_state_dict(torch.load(args.restore_ckpt))
model = model.module
model.to(DEVICE)
model.eval()

left_images = sorted(glob.glob(args.left_imgs, recursive=True))
right_images = sorted(glob.glob(args.right_imgs, recursive=True))
print(f"Found {len(left_images)} images.")


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

if __name__ == '__main__':

    fps_list = np.array([])
    videoWrite = cv2.VideoWriter('./IGEV_Stereo.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (1242, 750))
    for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
        image1 = load_image(imfile1)
        image2 = load_image(imfile2)
        padder = InputPadder(image1.shape, divis_by=32)
        image1_pad, image2_pad = padder.pad(image1, image2)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=half_precision):
                disp = model(image1_pad, image2_pad, iters=16, test_mode=True)
                disp = padder.unpad(disp)
        end.record()
        torch.cuda.synchronize()
        runtime = start.elapsed_time(end)
        fps = 1000/runtime
        fps_list = np.append(fps_list, fps)
        if len(fps_list) > 5:
            fps_list = fps_list[-5:]
        avg_fps = np.mean(fps_list)
        print('Stereo runtime: {:.3f}'.format(1000/avg_fps))

        disp_np = (2*disp).data.cpu().numpy().squeeze().astype(np.uint8)
        disp_np = cv2.applyColorMap(disp_np, cv2.COLORMAP_PLASMA)
        image_np = np.array(Image.open(imfile1)).astype(np.uint8)       
        out_img = np.concatenate((image_np, disp_np), 0)
        cv2.putText(
            out_img,
            "%.1f fps" % (avg_fps),
            (10, image_np.shape[0]+30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('img', out_img)
        cv2.waitKey(1)
        videoWrite.write(out_img)
    videoWrite.release()
