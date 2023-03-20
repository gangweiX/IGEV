import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import random
import time
from torch.utils.tensorboard import SummaryWriter
from datasets import find_dataset_def
from core.igev_mvs import IGEVMVS
from core.submodule import depth_normalization, depth_unnormalization
from utils import *
import sys
import datetime
from tqdm import tqdm

cudnn.benchmark = True



parser = argparse.ArgumentParser(description='IterMVStereo for high-resolution multi-view stereo')
parser.add_argument('--mode', default='train', help='train or val', choices=['train', 'val'])

parser.add_argument('--dataset', default='dtu_yao', help='select dataset')
parser.add_argument('--trainpath', default='/data/DTU_data/dtu_train/', help='train datapath')
parser.add_argument('--valpath', help='validation datapath')
parser.add_argument('--trainlist', default='./lists/dtu/train.txt', help='train list')
parser.add_argument('--vallist', default='./lists/dtu/val.txt', help='validation list')
parser.add_argument('--maxdisp', default=256)

parser.add_argument('--epochs', type=int, default=32, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--wd', type=float, default=.00001, help='weight decay')

parser.add_argument('--batch_size', type=int, default=6, help='train batch size')
parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
parser.add_argument('--logdir', default='./checkpoints/', help='the directory to save checkpoints/logs')
parser.add_argument('--resume', action='store_true', help='continue to train the model')
parser.add_argument('--regress', action='store_true', help='train the regression and confidence')
parser.add_argument('--small_image', action='store_true', help='train with small input as 640x512, otherwise train with 1280x1024')

parser.add_argument('--summary_freq', type=int, default=20, help='print and summary frequency')
parser.add_argument('--save_freq', type=int, default=1, help='save checkpoint frequency')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
parser.add_argument('--iteration', type=int, default=22, help='num of iteration of GRU')

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

def sequence_loss(disp_preds, disp_init_pred, depth_gt, mask, depth_min, depth_max, loss_gamma=0.9):
    """ Loss function defined over sequence of depth predictions """
    cross_entropy = nn.BCEWithLogitsLoss()
    n_predictions = len(disp_preds)
    assert n_predictions >= 1
    loss = 0.0
    mask = mask > 0.5
    batch, _, height, width = depth_gt.size()
    inverse_depth_min = (1.0 / depth_min).view(batch, 1, 1, 1)
    inverse_depth_max = (1.0 / depth_max).view(batch, 1, 1, 1)

    normalized_disp_gt = depth_normalization(depth_gt, inverse_depth_min, inverse_depth_max)
    loss += 1.0 * F.l1_loss(disp_init_pred[mask], normalized_disp_gt[mask], reduction='mean')

    if args.iteration != 0:
        for i in range(n_predictions):
            adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
            i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
            loss += i_weight * F.l1_loss(disp_preds[i][mask], normalized_disp_gt[mask], reduction='mean')

    return loss

# parse arguments and check
args = parser.parse_args()
if args.resume: # store_true means set the variable as "True"
    assert args.mode == "train"
    assert args.loadckpt is None
if args.valpath is None:
    args.valpath = args.trainpath

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

if args.mode == "train":
    if not os.path.isdir(args.logdir):
        os.mkdir(args.logdir)

    current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    print("current time", current_time_str)

    print("creating new summary file")
    logger = SummaryWriter(args.logdir)

print("argv:", sys.argv[1:])
print_args(args)

# dataset, dataloader
MVSDataset = find_dataset_def(args.dataset)

train_dataset = MVSDataset(args.trainpath, args.trainlist, "train", 5, robust_train=True)
test_dataset = MVSDataset(args.valpath, args.vallist, "val", 5,  robust_train=False)

TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=4, drop_last=True)
TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=4, drop_last=False)

# model, optimizer
model = IGEVMVS(args)
if args.mode in ["train", "val"]:
    model = nn.DataParallel(model)
model.cuda()
model_loss = sequence_loss
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd, eps=1e-8)

# load parameters
start_epoch = 0
if (args.mode == "train" and args.resume) or (args.mode == "val" and not args.loadckpt):
    saved_models = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
    saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # use the latest checkpoint file
    loadckpt = os.path.join(args.logdir, saved_models[-1])
    print("resuming", loadckpt)
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'], strict=False)
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch'] + 1
elif args.loadckpt:
    # load checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'], strict=False)
print("start at epoch {}".format(start_epoch))
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


# main function
def train(args):
    total_steps = len(TrainImgLoader) * args.epochs + 100
    lr_scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, total_steps, pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')

    for epoch_idx in range(start_epoch, args.epochs):
        print('Epoch {}:'.format(epoch_idx))
        global_step = len(TrainImgLoader) * epoch_idx

        # training
        tbar = tqdm(TrainImgLoader)
        for batch_idx, sample in enumerate(tbar):
            start_time = time.time()
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0
            scaler = GradScaler(enabled=True)
            loss, scalar_outputs = train_sample(args, sample, detailed_summary=do_summary, scaler=scaler)
            if do_summary:
                save_scalars(logger, 'train', scalar_outputs, global_step)
            del scalar_outputs

            tbar.set_description(
                'Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}'.format(epoch_idx, args.epochs, batch_idx, len(TrainImgLoader), loss, time.time() - start_time))

        lr_scheduler.step()

        # checkpoint
        if (epoch_idx + 1) % args.save_freq == 0:
            torch.save({
                'model': model.state_dict()},
                "{}/model_{:0>6}.ckpt".format(args.logdir, epoch_idx))
        torch.cuda.empty_cache()
        # testing
        avg_test_scalars = DictAverageMeter()
        tbar = tqdm(TestImgLoader)
        for batch_idx, sample in enumerate(tbar):
            start_time = time.time()
            global_step = len(TestImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs = test_sample(args, sample, detailed_summary=do_summary)
            if do_summary:
                save_scalars(logger, 'test', scalar_outputs, global_step)
            avg_test_scalars.update(scalar_outputs)
            del scalar_outputs
            tbar.set_description('Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(epoch_idx, args.epochs, batch_idx,
                                                                                     len(TestImgLoader), loss,
                                                                                     time.time() - start_time))
        save_scalars(logger, 'fulltest', avg_test_scalars.mean(), global_step)
        print("avg_test_scalars:", avg_test_scalars.mean())
        torch.cuda.empty_cache()

def test(args):
    avg_test_scalars = DictAverageMeter()
    for batch_idx, sample in enumerate(TestImgLoader):
        start_time = time.time()
        loss, scalar_outputs = test_sample(args, sample, detailed_summary=True)
        avg_test_scalars.update(scalar_outputs)
        del scalar_outputs
        print('Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(batch_idx, len(TestImgLoader), loss,
                                                                    time.time() - start_time))
        if batch_idx % 100 == 0:
            print("Iter {}/{}, test results = {}".format(batch_idx, len(TestImgLoader), avg_test_scalars.mean()))
    print("final", avg_test_scalars)


def train_sample(args, sample, detailed_summary=False, scaler=None):
    model.train()
    optimizer.zero_grad()
    sample_cuda = tocuda(sample)
    depth_gt = sample_cuda["depth"] 
    mask = sample_cuda["mask"]      
    depth_gt_0 = depth_gt['level_0']
    mask_0 = mask['level_0']
    depth_gt_1 = depth_gt['level_2']
    mask_1 = mask['level_2']

    disp_init, disp_predictions = model(sample_cuda["imgs"], sample_cuda["proj_matrices"],
                    sample_cuda["depth_min"], sample_cuda["depth_max"])

    loss = model_loss(disp_predictions, disp_init, depth_gt_0, mask_0, sample_cuda["depth_min"], sample_cuda["depth_max"])
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

    inverse_depth_min = (1.0 / sample_cuda["depth_min"]).view(args.batch_size, 1, 1, 1)
    inverse_depth_max = (1.0 / sample_cuda["depth_max"]).view(args.batch_size, 1, 1, 1)
    depth_init = depth_unnormalization(disp_init, inverse_depth_min, inverse_depth_max)

    depth_predictions = []
    for disp in disp_predictions:
        depth_predictions.append(depth_unnormalization(disp, inverse_depth_min, inverse_depth_max))

    scalar_outputs = {"loss": loss}
    scalar_outputs["abs_error_initial"] = AbsDepthError_metrics(depth_init, depth_gt_0, mask_0 > 0.5)
    scalar_outputs["thres1mm_initial"] = Thres_metrics(depth_init, depth_gt_0, mask_0 > 0.5, 1)
    scalar_outputs["abs_error_final_full"] = AbsDepthError_metrics(depth_predictions[-1], depth_gt_0, mask_0 > 0.5)
    return tensor2float(loss), tensor2float(scalar_outputs)


@make_nograd_func
def test_sample(args, sample, detailed_summary=True):
    model.eval()
    sample_cuda = tocuda(sample)
    depth_gt = sample_cuda["depth"] 
    mask = sample_cuda["mask"]      
    depth_gt_0 = depth_gt['level_0']
    mask_0 = mask['level_0']
    depth_gt_1 = depth_gt['level_2']
    mask_1 = mask['level_2']

    disp_init, disp_predictions = model(sample_cuda["imgs"], sample_cuda["proj_matrices"],
                    sample_cuda["depth_min"], sample_cuda["depth_max"])

    loss = model_loss(disp_predictions, disp_init, depth_gt_0, mask_0, sample_cuda["depth_min"], sample_cuda["depth_max"])


    inverse_depth_min = (1.0 / sample_cuda["depth_min"]).view(sample_cuda["depth_min"].size()[0], 1, 1, 1)
    inverse_depth_max = (1.0 / sample_cuda["depth_max"]).view(sample_cuda["depth_max"].size()[0], 1, 1, 1)
    depth_init = depth_unnormalization(disp_init, inverse_depth_min, inverse_depth_max)

    depth_predictions = []
    for disp in disp_predictions:
        depth_predictions.append(depth_unnormalization(disp, inverse_depth_min, inverse_depth_max))

    scalar_outputs = {"loss": loss}
    scalar_outputs["abs_error_initial"] = AbsDepthError_metrics(depth_init, depth_gt_0, mask_0 > 0.5)
    scalar_outputs["thres1mm_initial"] = Thres_metrics(depth_init, depth_gt_0, mask_0 > 0.5, 1)
    scalar_outputs["abs_error_final_full"] = AbsDepthError_metrics(depth_predictions[-1], depth_gt_0, mask_0 > 0.5)
    return tensor2float(loss), tensor2float(scalar_outputs)


if __name__ == '__main__':
    if args.mode == "train":
        train(args)
    elif args.mode == "val":
        test(args)
    
