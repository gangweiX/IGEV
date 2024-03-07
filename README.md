# IGEV-Stereo & IGEV-MVS (CVPR 2023)

This repository contains the source code for our paper:

[Iterative Geometry Encoding Volume for Stereo Matching](https://arxiv.org/pdf/2303.06615.pdf)<br/>
Gangwei Xu, Xianqi Wang, Xiaohuan Ding, Xin Yang<br/>

<img src="IGEV-Stereo/IGEV-Stereo.png">

## Demos
Pretrained models can be downloaded from [google drive](https://drive.google.com/drive/folders/1SsMHRyN7808jDViMN1sKz1Nx-71JxUuz?usp=share_link)

We assume the downloaded pretrained weights are located under the pretrained_models directory.

You can demo a trained model on pairs of images. To predict stereo for Middlebury, run
```
python demo.py --restore_ckpt ./pretrained_models/sceneflow/sceneflow.pth
```

<img src="IGEV-Stereo/demo-imgs.png" width="90%">

## Comparison with RAFT-Stereo

| Method | KITTI 2012 <br> (3-noc) | KITTI 2015 <br> (D1-all) | Memory (G) | Runtime (s) |
|:-:|:-:|:-:|:-:|:-:|
| RAFT-Stereo | 1.30 % | 1.82 % | 1.02 | 0.38 |
| IGEV-Stereo | 1.12 % | 1.59 % | 0.66 | 0.18 |


## Environment
* NVIDIA RTX 3090
* Python 3.8
* Pytorch 1.12

### Create a virtual environment and activate it.

```
conda create -n IGEV_Stereo python=3.8
conda activate IGEV_Stereo
```
### Dependencies

```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -c nvidia
pip install opencv-python
pip install scikit-image
pip install tensorboard
pip install matplotlib 
pip install tqdm
pip install timm==0.5.4
```

## Required Data
To evaluate/train IGEV-Stereo, you will need to download the required datasets. 
* [Scene Flow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
* [KITTI](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)
* [Middlebury](https://vision.middlebury.edu/stereo/submit3/)
* [ETH3D](https://www.eth3d.net/datasets#low-res-two-view-test-data)

By default `stereo_datasets.py` will search for the datasets in these locations. 

```
├── /data
    ├── sceneflow
        ├── frames_finalpass
        ├── disparity
    ├── KITTI
        ├── KITTI_2012
            ├── training
            ├── testing
            ├── vkitti
        ├── KITTI_2015
            ├── training
            ├── testing
            ├── vkitti
    ├── Middlebury
        ├── trainingH
        ├── trainingH_GT
    ├── ETH3D
        ├── two_view_training
        ├── two_view_training_gt
    ├── DTU_data
        ├── dtu_train
        ├── dtu_test
```

## Evaluation

To evaluate on Scene Flow or Middlebury or ETH3D, run

```Shell
python evaluate_stereo.py --restore_ckpt ./pretrained_models/sceneflow/sceneflow.pth --dataset sceneflow
```
or
```Shell
python evaluate_stereo.py --restore_ckpt ./pretrained_models/sceneflow/sceneflow.pth --dataset middlebury_H
```
or
```Shell
python evaluate_stereo.py --restore_ckpt ./pretrained_models/sceneflow/sceneflow.pth --dataset eth3d
```

## Training

To train on Scene Flow, run

```Shell
python train_stereo.py --logdir ./checkpoints/sceneflow
```

To train on KITTI, run
```Shell
python train_stereo.py --logdir ./checkpoints/kitti --restore_ckpt ./pretrained_models/sceneflow/sceneflow.pth --train_datasets kitti
```

## Submission

For submission to the KITTI benchmark, run
```Shell
python save_disp.py
```

## MVS training and evaluation

To train on DTU, run

```Shell
python train_mvs.py
```

To evaluate on DTU, run

```Shell
python evaluate_mvs.py
```

## Citation

If you find our work useful in your research, please consider citing our paper:

```bibtex
@inproceedings{xu2023iterative,
  title={Iterative Geometry Encoding Volume for Stereo Matching},
  author={Xu, Gangwei and Wang, Xianqi and Ding, Xiaohuan and Yang, Xin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={21919--21928},
  year={2023}
}
```


# Acknowledgements

This project is heavily based on [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo), we thank the original authors for their excellent work.

