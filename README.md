# IGEV-Stereo & IGEV-MVS
This repository contains the source code for our paper:

Iterative Geometry Encoding Volume for Stereo Matching<br/>
CVPR 2023 <br/>
Gangwei Xu, Xianqi Wang, Xiaohuan Ding, Xin Yang<br/>

<img src="IGEV-Stereo/IGEV-Stereo.png">

## Demos
Pretrained models can be downloaded from [google drive](https://drive.google.com/drive/folders/1SsMHRyN7808jDViMN1sKz1Nx-71JxUuz?usp=share_link)

You can demo a trained model on pairs of images. To predict stereo for Middlebury, run
```
python demo.py --restore_ckpt ./pretrained_models/sceneflow/sceneflow.pth
```

<img src="IGEV-Stereo/demo-imgs.png" width="90%">

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
* [Middlebury](https://vision.middlebury.edu/stereo/data/)
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
```


## Evaluation

To evaluate a trained model on a test set (e.g. Scene Flow), run
```Shell
python evaluate_stereo.py --restore_ckpt ./pretrained_models/sceneflow/sceneflow.pth --dataset sceneflow
```

## Training

To train on Scene Flow, run

```Shell
python train_stereo.py
```

To train on KITTI, run
```Shell
python train_stereo.py --restore_ckpt ./pretrained_models/sceneflow/sceneflow.pth --dataset kitti
```

## Submission

For submission to the KITTI benchmark, run
```Shell
python save_disp.py
```

# Acknowledgements

This project is heavily based on [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo), We thank the original authors for their excellent work.

