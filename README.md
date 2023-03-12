# IGEV-Stereo & IGEV-MVS
This repository contains the source code for our paper:

Iterative Geometry Encoding Volume for Stereo Matching<br/>
CVPR 2023 <br/>
Gangwei Xu, Xianqi Wang, Xiaohuan Ding, Xin Yang<br/>

<img src="IGEV-Stereo/IGEV-Stereo.png">


#How to Use

## Environment
* NVIDIA RTX 3090
* Python 3.8
* Pytorch 1.12

## Install

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
