
## Download the CUB Data
1. Download CUB-200-2011 images.

```
wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz && tar -xf CUB_200_2011.tgz
```

2. Download our [annotation files](https://cmu.box.com/s/e8y71bjuwxrdrypy44zu3z8qexmnyqp4) and pre-computed UV parameterization outputs. Do this from the csm_root/csm/ directory, and this should make csm_root/csm/cachedir directory. This also contains weights from the pretrained models.

```
wget https://cmu.box.com/s/goter3meyi7rssbh2rujg36dfwygnc3f && unzip cachedir.zip
Alternate link.
wget https://umich.box.com/s/nncgnzht90s9dnahodel116hgykmmf6o && unzip cachedir.zip
```
<!--
```
wget https://cmu.box.com/s/e8y71bjuwxrdrypy44zu3z8qexmnyqp4 && tar -xf cachedir.tgz
```
-->

## Computing the UV parameterization
Please read the instruction here to create the above parametrization if you would like to recompute [here]()


## Model Training

### Train Birds
Without known pose
```
CODE_ROOT=/home/nileshk/csm_root/
cd $CODE_ROOT
python -m csm.experiments.csm.csp --name=csm_bird_net --n_data_workers=4 --dataset=cub  --display_port=8094 --scale_bias=0.75 --warmup_pose_iter=2000
```

With pose
```
CODE_ROOT=/home/nileshk/csm_root/
cd $CODE_ROOT
python -m csm.experiments.csm.csp --name=csm_bird_net_wpose --n_data_workers=4 --dataset=cub  --display_port=8094 --scale_bias=0.75 --multiple_cam_hypo=False --use_gt_quat=True --pred_cam=True 
```


### Train Cars
Without known pose
```
CODE_ROOT=/home/nileshk/csm_root/
cd $CODE_ROOT
python -m csm.experiments.csm.csp --name=csm_car_net --n_data_workers=4 --dataset=p3d --p3d_class=car  --display_port=8094 --scale_bias=1.0 --warmup_pose_iter=2000
```

With pose
```
CODE_ROOT=/home/nileshk/csm_root/
cd $CODE_ROOT
python -m csm.experiments.csm.csp --name=csm_car_net_wpose --n_data_workers=4 --dataset=p3d --p3d_class=car  --display_port=8094 --scale_bias=1.0 --multiple_cam_hypo=False --use_gt_quat=True --pred_cam=True --warmup_pose_iter=2000
```

### Train on Imagenet Classes
Download Imagenet data [here](https://cmu.box.com/s/9191n946eabqiaabe3owgchoq1buif5o) for categories in the paper.
```
imnet_class=horse
CODE_ROOT=/home/nileshk/csm_root/
cd $CODE_ROOT
python -m csm.experiments.csm.csp --name=csm_imnet_net_$imnet_class --n_data_workers=4 --dataset=imnet --imnet_class=$imnet_class  --display_port=8094 --scale_bias=1.0 --warmup_pose_iter=500
```
