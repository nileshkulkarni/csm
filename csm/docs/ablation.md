## Ablation on CUBS

### Training : Without known pose
CSM
```
CODE_ROOT=/home/nileshk/csm_root/
cd $CODE_ROOT
python -m csm.experiments.csm.csp --name=csm_bird_net --n_data_workers=4 --dataset=cub  --display_port=8094 --scale_bias=0.75
```

CSM - mask
```
python -m csm.experiments.csm.csp --name=csm_bird_net_wo_mask --n_data_workers=4 --dataset=cub  --display_port=8094 --scale_bias=0.75 --ignore_mask_gcc=True --ignore_mask_vis=True
```


CSM - vis
```
python -m csm.experiments.csm.csp --name=csm_bird_net_wo_vis --n_data_workers=4 --dataset=cub  --display_port=8094 --scale_bias=0.75 --depth_loss_wt=0.0
```


### Training : With pose
CSM w/ Pose
```
python -m csm.experiments.csm.csp --name=csm_bird_net_wpose --n_data_workers=4 --dataset=cub  --display_port=8094 --scale_bias=0.75 --multiple_cam_hypo=False --use_gt_quat=True --pred_cam=True
```

CSM w/ Pose - mask
```
python -m csm.experiments.csm.csp --name=csm_bird_net_wpose_wo_mask --n_data_workers=4 --dataset=cub  --display_port=8094 --scale_bias=0.75 --multiple_cam_hypo=False --use_gt_quat=True --pred_cam=True --ignore_mask_gcc=True --ignore_mask_vis=True
```


CSM w/ Pose - vis
```
python -m csm.experiments.csm.csp --name=csm_bird_net_wpose_wo_mask --n_data_workers=4 --dataset=cub  --display_port=8094 --scale_bias=0.75 --multiple_cam_hypo=False --use_gt_quat=True --pred_cam=True --depth_loss_wt=0.0
```

### Testing
CSM
```
python -m csm.benchmark.csm.kp_transfer --name=csm_bird_net --n_data_workers=4 --dataset=cub  --display_port=8094 --scale_bias=0.75 --num_train_epochs=200
```

CSM - mask
```
python -m csm.benchmark.csm.kp_transfer --name=csm_bird_net_wo_mask --n_data_workers=4 --dataset=cub --display_port=8094 --scale_bias=0.75 --num_train_epochs=200
```

CSM - vis
```
python -m csm.benchmark.csm.kp_transfer --name=csm_bird_net_wo_vis --n_data_workers=4 --dataset=cub --display_port=8094 --scale_bias=0.75 --num_train_epochs=200
```

CSM w/ Pose
```
python -m csm.benchmark.csm.kp_transfer --name=csm_bird_net_wpose --n_data_workers=4 --dataset=cub  --display_port=8094 --scale_bias=0.75 --num_train_epochs=200
```

CSM w/ Pose - mask
```
python -m csm.benchmark.csm.kp_transfer --name=csm_bird_net_wpose_wo_mask --n_data_workers=4 --dataset=cub  --display_port=8094 --scale_bias=0.75 --num_train_epochs=200
```

CSM w/ Pose - vis
```
python -m csm.benchmark.csm.kp_transfer --name=csm_bird_net_wpose_wo_vis --n_data_workers=4 --dataset=cub  --display_port=8094 --scale_bias=0.75 --num_train_epochs=200
```