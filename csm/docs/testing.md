## Eval keypoint transfer on birds
```
python -m csm.benchmark.csm.kp_transfer --name=csm_bird_net --n_data_workers=4 --dataset=cub  --display_port=8094 --scale_bias=0.75 --num_train_epochs=200
```


## Eval keypoint transfer on cars
```
python -m csm.benchmark.csm.kp_transfer --name=csm_car_net --n_data_workers=4 --dataset=p3d --p3d_class=car --display_port=8094 --scale_bias=0.5 --num_train_epochs=200
```


# Generate APK Plots
```
CSM_ROOT=/home/nileshk/csm_root/
cd CSM_ROOT/csm/pr_plots/
python pr_plots.py csm_bird_net
```

