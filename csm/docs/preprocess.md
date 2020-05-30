## UV Parameterization for Template Shape
Do this if you need to create a UV parameterization for a new object class. Below the process is highlighted for the class car.
For classes bird, car, horse, sheep, cow, zebra the uv parametrization exists in the [cachedir]()

CSM_ROOT=/home/nileshk/csm_root/


```
cd  $CSM_ROOT/csm/preprocess/meshdeform/
```
Please edit the `startup.m` file with correct `class_name` and `output_map_file`. 
```
output_map_file='/home/nileshk/CorrespNet/csm_root/csm/cachedir/downloaded_models/car/car_mapping.mat'
class_name='car'
```
Now create mapping from `uv` to face ids.

```
python uv_to_vertex_map.py $CSM_ROOT/csm/cachedir/downloaded_models/car/car_mapping.mat  $CSM_ROOT/csm/csm/cachedir/shapenet/car/shape.mat  $CSM_ROOT/csm/csm/cachedir/downloaded_models/car/transform.mat
```

## Preprocess ImageNet for Pascal3D+ to dump masks using Mask-RCNN
Follow instructions [here](https://github.com/akanazawa/cmr/blob/p3d/preprocess/pascal/pascal.md)

## Preprocess Imagenet to dump masks using Mask-RCNN
Install detectron using instruction [here]()

We filtered images from ImageNet which were occuluded and truncated. Those annotations are in `*_quality.json` file. Edit `imageNetDir` and `sysnetId` with the appropriate values.

```
cd $CSM_ROOT/csm/preprocess/imagenet/
python extract_class_data.py
```





