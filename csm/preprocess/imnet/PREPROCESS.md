
## Create Masks using MaskRCNN
Running caffe2 detectron.

cd /home/nileshk/CorrespNet/icn/preprocess/seg_masks
sh caffe.sh
source caffe_sing.bashrc
export detectron_export_path.bashrc


```
python infer_simple.py     --cfg /home/nileshk/caffe2/detectron/configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml     --output-dir ../../../videos/video2/masks/    --image-ext jpg     --wts https://dl.fbaipublicfiles.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl    ../../../videos/video2/frames/

```


## Create UV frames
python -m icn.demo.model_demo --name=feb8_birds_depth_uv_vgg16_feat_flip_train_ent0pt05_rot_reg_0pt5_learn_mask --outdir=videos/video6/uv_contours/ --num_train_epoch=160  --n_data_workers=2 --videodir=videos/video6/ --pred_mask=True --use_alexnet_pp_feat=True --render_depth --render_mask --multiple_cam_hypo=True


## Incase you want to recreate visualizations.
python -m icn.demo.render_demo --name=feb8_birds_depth_uv_vgg16_feat_flip_train_ent0pt05_rot_reg_0pt5_learn_mask --outdir=videos/video2/uv_contours_4/ --num_train_epoch=160 --videodir=videos/video2/ --pred_mask=True --use_alexnet_pp_feat=True --render_depth --render_mask --multiple_cam_hypo=True --n_data_workers=2 --batch_size=24




## Convert Frames to Video
ffmpeg -start_number 0 -i uv_contour_frame%05d.jpg  ../out1.mp4
Run this on your desktop
