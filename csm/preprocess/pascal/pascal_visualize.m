% Visualize images and corresponding rotated models?

p3d_dir = '/nfs.yoda/nileshk/CorrespNet/datasets/PASCAL3D+_release1.1/';
category = 'aeroplane';
images_dir =  fullfile(p3d_dir, 'Images');
basedir = fullfile(pwd, '..', '..');
seg_kp_dir = fullfile(basedir, 'cachedir', 'pascal', 'segkps');
addpath('../sfm');
img_anno_dir = fullfile(basedir, 'cachedir', 'p3d', 'data');
sfm_anno_dir = fullfile(basedir, 'cachedir', 'p3d', 'sfm');

kps = load(fullfile(img_anno_dir, strcat(category ,'_kps.mat')));
all_data = load(fullfile(img_anno_dir, strcat(category , '_all.mat')));

sfm_data = load(fullfile(sfm_anno_dir,  strcat(category , '_all.mat')));
mean_model = sfm_data.S;
index = [13];
% kp_names = {};
kp_names = kps.kp_names;
for i=1:size(index,2)
    close all;
    id = index(i);
    rel_path = all_data.images(id).rel_path;
    image_path = fullfile(images_dir, rel_path);
    sfm = sfm_data.sfm_anno(id);
    figure(); imshow(image_path);
    % figure(); show3dModel(mean_model, kp_names, 'convex_hull') 
    model = (sfm.rot*mean_model);
    figure(); show3dModel(model, kp_names, 'convex_hull')
end
'done'

