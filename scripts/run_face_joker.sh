# face joker

# download trained 3dgs in https://drive.google.com/drive/folders/1DWcMyx5ab7PW3QkCwtJmkKUBL_feQxNW?usp=sharing
# sample views for stepwise personalization
python train_3DGS.py --fp16 --workspace ./res_gaussion/colmap_face  \
--test --sample \
--bounding_box_path './data/3d_box/face_head_bbox.ply' \
--radius_list 0.75  --fovy 50 --phi_list   -30 -15 0 15 30   --theta_list  90  --bbox_size_factor 0.75

# run stepwise personalization
# 1. scene personalization
export MODEL_NAME="/apdcephfs_cq11/share_1467498/home/zhuangjingyu/model_zoo/models--stabilityai--stable-diffusion-2-1-base/snapshots/5ede9e4bf3e3fd1cb0ef2f7a3fff13ee514fdf06"
export OUTPUT_DIR=./res_gaussion/colmap_face/scene_personalization
export image_root=./res_gaussion/colmap_face/sample_views/rgb
python personalization/scene_personalization.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --enable_xformers_memory_efficient_attention \
  --instance_data_dir $image_root \
  --instance_prompt 'a photo of a <ktn> man wearing a gray sweater in front of a wall' \
  --instance_prompt 'a photo of a <ktn> joker wearing a gray sweater in front of a wall' \
  --with_prior_preservation \
  --class_data_dir './res_gaussion/colmap_face/class_samples' \
  --class_prompt 'a photo of a man' \
  --validation_prompt='a photo of a <ktn> joker wearing a gray sweater in front of a wall' \
  --instance_pos -8 \
  --output_dir $OUTPUT_DIR \
  --validation_images $image_root/0.75_90_-15.png \
    $image_root/0.75_90_0.png  \
    $image_root/0.75_90_15.png  \
  --max_train_steps=1000

# 2. novel content personalization

export MODEL_NAME="./res_gaussion/colmap_face/scene_personalization/checkpoint-1000"
export OUTPUT_DIR="./res_gaussion/colmap_face/content_personalization"

python personalization/content_personalization.py \
--pretrained_model_name_or_path $MODEL_NAME  \
--enable_xformers_memory_efficient_attention \
--instance_data_dir $image_root \
--instance_data_dir "./data/object/joker1" \
--class_data_dir './res_gaussion/colmap_face/class_samples' \
--instance_prompt 'a photo of a <ktn> man wearing a gray sweater in front of a wall' \
--instance_prompt 'a photo of a <pth> joker' \
--class_prompt 'a photo of a man' \
--validation_prompt 'a photo of a <pth> joker wearing a gray sweater in front of a wall'  \
--output_dir $OUTPUT_DIR \
--scene_frequency 200 \
--validation_steps 100 \
--validation_images $image_root/0.75_90_-15.png \
  $image_root/0.75_90_0.png  \
  $image_root/0.75_90_15.png  \
--max_train_steps=500


# 3. Coarse Editing via SDS Loss
python train_coarse_editing.py  \
  --editing_type 1 \
  --iters 1000 \
  --bbox_size_factor 1. \
  --guidance_scale 5.0 \
  --batch_size 4   \
  --start_gamma 1.0 \
  --end_gamma 0.9 \
  --min_opacity 0.001 \
  --position_lr_init 0.0001 \
  --seed 1 --eval_interval 1   \
  --load_path  res_gaussion/colmap_face/checkpoints/None.pth \
  --bbox_path data/3d_box/face_head_bbox.ply \
  --text_global  'a photo of a <pth> joker wearing a gray sweater in front of a wall'   	\
  --text_local "a photo of a <pth> joker in a black background"     	\
  --sd_path res_gaussion/colmap_face/content_personalization/checkpoint-500  \
  --radius_range 0.75 0.75 --fovy_range 50 50 --pose_sample_strategy '360' \
  --phi_range -30 30 --theta_range 75 105   \
  --workspace ./res_gaussion/colmap_face/coarse_editing_res

# 4. Pixel-Level Image Refinement
python train_refinement.py  \
  --eval_interval 20  \
  --iters 1000 \
  --seed 1   \
  --coarse_gs_path res_gaussion/colmap_face/coarse_editing_res/checkpoints/None.pth \
  --initial_gs_path res_gaussion/colmap_face/checkpoints/None.pth  \
  --bbox_path ./data/3d_box/face_head_bbox.ply \
  --text_global 'a photo of a <pth> joker wearing a gray sweater in front of a wall'   \
  --sd_path res_gaussion/colmap_face/content_personalization/checkpoint-500  \
  --radius_range 0.75 0.75 --fovy_range 50 50 --pose_sample_strategy '360' \
  --phi_range -30 30 --theta_range 75 105   \
  --radius_list 0.75 \
  --phi_list -30 -15 0 15 30   \
  --theta_list 75 90 105 \
  --workspace res_gaussion/colmap_face/refine_res/

