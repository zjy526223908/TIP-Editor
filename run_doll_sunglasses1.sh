# doll sunglasses1

# train initial 3DGS, we provide trained 3dgs in https://drive.google.com/drive/folders/1DWcMyx5ab7PW3QkCwtJmkKUBL_feQxNW?usp=sharing
python train_3DGS.py --fp16 --workspace ./res_gaussion/colmap_doll  \
--min_opacity 0.001 \
--percent_dense 0.1 \
--iters 40000 \
--data_path /apdcephfs/private_piperzhuang/DreamEditor/data/colmap_doll  \
--R_path ./data/colmap_doll/Orient_R.npy  \
--initial_points  ./data/colmap_doll/sparse_points.ply \
--train_resolution_level 2 --eval_resolution_level 2 \
--data_type 'colmap'  --eval_interval 50

# sample views for stepwise personalization
python train_3DGS.py --fp16 --workspace ./res_gaussion/colmap_doll  \
--test --sample \
--bounding_box_path './data/3d_box/doll_glass_bbox.ply' \
--radius_list 1.3  --fovy 50 --phi_list  -45 -30 -15 0 15 30 45  --theta_list 60 75 90


# run stepwise personalization
# 1. scene personalization

export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export OUTPUT_DIR="./res_gaussion/colmap_doll/scene_personalization"
export image_root=./res_gaussion/colmap_doll/sample_views/rgb
python personalization/scene_personalization.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --enable_xformers_memory_efficient_attention \
  --instance_data_dir $image_root \
  --instance_prompt 'a photo of a <ktn> plush toy' \
  --instance_prompt 'a photo of a <ktn> plush toy wearing a sunglasses' \
  --with_prior_preservation \
  --class_data_dir './res_gaussion/colmap_doll/class_samples' \
  --class_prompt 'a photo of a plush toy' \
  --validation_prompt='a photo of a <ktn> plush toy wearing a sunglasses' \
  --instance_pos -1 \
  --output_dir $OUTPUT_DIR \
  --validation_images $image_root/1.3_75_-30.png \
    $image_root/1.3_75_0.png  \
    $image_root/1.3_75_30.png  \
  --max_train_steps=1000

# 2. novel content personalization
export MODEL_NAME="./res_gaussion/colmap_doll/scene_personalization/checkpoint-1000"
export OUTPUT_DIR="./res_gaussion/colmap_doll/content_personalization"

python personalization/content_personalization.py \
--pretrained_model_name_or_path $MODEL_NAME  \
--enable_xformers_memory_efficient_attention \
--instance_data_dir $image_root \
--instance_data_dir "./data/object/sunglasses1" \
--class_data_dir './res_gaussion/colmap_doll/class_samples' \
--instance_prompt 'a photo of a <ktn> plush toy' \
--instance_prompt 'a photo of a <pth> sunglasses' \
--class_prompt 'a photo of a plush toy' \
--validation_prompt "a photo of a <ktn> plush toy wearing a <pth> sunglasses"  \
--output_dir $OUTPUT_DIR \
--scene_frequency 200 \
--validation_images $image_root/1.3_75_-30.png \
  $image_root/1.3_75_0.png  \
  $image_root/1.3_75_30.png  \
--max_train_steps=500

# 3. Coarse Editing via SDS Loss
python train_coarse_editing.py  \
  --reset_points True \
  --editing_type 0 \
  --bbox_size_factor 1.25 \
  --batch_size 2 \
  --seed 1 --eval_interval 4   \
  --load_path  ./res_gaussion/colmap_doll/checkpoints/df_ep0625.pth \
  --bbox_path ./data/3d_box/doll_glass_bbox.ply \
  --text_global "a photo of a <ktn> plush toy wearing a <pth> sunglasses"    	\
  --text_local "a photo of a <pth> sunglasses"      	\
  --sd_path ./res_gaussion/colmap_doll/content_personalization/checkpoint-500  \
  --radius_range 1.3 1.3 --fovy_range 50 50 --pose_sample_strategy '360' \
  --phi_range -45 45 --theta_range 45 75   \
  --workspace ./res_gaussion/colmap_doll/coarse_editing_res

# 4. Pixel-Level Image Refinement
python train_refinement.py  \
  --eval_interval 20  \
  --iters 1000 \
  --seed 1   \
  --coarse_gs_path res_gaussion/colmap_doll/coarse_editing_res/checkpoints/None.pth \
  --initial_gs_path res_gaussion/colmap_doll/checkpoints/df_ep0625.pth  \
  --bbox_path ./data/3d_box/doll_glass_bbox.ply \
  --text_global "a photo of a <ktn> plush toy wearing a <pth> sunglasses" \
  --sd_path ./res_gaussion/colmap_doll/content_personalization/checkpoint-500  \
  --radius_range 1.3 1.3 --fovy_range 50 50 --pose_sample_strategy '360' \
  --phi_range -45 45 --theta_range 75 75 \
  --radius_list 1.3 \
  --phi_list -45 -30 -15 0 15 30 45   \
  --theta_list 45 60 \
  --workspace res_gaussion/colmap_doll/refine_res/
