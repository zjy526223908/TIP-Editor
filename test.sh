# doll
python train_3DGS.py --test --save_video \
  --radius_range 1.3 1.3 --fovy_range 50 50  \
  --phi_range -45 45 --theta_range 75 75 \
  --workspace ./res_edited/doll_sunglasses1

# doll_sunglasses2, doll_sunglasses3, doll_pattern1

python train_3DGS.py --test --save_video \
  --radius_range 1.3 1.3 --fovy_range 50 50  \
  --phi_range -45 45 --theta_range 45 45 \
  --workspace ./res_edited/doll_hat2

# horse
python train_3DGS.py --test --save_video \
  --radius_range 1.5 1.5 --fovy_range 50 50  \
  --phi_range -60 60 --theta_range 66 66 \
  --workspace ./res_edited/horse_white_giraffe

# horse_rare_giraffe, horse_robot, horse_robot, horse_ghost, horse_white_giraffe_sunglass2

# bear

python train_3DGS.py --test --save_video \
  --radius_range 1. 1. --fovy_range 50 50  \
  --phi_range -90 90 --theta_range 68 68 \
  --workspace res_edited/bear_dog/
#  bear_gold

# face
python train_3DGS.py --test --save_video \
  --radius_range 0.75 0.75 --fovy_range 50 50  \
  --phi_range -30 30 --theta_range 90 90 \
  --workspace res_edited/face_beard

# face_jacket, face_joker, face_style
