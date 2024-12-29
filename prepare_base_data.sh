#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

# SMPL Neutral model
echo -e "\nYou need to register at https://smplify.is.tue.mpg.de"
read -p "Username (SMPLify):" username
read -p "Password (SMPLify):" password
username=$(urle $username)
password=$(urle $password)

mkdir -p dataset/body_models/smpl
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smplify&resume=1&sfile=mpips_smplify_public_v2.zip' -O './dataset/body_models/smplify.zip' --no-check-certificate --continue
unzip dataset/body_models/smplify.zip -d dataset/body_models/smplify
mv dataset/body_models/smplify/smplify_public/code/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl dataset/body_models/smpl/SMPL_NEUTRAL.pkl
rm -rf dataset/body_models/smplify
rm -rf dataset/body_models/smplify.zip

# SMPL Male and Female model
echo -e "\nYou need to register at https://smpl.is.tue.mpg.de"
read -p "Username (SMPL):" username
read -p "Password (SMPL):" password
username=$(urle $username)
password=$(urle $password)

wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.0.0.zip' -O './dataset/body_models/smpl.zip' --no-check-certificate --continue
unzip dataset/body_models/smpl.zip -d dataset/body_models/smpl
mv dataset/body_models/smpl/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl dataset/body_models/smpl/SMPL_FEMALE.pkl
mv dataset/body_models/smpl/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl dataset/body_models/smpl/SMPL_MALE.pkl
rm -rf dataset/body_models/smpl/smpl
rm -rf dataset/body_models/smpl.zip

# body data
wget "https://1drv.ms/u/c/d70f26d613e83858/ESa0x1hfX85PlGUePYE_IFcB9fKnJg55sg5HA9rodzJoAQ?e=wjO1FN" -O 'dataset/body_models/coco_aug_dict.pth'
wget "https://1drv.ms/u/c/d70f26d613e83858/EZwpE1imYBZJqExM8oQW7GABxRnP5FrIIfFt522EBg2OkQ?e=xO6Bfe" -O 'dataset/body_models/J_regressor_feet.npy'
wget "https://1drv.ms/u/c/d70f26d613e83858/Ec53_98qGG1NlxjAB4RON0UB5D6EvhwVtZ8K1hD7DhDPKg?e=FjVrQk" -O 'dataset/body_models/J_regressor_h36m.npy'
wget "https://1drv.ms/u/c/d70f26d613e83858/EaJ8VZurhK1Ptiy83mxuhDEB3ENGgUNGByqjAD7J4P5HHw?e=AYnnaZ" -O 'dataset/body_models/J_regressor_wham.npy'
wget "https://1drv.ms/u/c/d70f26d613e83858/EWWd-dC1B9RCrZF5spwQU2UBHWaz1WA4UJr4a5eW6IVXIw?e=7C1NuA" -O 'dataset/body_models/smpl_mean_params.npz'

# marker data
mkdir -p dataset/marker
wget "https://1drv.ms/u/c/d70f26d613e83858/EahKjS6hWyFMmKJj-Yh_PVQBGGHgBWPwebV8XK805X0xVg?e=YracdI" -O 'dataset/marker/J_regressor_h36m_correct.npy'
wget "https://1drv.ms/u/c/d70f26d613e83858/EU5nYyNX_YlDrO2hE91-tzEBNvXDIyKlgPN3UlBg7t3jnw?e=5pwSNE" -O 'dataset/marker/marker_final.pth.tar'
wget "https://1drv.ms/u/c/d70f26d613e83858/EaZ2NAUVbpxPhU6RQUUsZQYBa3lsAGU6Zw4r8pTi6As8MQ?e=a3qPDq" -O 'dataset/marker/pose_hrnet_w48_384x288.pth'
wget "https://1drv.ms/u/c/d70f26d613e83858/EYVvgl56_O9Mvay7GqnQhH0B0jZp387uNczOl2aMEuFSkQ?e=futw3P" -O 'dataset/marker/smpl_indices.pkl'
wget "https://1drv.ms/u/c/d70f26d613e83858/EXxdwbagg11OhxpDHmCXWwEBvm7gOnq-Bt0zWitA17tXsg?e=7bGOwL" -O 'dataset/marker/vm_A_sym.npz'
wget "https://1drv.ms/u/c/d70f26d613e83858/EWHf0CKSB2lAnKJQly9Xs8UBPK_41bNWxjEuITwYno-BQA?e=4Syybr" -O 'dataset/marker/vm_B_sym.npz'
wget "https://1drv.ms/u/c/d70f26d613e83858/ERTqlVaJC51JinwFYccsH9gBOGJa7a7-6aox3iof0bxk1w?e=YNzBEa" -O 'dataset/marker/vm_info.npz'

# training and testing data
mkdir -p dataset/parsed_data
wget "https://1drv.ms/u/c/d70f26d613e83858/ETQNvedVtNNDidfspi7MzF0BCNHiREBy5yIuO3lKGW5tGw?e=a2sV2d" -O 'dataset/parsed_data/3dpw_vm_test_vit.pth'
wget "https://1drv.ms/u/c/d70f26d613e83858/EamTCGTi6stHhJNX2vfGR1oB9QOEohUkYGpRWax3jLlihg?e=rgKj39" -O 'dataset/parsed_data/3dpw_vm_train_vit.pth'
wget "https://1drv.ms/u/c/d70f26d613e83858/ET4D7yJ2_MZGkpUG6-lQ6AYBRihNUiWKcm-79-psNd7ZBQ?e=IGW2kz" -O 'dataset/parsed_data/3dpw_vm_val_vit.pth'
wget "https://1drv.ms/u/c/d70f26d613e83858/EYnAPQhaqINLoPT7V9OpmrYBFhBLd8HGDI5rxKmbhPwKbA?e=aCwbM8" -O 'dataset/parsed_data/amass.pth'
wget "https://1drv.ms/u/c/d70f26d613e83858/EVi5wGVZDJRJtig5MFN8jkIBZz6kkSORjVaXTqt-UhN4iQ?e=SEWkJM" -O 'dataset/parsed_data/h36m_vm_test_vit.pth'
wget "https://1drv.ms/u/c/d70f26d613e83858/EZWm5YEGi-hLp_WKhsGotGMBl2B7cXIHrmK1tcexYqf0EA?e=zMXESm" -O 'dataset/parsed_data/human36m_vm_train_tight_vit.pth'
wget "https://1drv.ms/u/c/d70f26d613e83858/EbPKlFvm7aNGmox28cYDMGABBVoytFvtes29bsdzZq0YQQ?e=2dTANm" -O 'dataset/parsed_data/mpii3d_vm_test_vit.pth'
wget "https://1drv.ms/u/c/d70f26d613e83858/EVVUecf6rsZOp0kDBEsK7qEBezIgiktaxqZSrCPKcapz6Q?e=LoyPqj" -O 'dataset/parsed_data/mpii3d_vm_train_vit.pth'
wget "https://1drv.ms/u/c/d70f26d613e83858/EX86VxDtZhBIo9AJqjGXASoBOtRJqDxrvsGnOjjQOgRt8g?e=vSmivL" -O 'dataset/parsed_data/surreal_vm_train_vit.pth'

# Checkpoints
mkdir checkpoints
wget "https://1drv.ms/u/c/d70f26d613e83858/EaGfFj1fOWpDpi3FXLhgbt8BY7B_NbFPQDYzAje6-SjzZw?e=iGctVz" -O 'checkpoints/pmce_plus_3dpw.pth.tar'
wget "https://1drv.ms/u/c/d70f26d613e83858/EY5ePCRcJeVOk5WFwTo3RPgBzWVpQH_SfXLLlVYM5rJQiA?e=Q8JNV2" -O 'checkpoints/pmce_plus_h36m.pth.tar'
wget "https://1drv.ms/u/c/d70f26d613e83858/EVje1dWfXt1OvaO_tAnZmsgBmg_EXSTO7hKflxx2t2ev9Q?e=uGZ9Pu" -O 'checkpoints/pmce_plus_mpii3d.pth.tar'
wget "https://1drv.ms/u/c/d70f26d613e83858/EbknBa1xte9Prn4LLpMnEGQBPCkf_mLFapkG56WhUzWPzg?e=mGXQI1" -O 'checkpoints/hmr2a.ckpt'
wget "https://1drv.ms/u/c/d70f26d613e83858/Eem7_1435CxHmx-DnoklgyYByiqo-p5i3CdI2Et0M43D0g?e=FH1tjP" -O 'checkpoints/yolov8x.pt'
wget "https://1drv.ms/u/c/d70f26d613e83858/EaYBOy7hU59Mh_nn07jNsqoBaWQ1kUlBUiGwxH9fslE71A?e=oGg39P" -O 'checkpoints/vitpose-h-multi-coco.pth'

# Demo video
mkdir examples
wget "https://1drv.ms/v/c/d70f26d613e83858/EcAI3fXKSeJMmj-064HZszIBoekxiROEk0f1uFr9Wj82wA?e=tnRZJv" -O 'examples/demo.mp4'