################################################### 
# GCD Known Classes
###################################################
hostname
nvidia-smi

export CUDA_VISIBLE_DEVICES=3

############################
# All_Samples (proto_based False)
############################

python -m test  \
          --dataset_name 'aircraft' \
          --ckpt_dir 'Logs/ALGCD/All_Samples/aircraft-Acc-Final[67.0_65.4_67.9]-Best[67.1_65.5_67.9]/checkpoints' \
          --proto_based False

python -m test  \
          --dataset_name 'cifar100' \
          --ckpt_dir 'Logs/ALGCD/All_Samples/cifar100-Acc-Final[80.4_86.2_68.6]-Best[80.6_86.3_69.1]/checkpoints' \
          --proto_based False

python -m test  \
          --dataset_name 'cub' \
          --ckpt_dir 'Logs/ALGCD/All_Samples/cub-Acc-Final[85.4_79.5_88.4]-Best[85.7_79.1_89.0]/checkpoints' \
          --proto_based False

python -m test  \
          --dataset_name 'herbarium_19' \
          --ckpt_dir 'Logs/ALGCD/All_Samples/herbarium_19-Acc-Final[52.3_57.7_49.3]-Best[52.7_58.0_49.9]/checkpoints' \
          --proto_based False

python -m test  \
          --dataset_name 'imagenet_100' \
          --ckpt_dir 'Logs/ALGCD/All_Samples/imagenet_100-Acc-Final[95.3_97.5_94.2]-Best[95.3_97.5_94.2]/checkpoints' \
          --proto_based False

python -m test  \
          --dataset_name 'scars' \
          --ckpt_dir 'Logs/ALGCD/All_Samples/scars-Acc-Final[81.5_92.0_76.4]-Best[81.5_92.0_76.4]/checkpoints' \
          --proto_based False


############################
# Proto_Based (proto_based True)
############################

python -m test  \
          --dataset_name 'aircraft' \
          --ckpt_dir 'Logs/ALGCD/Proto_Based/aircraft-Acc-Final[66.0_62.8_67.6]-Best[66.4_64.6_67.3]/checkpoints' \
          --proto_based True

python -m test  \
          --dataset_name 'cifar100' \
          --ckpt_dir 'Logs/ALGCD/Proto_Based/cifar100-Acc-Final[79.1_84.5_68.2]-Best[79.1_84.5_68.2]/checkpoints' \
          --proto_based True

python -m test  \
          --dataset_name 'cub' \
          --ckpt_dir 'Logs/ALGCD/Proto_Based/cub-Acc-Final[84.5_77.8_87.9]-Best[84.9_79.9_87.4]/checkpoints' \
          --proto_based True

python -m test  \
          --dataset_name 'herbarium_19' \
          --ckpt_dir 'Logs/ALGCD/Proto_Based/herbarium_19-Acc-Final[48.0_56.3_43.6]-Best[49.7_58.2_45.1]/checkpoints' \
          --proto_based True

python -m test  \
          --dataset_name 'imagenet_100' \
          --ckpt_dir 'Logs/ALGCD/Proto_Based/imagenet_100-Acc-Final[93.5_97.2_91.6]-Best[93.7_96.9_92.1]/checkpoints' \
          --proto_based True

python -m test  \
          --dataset_name 'scars' \
          --ckpt_dir 'Logs/ALGCD/Proto_Based/scars-Acc-Final[80.9_92.1_75.6]-Best[80.9_91.6_75.8]/checkpoints' \
          --proto_based True
