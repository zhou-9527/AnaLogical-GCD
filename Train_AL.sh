###################################################
# GCD Known Classes
###################################################
hostname
nvidia-smi

export CUDA_VISIBLE_DEVICES=5

python -m AL_train   \
          --lr 0.1 \
          --dataset_name 'cub'

python -m AL_train   \
           --lr 0.1 \
           --dataset_name 'scars'

python -m AL_train   \
           --lr 0.1 \
           --dataset_name 'aircraft'

python -m AL_train  \
           --lr 0.1 \
           --grad_from_block 10 \
           --dataset_name 'herbarium_19' \
           --unbalanced  True

python -m   AL_train \
            --lr 0.05 \
            --grad_from_block 10 \
            --epochs 100 \
            --dataset_name 'imagenet_100'


python -m   AL_train \
            --lr 0.1 \
            --vis_rate 0.75 \
            --grad_from_block 10 \
            --dataset_name 'cifar100'



