#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=train
#SBATCH -o res.out


# source /home/zhoukun/miniconda3/bin/activate s2s


# you can set the hparams by using --hparams=xxx


python train.py -l logdir \
-o emoVC_for_0011 --n_gpus=1 -c 'pre-trained-model/checkpoint_92999' --warm_start

