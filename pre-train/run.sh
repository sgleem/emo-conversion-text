#!/bin/bash

# you can set the hparams by using --hparams=xxx
CUDA_VISIBLE_DEVICES=0 python train.py -l logdir \
-o exp/model_0627 --n_gpus=1 --hparams=speaker_adversial_loss_w=20.,ce_loss=False,speaker_classifier_loss_w=0.1,contrastive_loss_w=30.