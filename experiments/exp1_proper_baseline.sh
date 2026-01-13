#!/bin/bash

# Experiment 1: Proper Baseline with Fixed Validation Protocol
# This establishes a proper baseline using the correct train/val/test split
# WITHOUT any improvements (to see the real performance)

echo "=========================================="
echo "Experiment 1: Proper Baseline"
echo "=========================================="

python main.py \
    --mode train \
    --exper-name "exp1_proper_baseline" \
    --dataset RAER \
    --gpu 0 \
    --workers 4 \
    --seed 42 \
   --root-dir /kaggle/input/raer-video-emotion-dataset/ \
    --train-annotation /kaggle/input/raer-annot/annotation/train_80.txt \
    --val-annotation /kaggle/input/raer-annot/annotation/val_20.txt \
    --test-annotation /kaggle/input/raer-annot/annotation/test.txt \
    --clip-path ViT-B/32 \
    --bounding-box-face /kaggle/input/raer-video-emotion-dataset/RAER/bounding_box/face.json \
    --bounding-box-body /kaggle/input/raer-video-emotion-dataset/RAER/bounding_box/body.json \
    --epochs 30 \
    --batch-size 8 \
    --print-freq 10 \
    --use-amp \
    --grad-clip 1.0 \
    --optimizer AdamW \
    --lr 1e-4 \
    --lr-image-encoder 1e-6 \
    --lr-prompt-learner 5e-4 \
    --lr-adapter 1e-4 \
    --weight-decay 1e-4 \
    --momentum 0.9 \
    --milestones 20 35 \
    --gamma 0.1 \
    --lambda_mi 0.02 \
    --lambda_dc 0.08 \
    --mi-warmup 15 \
    --mi-ramp 25 \
    --dc-warmup 5 \
    --dc-ramp 10 \
    --use-weighted-sampler \
    --label-smoothing 0.02 \
    --text-type prompt_ensemble \
    --temporal-layers 1 \
    --contexts-number 12 \
    --class-token-position end \
    --class-specific-contexts True \
    --load_and_tune_prompt_learner True \
    --num-segments 16 \
    --duration 1 \
    --image-size 224 \
    --slerp-weight 0.0 \
    --temperature 0.07 \
    --crop-body

echo ""
echo "Experiment 1 completed!"
echo "Check outputs/exp1_proper_baseline-*/ for results"
