#!/bin/bash

# Experiment 3: Full Stack - All Phase 1 Improvements
# Includes:
#   - Proper validation protocol (train_80 / val_20 / test)
#   - Class-balanced Focal Loss with automatic weights
#   - Lower temperature (0.07 instead of 0.5)
#   - Reduced MI loss interference (lambda_mi=0.02, longer warmup)
#   - WeightedRandomSampler
#
# Target: Valid UAR >= 65%, Class Confusion recall >= 40%

echo "=========================================="
echo "Experiment 3: Full Stack (All Improvements)"
echo "=========================================="

python main.py \
    --mode train \
    --exper-name "exp3_full_stack" \
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
    --epochs 35 \
    --batch-size 16 \
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
    --milestones 22 30 \
    --gamma 0.1 \
    --lambda_mi 0.02 \
    --lambda_dc 0.08 \
    --mi-warmup 15 \
    --mi-ramp 25 \
    --dc-warmup 5 \
    --dc-ramp 10 \
    --class-balanced-loss \
    --use-weighted-sampler \
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
echo "=========================================="
echo "Experiment 3 completed!"
echo "=========================================="
echo "Expected Results:"
echo "  - Valid UAR: 65-70%"
echo "  - Class 2 (Confusion) Recall: 40-50%"
echo "  - Train-Val UAR gap: <20%"
echo ""
echo "Check outputs/exp3_full_stack-*/ for results"
echo "Review per-class recall in log.txt"
