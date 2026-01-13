# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is CLIP-CAER, a CLIP-based Context-aware Academic Emotion Recognition system for the RAER dataset. The model recognizes student learning states (focused/distracted) using dual-stream video processing with facial expressions and contextual information. Accepted at ICCV 2025.

## Key Commands

### Training
```bash
# Standard training
bash train.sh

# Google Colab training
bash train_colab.sh

# Local training
bash train_local.sh

# Debug training
bash train_debug.sh
```

### Evaluation
```bash
# Standard evaluation
bash valid.sh

# Google Colab evaluation
bash valid_colab.sh
```

### Direct Execution
```bash
python main.py \
    --mode train \
    --exper-name <experiment_name> \
    --gpu 0 \
    --epochs 50 \
    --batch-size 8 \
    --root-dir <path_to_RAER_dataset> \
    --train-annotation <path_to_train.txt> \
    --test-annotation <path_to_test.txt> \
    --clip-path ViT-B/32 \
    --bounding-box-face <path_to_face.json> \
    --bounding-box-body <path_to_body.json>
```

## Architecture

### Core Model Pipeline
The architecture consists of several key components working together:

1. **Dual-Stream Visual Processing** (`models/Generate_Model.py:59-79`)
   - Face Stream: Crops facial regions → CLIP Visual Encoder → Expression-Aware Adapter (EAA)
   - Context Stream: Full frames/body regions → CLIP Visual Encoder
   - Each stream has its own Temporal Transformer (`temporal_net` for face, `temporal_net_body` for context)
   - Streams are concatenated and projected to 512-dim via `project_fc`

2. **Expression-Aware Adapter (EAA)** (`models/Adapter.py`)
   - Lightweight bottleneck adapter (reduction=4) applied only to face stream
   - Allows fine-tuning emotion-specific facial features without modifying pre-trained CLIP
   - Trainable while main visual encoder can be frozen

3. **Dual-View Prompting System** (`models/Generate_Model.py:82-92`)
   - Learnable prompts: CoOp-style context vectors optimized during training (`prompt_learner`)
   - Hand-crafted prompts: Static descriptive prompts with AU-like micro-expressions (`class_descriptor_5_only_face`)
   - MI Loss maximizes mutual information between both views to prevent prompt drift

4. **Instance-Enhanced Classifier (IEC)** (`models/Generate_Model.py:109-117`)
   - Uses Spherical Linear Interpolation (slerp) to mix text prototypes with visual features
   - Creates instance-specific classifiers: `t_mix = slerp(t_desc, z, λ_slerp)`
   - Controlled by `--slerp-weight` (default: 0.5)

5. **Temporal Modeling** (`models/Temporal_Model.py`)
   - `Temporal_Transformer_Cls`: Transformer with CLS token for sequence aggregation
   - Processes 16 video segments by default
   - Separate transformers for face and body streams

### Loss Functions
The model uses a composite loss system (`trainer.py:128-141`):

- **Classification Loss**: Cross-entropy (or FocalLoss with `--class-balanced-loss`, or LSR2 with `--label-smoothing`)
- **MI Loss** (`utils/loss.py:52-73`): InfoNCE-based loss ensuring learnable prompts stay semantically aligned with hand-crafted ones
- **DC Loss** (`utils/loss.py:36-50`): Decorrelation loss penalizing similarity between class prototypes
- **Logit Adjustment**: Optional post-hoc calibration for class imbalance (`--logit-adj`)

Both MI and DC losses support warmup/ramp schedules to gradually introduce regularization.

### Data Flow
1. Video loaded and sampled into 16 segments (`--num-segments`)
2. Face and body regions cropped using bounding box annotations
3. Both regions processed through CLIP visual encoder
4. Face features pass through adapter
5. Temporal transformers aggregate frame-level to video-level features
6. Dual-view text prompts generate class prototypes
7. Classification via cosine similarity (with optional IEC interpolation)

## Important Configuration

### Dataset Paths
- Root directory: Contains RAER dataset videos
- Annotations: Train/val/test splits in `.txt` format (format: `<video_path> <video_id> <label>`)
- Bounding boxes: Face and body crops in JSON format

### Key Hyperparameters
- `--lr-image-encoder`: Set to 0 to freeze CLIP encoder (recommended)
- `--lr-prompt-learner`: Learning rate for learnable prompts (default: 1e-5)
- `--lr-adapter`: Learning rate for EAA (default: 1e-5)
- `--lambda_mi`: Weight for MI loss (default: 0.7)
- `--lambda_dc`: Weight for DC loss (default: 1.2)
- `--slerp-weight`: IEC interpolation factor; 0 disables IEC
- `--temperature`: Softmax temperature for classification (default: 0.07)
- `--text-type`: Options are `class_names`, `class_names_with_context`, `class_descriptor`, `prompt_ensemble`

### Imbalance Handling Options
- `--class-balanced-loss`: Use FocalLoss instead of CrossEntropyLoss
- `--logit-adj`: Apply logit adjustment with tau parameter
- `--use-weighted-sampler`: Use WeightedRandomSampler for training
- `--label-smoothing`: Apply label smoothing (LSR2)

### Training Features
- `--use-amp`: Enable automatic mixed precision training
- `--grad-clip`: Gradient clipping threshold (default: 1.0)
- Warmup and ramp schedules for MI/DC losses prevent early training instability

## Module Organization

- `main.py`: Entry point with argument parsing and training/eval orchestration
- `trainer.py`: Training loop, validation, and metric computation (WAR/UAR)
- `models/Generate_Model.py`: Main model combining all components
- `models/Prompt_Learner.py`: CoOp-style learnable prompt implementation
- `models/Temporal_Model.py`: Temporal transformer variants
- `models/Adapter.py`: Expression-aware adapter module
- `models/Text.py`: Hand-crafted prompt definitions for each class
- `dataloader/video_dataloader.py`: Video loading with face/body cropping
- `utils/builders.py`: Model and dataloader construction logic
- `utils/loss.py`: Custom loss functions (FocalLoss, MILoss, DCLoss, LSR2)
- `utils/utils.py`: Utilities including slerp interpolation, metrics, checkpointing

## Metrics

The model reports:
- **WAR** (Weighted Average Recall): Overall accuracy
- **UAR** (Unweighted Average Recall): Mean per-class recall (primary metric, target >70%)
- Confusion matrices saved as PNG files in output directory

Best model selected by UAR on validation set.

## Output Structure

Training creates: `outputs/<experiment_name>-[timestamp]/`
- `log.txt`: Training logs with hyperparameters and epoch metrics
- `log.png`: Training curves
- `confusion_matrix.png`: Final test set confusion matrix
- `model.pth`: Latest checkpoint
- `model_best.pth`: Best model by validation UAR

## Notes

- Default dataset: RAER with 5 classes (Neutrality, Enjoyment, Confusion, Fatigue, Distraction)
- CLIP model: ViT-B/32 is the default backbone
- The model can run on CUDA, CPU, or Apple Silicon MPS
- Debug mode saves prediction samples to `debug_predictions/` directory
- Set `--crop-body` flag to use body crops instead of full frames for context stream
