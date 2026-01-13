# builders.py

import argparse
from typing import Tuple
import os
import torch
import torch.utils.data
from clip import clip

from dataloader.video_dataloader import train_data_loader, test_data_loader
from models.Generate_Model import GenerateModel
from models.Text import *
from utils.utils import *
from collections import Counter
import numpy as np


def get_class_weights_from_annotation(annotation_file: str, num_classes: int = 5, beta: float = 0.9999) -> torch.Tensor:
    """
    Compute class-balanced weights from annotation file using effective number of samples.

    Args:
        annotation_file: Path to annotation file (train.txt)
        num_classes: Number of classes
        beta: Hyperparameter for effective number (default 0.9999)

    Returns:
        Class weights tensor of shape (num_classes,)
    """
    # Read labels from annotation file
    labels = []
    with open(annotation_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                # Labels are 1-indexed in file, convert to 0-indexed
                label = int(parts[2]) - 1
                if 0 <= label < num_classes:
                    labels.append(label)

    # Count samples per class
    counter = Counter(labels)
    cls_num_list = [counter.get(i, 0) for i in range(num_classes)]

    # Compute effective number of samples (from Class-Balanced Loss paper)
    effective_num = 1.0 - np.power(beta, cls_num_list)
    weights = (1.0 - beta) / (np.array(effective_num) + 1e-8)

    # Normalize weights so they sum to num_classes
    weights = weights / weights.sum() * num_classes

    print(f"\n{'='*60}")
    print("CLASS DISTRIBUTION & WEIGHTS")
    print(f"{'='*60}")
    print(f"Annotation file: {annotation_file}")
    print(f"Total samples: {len(labels)}")
    print(f"\nPer-class statistics:")
    class_names = ['Neutrality', 'Enjoyment', 'Confusion', 'Fatigue', 'Distraction']
    for i in range(num_classes):
        count = cls_num_list[i]
        weight = weights[i]
        percentage = (count / len(labels) * 100) if len(labels) > 0 else 0
        print(f"  Class {i} ({class_names[i]:12s}): {count:4d} samples ({percentage:5.1f}%) → weight: {weight:.3f}")
    print(f"{'='*60}\n")

    return torch.FloatTensor(weights)


def build_model(args: argparse.Namespace, input_text: list) -> torch.nn.Module:
    print("Loading pretrained CLIP model...")
    CLIP_model, _ = clip.load(args.clip_path, device='cpu')

    print("\nInput Text Prompts:")
    # Handle the case where input_text is a list of lists for prompt ensembling
    if any(isinstance(i, list) for i in input_text):
        for class_prompts in input_text:
            print(f"- Class: {class_prompts}")
    else:
        for text in input_text:
            print(text)


    print("\nInstantiating GenerateModel...")
    model = GenerateModel(input_text=input_text, clip_model=CLIP_model, args=args)

    for name, param in model.named_parameters():
        param.requires_grad = False

    # Freeze CLIP image encoder if lr_image_encoder is 0
    # Otherwise, make it trainable.
    if args.lr_image_encoder > 0:
        for name, param in model.named_parameters():
            if "image_encoder" in name:
                param.requires_grad = True

    trainable_params_keywords = ["temporal_net", "prompt_learner", "temporal_net_body", "project_fc", "face_adapter"]
    
    print('\nTrainable parameters:')
    for name, param in model.named_parameters():
        if any(keyword in name for keyword in trainable_params_keywords):
            param.requires_grad = True
            print(f"- {name}")
    print('************************\n')

    return model


def get_class_info(args: argparse.Namespace) -> Tuple[list, list]:
    """
    根据数据集和文本类型获取 class_names 和 input_text（用于生成 CLIP 模型文本输入）。

    Returns:
        class_names: 类别名称，用于混淆矩阵等
        input_text: 输入文本，用于传入模型
    """
    if args.dataset == "RAER":
        class_names = ['Neutrality', 'Enjoyment', 'Confusion', 'Fatigue', 'Distraction.']
        class_names_with_context = class_names_with_context_5
        class_descriptor = class_descriptor_5
        ensemble_prompts = prompt_ensemble_5
    else:
        raise NotImplementedError(f"Dataset '{args.dataset}' is not implemented yet.")

    if args.text_type == "class_names":
        input_text = class_names
    elif args.text_type == "class_names_with_context":
        input_text = class_names_with_context
    elif args.text_type == "class_descriptor":
        input_text = class_descriptor
    elif args.text_type == "prompt_ensemble":
        input_text = ensemble_prompts
    else:
        raise ValueError(f"Unknown text_type: {args.text_type}")

    return class_names, input_text



def build_dataloaders(args: argparse.Namespace) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]: 
    train_annotation_file_path = args.train_annotation
    val_annotation_file_path = args.val_annotation
    test_annotation_file_path = args.test_annotation
    
    class_names, _ = get_class_info(args)
    num_classes = len(class_names)

    print("Loading train data...")
    train_data = train_data_loader(
        root_dir=args.root_dir, list_file=train_annotation_file_path, num_segments=args.num_segments,
        duration=args.duration, image_size=args.image_size,dataset_name=args.dataset,
        bounding_box_face=args.bounding_box_face,bounding_box_body=args.bounding_box_body,
        crop_body=args.crop_body,
        num_classes=num_classes
    )
    print(f"Total number of training images: {len(train_data)}")
    
    print("Loading validation data...")
    val_data = test_data_loader(
        root_dir=args.root_dir, list_file=val_annotation_file_path, num_segments=args.num_segments,
        duration=args.duration, image_size=args.image_size,
        bounding_box_face=args.bounding_box_face,bounding_box_body=args.bounding_box_body,
        crop_body=args.crop_body,
        num_classes=num_classes
    )

    print("Loading test data...")
    test_data = test_data_loader(
        root_dir=args.root_dir, list_file=test_annotation_file_path, num_segments=args.num_segments,
        duration=args.duration, image_size=args.image_size,
        bounding_box_face=args.bounding_box_face,bounding_box_body=args.bounding_box_body,
        crop_body=args.crop_body,
        num_classes=num_classes
    )

    print("Creating DataLoader instances...")
    
    sampler = None
    shuffle = True
    if args.use_weighted_sampler:
        print("=> Using WeightedRandomSampler.")
        class_counts = get_class_counts(train_annotation_file_path)
        class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
        
        # Create a weight for each sample
        sample_weights = []
        with open(train_annotation_file_path, 'r') as f:
            for line in f:
                label = int(line.strip().split()[2]) -1 # label is 1-based
                sample_weights.append(class_weights[label])
        
        sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))
        shuffle = False # Sampler and shuffle are mutually exclusive

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler,
        num_workers=args.workers, pin_memory=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader