import segmentation_models_pytorch as smp
import os
import cv2
import json
import albumentations as albu
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from segmentation_models_pytorch.utils.metrics import IoU
from segmentation_models_pytorch.losses import DiceLoss
from torch.utils.data import Dataset as BaseDataset
from segmentation_models_pytorch.utils.train import TrainEpoch, ValidEpoch

from model.substation.retrain import Dataset, get_validation_augmentation, get_preprocessing, preprocessing_fn, \
    visualize

DATA_DIR = "data/substation/ds"
TRAIN_DIR = f"../../data/substation/train"
TEST_DIR = f"../../data/substation/test"
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps'
print(DEVICE)
EPOCH = 100
x_train_dir = os.path.join(TRAIN_DIR, 'img')
y_train_dir = os.path.join(TRAIN_DIR, 'ann')
x_valid_dir = os.path.join(TEST_DIR, 'img')
y_valid_dir = os.path.join(TEST_DIR, 'ann')

CLASSES = ['breaker', 'closed_blade_disconnect_switch', 'closed_tandem_disconnect_switch', 'current_transformer',
           'fuse_disconnect_switch', 'glass_disc_insulator', 'lightning_arrester', 'muffle',
           'open_blade_disconnect_switch', 'open_tandem_disconnect_switch', 'porcelain_pin_insulator',
           'potential_transformer', 'power_transformer', 'recloser', 'tripolar_disconnect_switch']

ENCODER = 'resnet101'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATIONS = 'softmax2d'

if __name__ == "__main__":
    loss = DiceLoss(mode='multiclass')
    metrics = [
        IoU(threshold=0.5),
    ]

    model = torch.load('model_ResNet101.pth')

    test_dataset_vis = Dataset(
        x_valid_dir, y_valid_dir,
        classes=CLASSES,
    )
    for i in range(5):
        n = np.random.choice(len(test_dataset_vis))

        image_vis = test_dataset_vis[n][0].astype('uint8')
        image, gt_mask = test_dataset_vis[n]

        gt_mask = gt_mask.squeeze()

        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())

        visualize(
            image=image_vis,
            ground_truth_mask=gt_mask,
            predicted_mask=pr_mask
        )
