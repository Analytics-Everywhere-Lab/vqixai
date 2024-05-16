import segmentation_models_pytorch as smp
import os
import cv2
import json
import albumentations as albu
from PIL import Image, ImageDraw
import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from torch.utils.data import DataLoader
from segmentation_models_pytorch.utils.metrics import IoU
from segmentation_models_pytorch.losses import DiceLoss
from torch.utils.data import Dataset as BaseDataset
from segmentation_models_pytorch.utils.train import TrainEpoch

from model.semantic_segmentation_target import SemanticSegmentationTarget
from model.substation.retrain import Dataset, get_training_augmentation, get_preprocessing, visualize

DATA_DIR = "data/substation/ds"
TRAIN_DIR = "data/substation/train"
TEST_DIR = "data/substation/test"
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
XAI_METHODS = ["GradCAM", "GradCAMPlusPlus", "EigenCAM", "EigenGradCAM", "ScoreCAM", "HiResCAM", "AblationCAM",
               "XGradCAM"]

if __name__ == "__main__":
    category = "porcelain_pin_insulator"
    category_idx = CLASSES.index(category)

    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=len(CLASSES),
        activation=ACTIVATIONS,
    )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    loss = DiceLoss(mode='multiclass')
    metrics = [
        IoU(threshold=0.5),
    ]

    model = torch.load('model/substation/model_ResNet101.pth')

    test_dataset_vis = Dataset(
        x_valid_dir,
        y_valid_dir,
        classes=[category],
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    n = np.random.choice(len(test_dataset_vis))

    image_vis = test_dataset_vis[n][0].cpu().numpy().astype('uint8').transpose(1, 2, 0)
    rgb_img = np.float32(image_vis) / 255
    image, gt_mask = test_dataset_vis[n]

    gt_mask = gt_mask.squeeze()

    x_tensor = image.to(DEVICE).unsqueeze(0)  # No need to use from_numpy since image is already a tensor
    pr_mask = model.predict(x_tensor)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())
    pr_mask = pr_mask.argmax(axis=0)

    visualize(
        image=rgb_img,
        ground_truth_mask=gt_mask.cpu().numpy(),  # Convert gt_mask to numpy for visualization
        predicted_mask=pr_mask
    )

    target_layer = model.decoder.blocks[-1]

    with GradCAM(model=model, target_layers=[target_layer], use_cuda=torch.cuda.is_available()) as cam:
        grayscale_cam = cam(input_tensor=x_tensor)[0, :]
        cam_image = cam.show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    explanation = Image.fromarray(cam_image)
    explanation.show()
