import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
from pytorch_grad_cam import *
from pytorch_grad_cam.utils.image import show_cam_on_image
from segmentation_models_pytorch.losses import DiceLoss
from segmentation_models_pytorch.utils.metrics import IoU

from model.semantic_segmentation_target import SemanticSegmentationTarget
from model.substation.retrain import Dataset, get_training_augmentation, get_preprocessing, visualize

DATA_DIR = "data/substation/ds"
TRAIN_DIR = "data/substation/train"
TEST_DIR = "data/substation/test"
INFERENCE_DIR = "data/substation/inference"
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
XAI_METHODS = [GradCAM, GradCAMPlusPlus, EigenCAM, EigenGradCAM, ScoreCAM, HiResCAM, AblationCAM, XGradCAM, LayerCAM,
               FullGrad]


def save_image(image, filename):
    plt.figure(figsize=(30, 30))
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()


def overlay_mask_on_image(image, mask, color, alpha=0.01):
    color_mask = np.zeros_like(image)
    color_mask[mask == 1] = color
    overlay = cv2.addWeighted(image, 1, color_mask, alpha, 0)
    return overlay


if __name__ == "__main__":
    category = "recloser"
    category_idx = CLASSES.index(category)

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    loss = DiceLoss(mode='multiclass')
    metrics = [
        IoU(threshold=0.5),
    ]

    model = torch.load('model/substation/model_ResNet101.pth')
    model.eval()

    test_dataset_vis = Dataset(
        x_valid_dir,
        y_valid_dir,
        classes=[category],
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    for n in range(len(test_dataset_vis)):
        raw_img = test_dataset_vis[n][0].astype('uint8')
        image_vis = test_dataset_vis[n][1].cpu().numpy().astype('uint8').transpose(1, 2, 0)
        raw_img = np.float32(raw_img) / 255
        _, image, gt_mask = test_dataset_vis[n]

        gt_mask = gt_mask.squeeze()

        # Filter the ground truth mask to only keep the category of interest
        gt_mask_filtered = np.zeros_like(gt_mask)
        gt_mask_filtered[gt_mask == category_idx] = 1

        x_tensor = image.to(DEVICE).unsqueeze(0)
        pr_mask = model.predict(x_tensor)
        pr_mask = pr_mask.squeeze().cpu().numpy().round()

        # Filter the prediction to only keep the category of interest
        pr_mask_filtered = np.zeros_like(pr_mask[category_idx])
        if pr_mask.ndim > 2:  # Check if pr_mask has multiple channels
            pr_mask_filtered = pr_mask[category_idx]
        else:
            pr_mask_filtered[pr_mask == category_idx] = 1

        target_layer = model.decoder.block1
        targets = [SemanticSegmentationTarget(category_idx, pr_mask_filtered)]

        for xai_method in XAI_METHODS:
            with xai_method(model=model, target_layers=[target_layer]) as cam:
                grayscale_cam = cam(input_tensor=x_tensor, targets=targets)[0, :]
                cam_image = show_cam_on_image(raw_img, grayscale_cam, use_rgb=True)

                # Save the XAI method result
                xai_method_name = xai_method.__name__
                save_image(cam_image, f"{INFERENCE_DIR}/image_{n}_{xai_method_name}.png")

        # Overlay masks on the raw image
        gt_overlay = overlay_mask_on_image(raw_img, gt_mask_filtered, color=[0, 255, 0])  # Green for GT
        pr_overlay = overlay_mask_on_image(raw_img, pr_mask_filtered, color=[255, 0, 0])  # Red for Predicted

        visualize(
            image=raw_img,
            ground_truth_mask=gt_overlay,
            predicted_mask=pr_overlay,
        )

        # Save the visualized image
        save_image(raw_img, f"{INFERENCE_DIR}/image_{n}_raw.png")
        save_image(gt_overlay, f"{INFERENCE_DIR}/image_{n}_gt_overlay.png")
        save_image(pr_overlay, f"{INFERENCE_DIR}/image_{n}_pr_overlay.png")

        # Convert masks to tensors for IoU calculation
        gt_mask_tensor = torch.tensor(gt_mask_filtered, device=DEVICE, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        pr_mask_tensor = torch.tensor(pr_mask_filtered, device=DEVICE, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # Calculate IoU
        iou_score = metrics[0](pr_mask_tensor, gt_mask_tensor).item()
        print(f"Image {n} IoU Score: {iou_score}")
