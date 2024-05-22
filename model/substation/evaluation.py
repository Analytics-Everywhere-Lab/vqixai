import os

import numpy as np
import segmentation_models_pytorch as smp
import torch
from segmentation_models_pytorch.utils.metrics import IoU
from torch.utils.data import DataLoader

from model.substation.retrain import Dataset, get_preprocessing, get_validation_augmentation

DATA_DIR = "data/substation/ds"
TRAIN_DIR = "data/substation/train"
TEST_DIR = "data/substation/test"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # Updated to ensure compatibility
print(f"Using device: {DEVICE}")
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


def calculate_iou(model, dataloader, device, classes):
    iou_metric = IoU(threshold=0.5)
    iou_scores = {cls: [] for cls in classes}

    with torch.no_grad():
        for i, (image, gt_mask) in enumerate(dataloader):
            image, gt_mask = image.to(device), gt_mask.to(device)
            pr_mask = model(image).squeeze().cpu().numpy()
            pr_mask = np.argmax(pr_mask, axis=0)

            gt_mask = gt_mask.squeeze().cpu().numpy()

            if pr_mask.shape != gt_mask.shape:
                print(f"Shape mismatch: pr_mask shape {pr_mask.shape}, gt_mask shape {gt_mask.shape}")
                continue

            for idx, cls in enumerate(classes):
                gt_mask_filtered = (gt_mask == idx).astype(float)
                pr_mask_filtered = (pr_mask == idx).astype(float)

                gt_mask_tensor = torch.tensor(gt_mask_filtered, device=device, dtype=torch.float32).unsqueeze(
                    0).unsqueeze(0)
                pr_mask_tensor = torch.tensor(pr_mask_filtered, device=device, dtype=torch.float32).unsqueeze(
                    0).unsqueeze(0)

                iou_score = iou_metric(pr_mask_tensor, gt_mask_tensor).item()
                iou_scores[cls].append(iou_score)

    avg_iou_scores = {cls: np.mean(scores) for cls, scores in iou_scores.items()}
    miou = np.mean(list(avg_iou_scores.values()))

    return avg_iou_scores, miou


if __name__ == "__main__":
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    model = torch.load('model/substation/model_ResNet101.pth', map_location=DEVICE)
    model.to(DEVICE)
    model.eval()

    test_dataset = Dataset(
        x_valid_dir,
        y_valid_dir,
        classes=CLASSES,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    avg_iou_scores, miou = calculate_iou(model, test_dataloader, DEVICE, CLASSES)

    print("IoU Scores for each category:")
    for cls, score in avg_iou_scores.items():
        print(f"{cls}: {score:.4f}")

    print(f"\nMean IoU: {miou:.4f}")
