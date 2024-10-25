import os
import json
import numpy as np
import cv2
import torch
import albumentations as albu
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

from model.ttpla.config import *
from utils import DEVICE


class TTPLADataset(Dataset):
    def __init__(self, images_dir, masks_dir, classes=None, augmentation=None, preprocessing=None):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, os.path.splitext(image_id)[0] + '.json') for image_id in self.ids]

        # Convert class names to class indices
        self.class_values = [CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with open(self.masks_fps[i], 'r') as f:
            mask_data = json.load(f)

        # create mask
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        for shape in mask_data['shapes']:
            if shape['label'] in CLASSES:
                class_id = CLASSES.index(shape['label'])
                if class_id in self.class_values:
                    points = np.array(shape['points'], dtype=np.int32)
                    cv2.fillPoly(mask, [points], class_id)
                else:
                    print(f"Class {shape['label']} not in classes list, skipping")

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)


def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        albu.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=16, pad_width_divisor=16),
        albu.GaussNoise(p=0.2),
        albu.Perspective(p=0.5),
        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightnessContrast(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),
        albu.OneOf(
            [
                albu.Sharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),
        albu.OneOf(
            [
                albu.RandomBrightnessContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
        albu.Resize(height=256, width=256)
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    test_transform = [
        albu.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=16, pad_width_divisor=16),
        albu.Resize(height=256, width=256)
    ]
    return albu.Compose(test_transform)


# Preprocessing function
def to_tensor(x, **kwargs):
    if x.ndim == 2:  # If the input is a 2D array (e.g., mask), add a channel dimension
        x = np.expand_dims(x, axis=2)
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


# Main function
if __name__ == '__main__':
    # Initialize the model
    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=len(CLASSES),
        activation=ACTIVATIONS,
    )
    model.to(DEVICE)

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    # Directories for images and masks
    x_train_dir = 'data/images/train'
    y_train_dir = 'data/labels/train'

    x_valid_dir = 'data/images/val'
    y_valid_dir = 'data/labels/val'

    # Create datasets
    train_dataset = TTPLADataset(
        x_train_dir,
        y_train_dir,
        classes=CLASSES,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    valid_dataset = TTPLADataset(
        x_valid_dir,
        y_valid_dir,
        classes=CLASSES,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Define loss function and metrics
    loss = smp.utils.losses.DiceLoss(mode='multiclass')
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]

    # Define optimizer
    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=0.0001),
    ])

    # Create training and validation epochs
    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    # Training loop
    max_score = 0
    train_loss_log = []
    train_iou_log = []
    valid_loss_log = []
    valid_iou_log = []

    for i in range(0, EPOCHS):
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        # Save logs
        train_loss_log.append(train_logs['dice_loss'])
        train_iou_log.append(train_logs['iou_score'])
        valid_loss_log.append(valid_logs['dice_loss'])
        valid_iou_log.append(valid_logs['iou_score'])

        # Save best model
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, './best_model.pth')
            print('Model saved!')

        # Decrease learning rate
        if i == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')

    # Plot training and validation metrics
    plt.figure(figsize=(6, 6))
    plt.plot(train_loss_log, label='Train Dice Loss')
    plt.plot(valid_loss_log, label='Valid Dice Loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Dice Loss')
    plt.grid(True)
    plt.savefig("TTPLA_DiceLoss.pdf")
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.plot(train_iou_log, label='Train IoU')
    plt.plot(valid_iou_log, label='Valid IoU')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('IoU Score')
    plt.grid(True)
    plt.savefig("TTPLA_IoU.pdf")
    plt.show()

    # Save logs for future use
    np.save('train_loss_log.npy', train_loss_log)
    np.save('train_iou_log.npy', train_iou_log)
    np.save('valid_loss_log.npy', valid_loss_log)
    np.save('valid_iou_log.npy', valid_iou_log)
