import json

import albumentations as albu
import cv2
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
from segmentation_models_pytorch.utils.metrics import IoU
from segmentation_models_pytorch.utils.train import TrainEpoch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from model.substation.utils import *


class SubstationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, classes=None, augmentation=None, preprocessing=None):
        self.ids = os.listdir(images_dir)
        self.imgs_fps = [os.path.join(images_dir, img_id) for img_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, img_id + '.json') for img_id in self.ids]

        self.class_values = [CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        img = cv2.imread(self.imgs_fps[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        with open(self.masks_fps[i], 'r') as f:
            mask_data = json.load(f)
            # img_size = (mask_data['size']['height'], mask_data['size']['width'])

        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        for object in mask_data['objects']:
            if object['classTitle'] in CLASSES:
                class_index = CLASSES.index(object['classTitle'])
                if class_index in self.class_values:
                    mask = cv2.fillPoly(mask, np.array([object['points']['exterior']]), class_index)
                else:
                    print(f"Class {object['classTitle']} not in classes list, skipping")

        if self.augmentation:
            sample = self.augmentation(image=img, mask=mask)
            img, mask = sample['image'], sample['mask']

        if self.preprocessing:
            sample = self.preprocessing(image=img, mask=mask)
            img, mask = sample['image'], sample['mask']

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).long()

        return img, mask

    def __len__(self):
        return len(self.ids)


class NamedDiceLoss(DiceLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__name__ = "DiceLoss"


def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        print(name.split('_'))
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


def get_training_augmentation():
    """Add paddings to make image shape divisible by 16 and resize to a fixed size."""
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
    """Add paddings to make image shape divisible by 16 and resize to a fixed size."""
    test_transform = [
        albu.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=16, pad_width_divisor=16),
        albu.Resize(height=256, width=256)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    if x.ndim == 2:  # If the input is a 2D array (e.g., mask), add a channel dimension
        x = np.expand_dims(x, axis=2)
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callable): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    print(_transform)
    return albu.Compose(_transform)


if __name__ == '__main__':
    # Initialize the model
    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=len(CLASSES),
        activation=ACTIVATIONS,
    )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    train_dataset = SubstationDataset(
        x_train_dir,
        y_train_dir,
        classes=CLASSES,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    test_dataset = SubstationDataset(
        x_valid_dir,
        y_valid_dir,
        classes=CLASSES,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    # Print number of samples in train and test datasets
    print('Number of samples in train dataset:', len(train_dataset))
    print('Number of samples in test dataset:', len(test_dataset))

    loss = NamedDiceLoss(mode='multiclass')
    metrics = [
        IoU(threshold=0.5),
    ]

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=0.0001),
    ])

    train_epoch = TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)

    # Lists to store the logs
    train_loss_log = []
    train_iou_log = []

    for i in range(EPOCHS):
        print(f"Epoch: {i}")
        train_logs = train_epoch.run(train_loader)

        # Save the logs
        train_loss_log.append(train_logs['DiceLoss'])
        train_iou_log.append(train_logs['iou_score'])

        if i == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')

    # Save the model
    torch.save(model, 'model_ResNet101.pth')
    print('Model saved!')

    # Save the logs for future use
    np.save('train_loss_log_aug.npy', train_loss_log)
    np.save('train_iou_log_aug.npy', train_iou_log)

    # Plot different metrics
    plt.figure(figsize=(6, 6))
    plt.rcParams.update({'font.size': 16})
    plt.plot(train_loss_log, marker='o', markersize=5, markevery=10, markerfacecolor='red')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Loss')
    plt.grid(True)

    # Save the plot
    plt.savefig("DLv3PLoss.pdf")
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.rcParams.update({'font.size': 16})
    plt.plot(train_iou_log, marker='s', markersize=5, markevery=10, markerfacecolor='blue')
    plt.xlabel('Epochs')
    plt.ylabel('IoU Score')
    plt.grid(True)

    # Save the plot
    plt.savefig("DLv3PIoU.pdf")

    plt.show()
