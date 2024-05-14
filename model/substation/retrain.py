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
from segmentation_models_pytorch.utils.train import TrainEpoch

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


class Dataset(BaseDataset):
    def __init__(self, images_dir, masks_dir, classes=None, augmentation=None, preprocessing=None):
        self.ids = os.listdir(images_dir)
        self.imgs_fps = [os.path.join(images_dir, img_id) for img_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, img_id + '.json') for img_id in self.ids]

        self.class_values = [CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        img_fp = self.imgs_fps[i]
        mask_fp = self.masks_fps[i]

        img = cv2.imread(img_fp)
        if img is None:
            raise ValueError(f"Image not found or unable to read: {img_fp}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        with open(self.masks_fps[i], 'r') as f:
            mask_data = json.load(f)
            img_size = (mask_data['size']['height'], mask_data['size']['width'])

        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        for object in mask_data['objects']:
            if object['classTitle'] in CLASSES:
                class_index = CLASSES.index(object['classTitle'])
                mask = cv2.fillPoly(mask, np.array([object['points']['exterior']]), class_index)

        if self.augmentation:
            sample = self.augmentation(image=img, mask=mask)
            img, mask = sample['image'], sample['mask']

        if self.preprocessing:
            sample = self.preprocessing(image=img, mask=mask)
            img, mask = sample['image'], sample['mask']

        # Convert to PyTorch tensors
        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).long()

        return img, mask

    def __len__(self):
        return len(self.ids)


dataset = Dataset(x_train_dir, y_train_dir, classes=CLASSES)


# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


def get_training_augmentation():
    """Add paddings to make image shape divisible by 16 and resize to a fixed size."""
    train_transform = [
        albu.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=16, pad_width_divisor=16),
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


class NamedDiceLoss(DiceLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__name__ = "DiceLoss"


if __name__ == '__main__':
    # Initialize the model
    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=len(CLASSES),
        activation=ACTIVATIONS,
    )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    train_dataset = Dataset(
        x_train_dir,
        y_train_dir,
        classes=CLASSES,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    test_dataset = Dataset(
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

    for i in range(EPOCH):
        print(f"Epoch: {i}")
        try:
            train_logs = train_epoch.run(train_loader)
        except ValueError as e:
            print(f"Skipping a problematic sample: {e}")
            continue

        # Save the logs
        train_loss_log.append(train_logs['DiceLoss'])
        train_iou_log.append(train_logs['IoU'])

        if i == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')

    # Save the model
    torch.save(model, './model.pth')
    print('Model saved!')

    # Save the logs for future use
    np.save('train_loss_log.npy', train_loss_log)
    np.save('train_iou_log.npy', train_iou_log)

    # Plot the logs
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_log, label='Train Loss')
    plt.title('Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_iou_log, label='Train IoU')
    plt.title('Train IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()

    plt.show()
