import albumentations as albu
import cv2
import numpy as np
import os
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from pytorch_grad_cam import HiResCAM
from segmentation_models_pytorch.losses import DiceLoss
from segmentation_models_pytorch.utils.metrics import IoU
from segmentation_models_pytorch.utils.train import TrainEpoch, ValidEpoch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from model.semantic_segmentation_target import SemanticSegmentationTarget
import uuid
from utils import *


class CamVidDataset(Dataset):
    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)

        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

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


class XAIEnhancedDiceLoss(DiceLoss):
    def __init__(self, model, target_layer, alignment_weight, classes=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.target_layer = target_layer
        self.category_indices = [CLASSES.index(cls.lower()) for cls in classes]
        self.__name__ = "dice_loss"
        self.alignment_weight = alignment_weight

    def forward(self, y_pred, y_true, x_tensor):
        global count
        # Ensure y_true is on the same device as y_pred
        y_true = y_true.to(y_pred.device)

        # Check for NaNs in inputs
        assert not torch.isnan(y_pred).any(), "y_pred contains NaNs"
        assert not torch.isnan(y_true).any(), "y_true contains NaNs"
        assert not torch.isnan(x_tensor).any(), "x_tensor contains NaNs"

        # Convert y_true from one-hot encoding to index tensor
        y_true_indices = torch.argmax(y_true, dim=1)

        # Compute the standard Dice loss
        dice_loss = super().forward(y_pred, y_true_indices)
        if self.alignment_weight == 0:
            alignment_loss = 0
        else:
            dice_loss = torch.log(dice_loss + 1e-8)
            # Compute the Grad-CAM heatmap for each category
            all_grayscale_cam_tensors = torch.zeros_like(y_true, device=y_pred.device)
            cam = HiResCAM(model=self.model, target_layers=[self.target_layer])
            batch_size, num_classes, height, width = y_true.shape

            for batch_idx in range(batch_size):
                grayscale_cams = cam(input_tensor=x_tensor[batch_idx].unsqueeze(0),
                                     targets=[SemanticSegmentationTarget(cat_idx, y_pred[batch_idx]) for cat_idx in
                                              range(num_classes)])

                for cat_idx, grayscale_cam in enumerate(grayscale_cams):
                    # Resize the heatmap to the size of y_true
                    grayscale_cam_resized = cv2.resize(grayscale_cam, (width, height))

                    # Convert to torch tensor
                    grayscale_cam_tensor = torch.from_numpy(grayscale_cam_resized).float().to(y_pred.device)

                    # Normalize the heatmap
                    grayscale_cam_tensor = (grayscale_cam_tensor - grayscale_cam_tensor.min()) / (
                            grayscale_cam_tensor.max() - grayscale_cam_tensor.min() + 1e-8)  # Add epsilon to avoid division by zero

                    # Store the heatmap in the batch tensor for the corresponding category
                    all_grayscale_cam_tensors[batch_idx, cat_idx, :, :] = grayscale_cam_tensor

                    # # Visualization for each category
                    # self.visualize(x_tensor, y_true, y_pred, grayscale_cam_tensor, batch_idx, cat_idx)

                # Align the heatmap with the ground truth
            alignment_loss = F.mse_loss(all_grayscale_cam_tensors, y_true.float())
            alignment_loss = torch.norm(alignment_loss, p=2)

        # Combine the Dice loss with the alignment loss
        # total_loss = (1 - self.alignment_weight) * dice_loss + self.alignment_weight * alignment_loss
        total_loss = (1 - self.alignment_weight) * alignment_loss - self.alignment_weight * dice_loss
        # Check for NaNs in the computed loss
        assert not torch.isnan(total_loss).any(), "total_loss contains NaNs"

        return total_loss


class TrainEpochWithInput(TrainEpoch):
    def batch_update(self, x, y):
        self.model.train()
        self.optimizer.zero_grad()
        prediction = self.model(x)
        loss = self.loss(prediction, y, x)
        loss.backward()
        self.optimizer.step()

        return loss, prediction


class ValidEpochWithInput(ValidEpoch):
    def batch_update(self, x, y):
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(x)

        # Enable gradient tracking for GradCAM
        with torch.enable_grad():
            prediction_for_gradcam = self.model(x)
            loss = self.loss(prediction_for_gradcam, y, x)

        return loss, prediction


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0.005):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, optimizer):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.counter == 3:
                optimizer.param_groups[0]['lr'] = 1e-5
                print('Decrease decoder learning rate to 1e-5.')
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model...')
        torch.save(model, f'ckpt_xai_log_l2.pth')
        self.val_loss_min = val_loss


def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True),
        albu.RandomCrop(height=320, width=320, always_apply=True),

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
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(384, 480)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


def visualize(x_tensor, y_true, y_pred, cat_idx, alignment_weight=0, grayscale_cam_tensor=None, batch_idx=0):
    save_folder = f'model/camvid/results/{alignment_weight}'
    os.makedirs(save_folder, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Plot input image
    axes[0].imshow(x_tensor[batch_idx].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
    axes[0].set_title(f'Input Image (Batch {batch_idx}, Category {cat_idx})')
    axes[0].axis('off')

    # Plot predicted mask
    axes[1].imshow(y_pred[batch_idx, cat_idx].squeeze().cpu().detach().numpy().round(), cmap='jet')
    axes[1].set_title(f'Predicted Mask (Batch {batch_idx}, Category {cat_idx})')
    axes[1].axis('off')

    # Plot true mask
    axes[2].imshow(y_true[batch_idx, cat_idx].cpu().numpy(), cmap='jet')
    axes[2].set_title(f'True Mask (Batch {batch_idx}, Category {cat_idx})')
    axes[2].axis('off')

    # Plot XAI heatmap
    if grayscale_cam_tensor is not None:
        axes[3].imshow(grayscale_cam_tensor.cpu().numpy(), cmap='jet')
        axes[3].set_title(f'HiResCAM Heatmap (Batch {batch_idx}, Category {cat_idx})')
        axes[3].axis('off')

    unique_id = uuid.uuid4()
    # Save
    plt.savefig(f'{save_folder}/batch_{batch_idx}_category_{cat_idx}_{unique_id}.png')
    plt.close(fig)


def evaluate_and_visualize(model, eval_loader, alignment_weight=0):
    model.to(DEVICE)
    model.eval()
    with torch.enable_grad():
        for x, y in eval_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_pred = model(x)
            cam = HiResCAM(model=model, target_layers=[model.segmentation_head])
            batch_size, num_classes, height, width = y.shape
            for batch_idx in range(batch_size):
                grayscale_cams = cam(input_tensor=x[batch_idx].unsqueeze(0),
                                     targets=[SemanticSegmentationTarget(cat_idx, y_pred[batch_idx]) for cat_idx in
                                              range(num_classes)])
                for cat_idx, grayscale_cam in enumerate(grayscale_cams):
                    # Resize the heatmap to the size of y_true
                    grayscale_cam_resized = cv2.resize(grayscale_cam, (width, height))

                    # Convert to torch tensor
                    grayscale_cam_tensor = torch.from_numpy(grayscale_cam_resized).float().to(y_pred.device)

                    # Normalize the heatmap
                    grayscale_cam_tensor = (grayscale_cam_tensor - grayscale_cam_tensor.min()) / (
                            grayscale_cam_tensor.max() - grayscale_cam_tensor.min() + 1e-8)

                    visualize(x,
                              y,
                              y_pred=y_pred,
                              cat_idx=cat_idx,
                              alignment_weight=alignment_weight,
                              grayscale_cam_tensor=grayscale_cam_tensor,
                              batch_idx=batch_idx)
    print('Evaluation and visualization complete.')


def evaluate_test_set(model, test_loader, loss_fn, metrics, device):
    model.eval()
    test_loss = 0
    test_iou = 0
    num_batches = len(test_loader)

    with torch.enable_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)

            loss = loss_fn(y_pred, y, x)
            test_loss += loss.item()

            for metric in metrics:
                test_iou += metric(y_pred, y).item()

    test_loss /= num_batches
    test_iou /= num_batches

    print(f"Test Loss: {test_loss}, Test IoU: {test_iou}")
    return test_loss, test_iou


if __name__ == '__main__':
    model = smp.DeepLabV3Plus(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS, classes=len(CLASSES),
                              activation=ACTIVATIONS)

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    # Dataset and DataLoader setup
    train_dataset = CamVidDataset(x_train_dir, y_train_dir, classes=CLASSES,
                                  augmentation=get_training_augmentation(),
                                  preprocessing=get_preprocessing(preprocessing_fn))
    train_loader = DataLoader(train_dataset, batch_size=48, shuffle=False, num_workers=4)
    val_dataset = CamVidDataset(x_valid_dir, y_valid_dir, classes=CLASSES,
                                augmentation=get_validation_augmentation(),
                                preprocessing=get_preprocessing(preprocessing_fn))
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

    test_dataset = CamVidDataset(x_test_dir, y_test_dir, classes=CLASSES,
                                 augmentation=get_validation_augmentation(),
                                 preprocessing=get_preprocessing(preprocessing_fn))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    print('Number of samples in train dataset:', len(train_dataset))
    print('Number of samples in val dataset:', len(val_dataset))
    # Loss, Metrics, and Optimizerl
    loss = XAIEnhancedDiceLoss(model=model, target_layer=model.segmentation_head, alignment_weight=ALIGNMENT_WEIGHT,
                               mode='multiclass', classes=CLASSES)
    # loss = smp.utils.losses.DiceLoss()
    metrics = [IoU(threshold=0.5)]
    optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.0001)])

    # Early Stopping Initialization
    early_stopping = EarlyStopping(patience=15, verbose=True)

    train_epoch = TrainEpochWithInput(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    val_epoch = ValidEpochWithInput(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    # Lists to store the logs
    train_loss_log = []
    train_iou_log = []
    val_loss_log = []
    val_iou_log = []

    for epoch in tqdm(range(EPOCHS)):
        train_logs = train_epoch.run(train_loader)

        # Save the logs
        train_loss_log.append(train_logs['dice_loss'])
        train_iou_log.append(train_logs['iou_score'])

        # Validation
        val_logs = val_epoch.run(val_loader)

        # Save the logs
        val_loss_log.append(val_logs['dice_loss'])
        val_iou_log.append(val_logs['iou_score'])

        print(f"Epoch: {epoch}, Train Loss: {train_logs['dice_loss']}, Train IoU: {train_logs['iou_score']}, "
              f"Val Loss: {val_logs['dice_loss']}, Val IoU: {val_logs['iou_score']}")

        # Early Stopping
        early_stopping(val_logs['dice_loss'], model, optimizer)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Save the logs for future use
    np.save('train_loss_log_aug_xai_reg.npy', train_loss_log)
    np.save('train_iou_log_aug_xai_reg.npy', train_iou_log)
    np.save('val_loss_log_aug_xai_reg.npy', val_loss_log)
    np.save('val_iou_log_aug_xai_reg.npy', val_iou_log)
    print(f'Train IOU: {train_iou_log}, Val IOU: {val_iou_log}')

    # Load model before eval
    model = torch.load('ckpt_xai_log_l2.pth', map_location=DEVICE)

    # Evaluate on the test set
    evaluate_and_visualize(model, test_loader, alignment_weight=ALIGNMENT_WEIGHT)
    # Print test loss and test IoU
    evaluate_test_set(model, test_loader, loss, metrics, DEVICE)