import numpy as np
import cv2
import torch
import torch.nn.functional as F
from tqdm import trange
from matplotlib import pyplot as plt


# Function to compute normalized Area Under Curve
def auc(arr):
    """
    Returns normalized Area Under Curve of the array.
    arr: type(np.ndarray) - shape:[n]
    """
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)


# Plausibility Metrics

def ebpg_metric(gt_mask, saliency_map, target_class):
    """
    Compute the EBPG metric for segmentation.

    Parameters:
    - gt_mask: Ground truth mask, shape [H, W], with class labels.
    - saliency_map: Saliency map, shape [H, W], saliency values.
    - target_class: The class label for which to compute the metric.

    Returns:
    - ebpg_value: The EBPG value.
    """
    # Create a binary mask where the ground truth is equal to the target class
    gt_binary_mask = (gt_mask == target_class).astype(np.float32)

    # Sum of saliency map over the ground truth region
    energy_inside = np.sum(saliency_map * gt_binary_mask)

    # Total sum of saliency map
    total_energy = np.sum(saliency_map)

    # Handle division by zero
    if total_energy == 0:
        ebpg_value = 0.0
    else:
        ebpg_value = energy_inside / total_energy

    return ebpg_value


def iou_metric(gt_mask, saliency_map, target_class, threshold=0.5):
    """
    Compute the IoU between the saliency map and the ground truth mask for the target class.

    Parameters:
    - gt_mask: Ground truth mask, shape [H, W], with class labels.
    - saliency_map: Saliency map, shape [H, W], saliency values.
    - target_class: The class label for which to compute the metric.
    - threshold: Threshold to binarize the saliency map.

    Returns:
    - iou_value: The IoU value.
    """
    # Threshold the saliency map to create a binary mask
    saliency_binary_mask = (saliency_map >= threshold * np.max(saliency_map)).astype(np.float32)

    # Create a binary mask where the ground truth is equal to the target class
    gt_binary_mask = (gt_mask == target_class).astype(np.float32)

    # Compute intersection and union
    intersection = np.logical_and(saliency_binary_mask, gt_binary_mask).sum()
    union = np.logical_or(saliency_binary_mask, gt_binary_mask).sum()

    if union == 0:
        iou_value = 0.0
    else:
        iou_value = intersection / union

    return iou_value


def bbox_metric(gt_mask, saliency_map, target_class, threshold=0.5):
    """
    Compute the IoU between the bounding boxes of the saliency map and the ground truth mask for the target class.

    Parameters:
    - gt_mask: Ground truth mask, shape [H, W], with class labels.
    - saliency_map: Saliency map, shape [H, W], saliency values.
    - target_class: The class label for which to compute the metric.
    - threshold: Threshold to binarize the saliency map.

    Returns:
    - bbox_iou_value: The IoU of the bounding boxes.
    """
    # Threshold the saliency map to create a binary mask
    saliency_binary_mask = (saliency_map >= threshold * np.max(saliency_map)).astype(np.uint8)

    # Create a binary mask where the ground truth is equal to the target class
    gt_binary_mask = (gt_mask == target_class).astype(np.uint8)

    # Find bounding boxes
    # For saliency map
    saliency_coords = cv2.findNonZero(saliency_binary_mask)
    if saliency_coords is not None:
        x, y, w, h = cv2.boundingRect(saliency_coords)
        saliency_bbox = [x, y, x + w, y + h]
    else:
        saliency_bbox = [0, 0, 0, 0]

    # For ground truth
    gt_coords = cv2.findNonZero(gt_binary_mask)
    if gt_coords is not None:
        x, y, w, h = cv2.boundingRect(gt_coords)
        gt_bbox = [x, y, x + w, y + h]
    else:
        gt_bbox = [0, 0, 0, 0]

    # Compute IoU of bounding boxes
    xA = max(saliency_bbox[0], gt_bbox[0])
    yA = max(saliency_bbox[1], gt_bbox[1])
    xB = min(saliency_bbox[2], gt_bbox[2])
    yB = min(saliency_bbox[3], gt_bbox[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (saliency_bbox[2] - saliency_bbox[0]) * (saliency_bbox[3] - saliency_bbox[1])
    boxBArea = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])

    unionArea = boxAArea + boxBArea - interArea

    if unionArea == 0:
        bbox_iou_value = 0.0
    else:
        bbox_iou_value = interArea / unionArea

    return bbox_iou_value


# Faithfulness Metrics

def faithfulness_metrics(model, img, gt_mask, saliency_map, target_class, mode, step):
    """
    Compute deletion or insertion metrics for segmentation models.

    Parameters:
    - model: Segmentation model.
    - img: Input image, shape [H, W, 3].
    - gt_mask: Ground truth mask, shape [H, W], with class labels.
    - saliency_map: Saliency map, shape [H, W], saliency values.
    - target_class: The class label for which to compute the metric.
    - mode: 'del' or 'ins'.
    - step: Number of pixels modified per iteration.

    Returns:
    - auc_value: The AUC of h_i over steps.
    - h_i_values: The h_i values over steps.
    """
    H, W = saliency_map.shape

    # Flatten saliency map and get indices sorted in decreasing order of saliency
    saliency_flat = saliency_map.flatten()
    indices = np.argsort(-saliency_flat)

    # Number of steps
    n_steps = (H * W + step - 1) // step

    # Prepare modified image
    if mode == 'del':
        # Start with the original image
        modified_img = img.copy()
        # The baseline is zero (black pixels)
        baseline = np.zeros_like(img)
    elif mode == 'ins':
        # Start with a blurred image
        modified_img = cv2.GaussianBlur(img, (11, 11), 0)
        # The baseline is the original image
        baseline = img.copy()
    else:
        raise ValueError("Mode must be 'del' or 'ins'")

    h_i_values = []

    for i in range(n_steps + 1):
        # Prepare input tensor
        input_tensor = torch.from_numpy(modified_img.transpose(2, 0, 1)).unsqueeze(0).float()
        input_tensor = input_tensor.to(next(model.parameters()).device)

        # Pass through model
        with torch.no_grad():
            output = model(input_tensor)

        # Get predicted mask
        if isinstance(output, dict) and 'out' in output:
            pred_mask = output['out'].squeeze(0).cpu().numpy()
        else:
            pred_mask = output.squeeze(0).cpu().numpy()

        # Get predicted class probabilities
        if pred_mask.ndim == 3:
            # Multiple classes
            pred_probs = F.softmax(torch.from_numpy(pred_mask), dim=0).numpy()
            pred_class_mask = np.argmax(pred_probs, axis=0)
        else:
            # Binary segmentation
            pred_probs = torch.sigmoid(torch.from_numpy(pred_mask)).numpy()
            pred_class_mask = (pred_probs >= 0.5).astype(np.int)

        # Compute h_i, e.g., IoU between predicted mask and gt_mask for target_class
        pred_binary_mask = (pred_class_mask == target_class).astype(np.uint8)
        gt_binary_mask = (gt_mask == target_class).astype(np.uint8)

        intersection = np.logical_and(pred_binary_mask, gt_binary_mask).sum()
        union = np.logical_or(pred_binary_mask, gt_binary_mask).sum()
        if union == 0:
            h_i = 0.0
        else:
            h_i = intersection / union

        h_i_values.append(h_i)

        if i == n_steps:
            break

        # Modify the image for the next step
        idx = indices[step * i: step * (i + 1)]
        coords = np.unravel_index(idx, (H, W))

        if mode == 'del':
            # Set the pixels to baseline (black)
            modified_img[coords[0], coords[1], :] = baseline[coords[0], coords[1], :]
        else:
            # Restore the pixels from baseline (original image)
            modified_img[coords[0], coords[1], :] = baseline[coords[0], coords[1], :]

    # Compute AUC
    h_i_array = np.array(h_i_values)
    auc_value = auc(h_i_array)

    return auc_value, h_i_values


if __name__ == "__main__":
    # Test the metrics
    import segmentation_models_pytorch as smp

    # Assume you have a trained segmentation model
    model = smp.DeepLabV3Plus(
        encoder_name='resnet101',
        encoder_weights='imagenet',
        classes=5,  # Number of classes in your dataset
        activation=None,
    )
    model.eval()
    model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Load an example image and ground truth mask
    # Replace 'image_path' and 'mask_path' with your actual paths
    image = cv2.imread('image_path')  # Shape [H, W, 3]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gt_mask = cv2.imread('mask_path', cv2.IMREAD_GRAYSCALE)  # Shape [H, W]

    # Assume you have a saliency map for the target class
    # For demonstration, we create a random saliency map
    saliency_map = np.random.rand(gt_mask.shape[0], gt_mask.shape[1])

    # Define the target class label
    target_class = 1  # Replace with your target class label

    # Compute Plausibility Metrics
    ebpg_value = ebpg_metric(gt_mask, saliency_map, target_class)
    iou_value = iou_metric(gt_mask, saliency_map, target_class)
    bbox_iou_value = bbox_metric(gt_mask, saliency_map, target_class)

    print(f"EBPG: {ebpg_value}")
    print(f"IoU: {iou_value}")
    print(f"Bbox IoU: {bbox_iou_value}")

    # Compute Faithfulness Metrics
    deletion_auc, deletion_h_i = faithfulness_metrics(
        model, image, gt_mask, saliency_map, target_class, mode='del', step=500
    )
    insertion_auc, insertion_h_i = faithfulness_metrics(
        model, image, gt_mask, saliency_map, target_class, mode='ins', step=500
    )

    print(f"Deletion AUC: {deletion_auc}")
    print(f"Insertion AUC: {insertion_auc}")

    # Plot the Deletion and Insertion Curves
    plt.figure(figsize=(8, 6))
    plt.plot(np.linspace(0, 1, len(deletion_h_i)), deletion_h_i, label='Deletion')
    plt.plot(np.linspace(0, 1, len(insertion_h_i)), insertion_h_i, label='Insertion')
    plt.xlabel('Fraction of Pixels Modified')
    plt.ylabel('IoU with Ground Truth')
    plt.title('Faithfulness Metrics')
    plt.legend()
    plt.grid(True)
    plt.show()
