import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_grad_cam.utils.image import show_cam_on_image
from tqdm import tqdm

from utils import DEVICE


def dice(a, b):
    return 2 * (a & b).sum() / (a.sum() + b.sum())


def generate_masks(n_masks, input_size, p1=0.1, initial_mask_size=(7, 7), binary=True):
    """
    Generate masks for RISE
    :param n_masks:
    :param input_size:
    :param p1:
    :param initial_mask_size:
    :param binary:
    :return:
    """
    # cell size in the upsampled mask
    Ch = np.ceil(input_size[0] / initial_mask_size[0])
    Cw = np.ceil(input_size[1] / initial_mask_size[1])

    resize_h = int((initial_mask_size[0] + 1) * Ch)
    resize_w = int((initial_mask_size[1] + 1) * Cw)

    masks = []

    for _ in range(n_masks):
        # generate binary mask
        binary_mask = torch.randn(
            1, 1, initial_mask_size[0], initial_mask_size[1])
        binary_mask = (binary_mask < p1).float()

        # upsampling mask
        if binary:
            mask = F.interpolate(
                binary_mask, (resize_h, resize_w), mode='nearest')  # , align_corners=False)
        else:
            mask = F.interpolate(
                binary_mask, (resize_h, resize_w), mode='bilinear', align_corners=False)

        # random cropping
        i = np.random.randint(0, Ch)
        j = np.random.randint(0, Cw)
        mask = mask[:, :, i:i + input_size[0], j:j + input_size[1]]

        masks.append(mask)

    masks = torch.cat(masks, dim=0)  # (N_masks, 1, H, W)

    return masks


def rise_segmentation(masks, image, model, fig_size, target=None, n_masks=None, box=None, DEVICE='cpu',
                      vis=True, vis_skip=1):
    input_tensor = preprocess(image)

    if box is None:
        y_start, y_end, x_start, x_end = 0, image.shape[0], 0, image.shape[1]
    else:
        y_start, y_end, x_start, x_end = box[0], box[1], box[2], box[3]

    coef = []

    if n_masks is None:
        n_masks = len(masks)

    output = model(input_tensor.unsqueeze(0).to(DEVICE))[0].detach().cpu()
    output_1 = output.argmax(axis=0)
    output_a = output_1[y_start:y_end, x_start:x_end]

    if target is None:
        target = output_a.max().item()

    for index, mask in tqdm(enumerate(masks[:n_masks])):
        # input_tensor = preprocess_transform(image)
        # output = model(input_tensor.unsqueeze(0).to(DEVICE))[0].detach().cpu()
        # output_1 = output.argmax(axis = 0)

        input_tensor_1 = input_tensor * mask
        output = model(input_tensor_1.unsqueeze(0).to(DEVICE))[0].detach().cpu()
        output_2 = output.argmax(axis=0)

        # output_a = output_1[y_start:y_end, x_start:x_end]
        #         if target is None:
        #             target = output_a.max().item()
        output_b = output_2[y_start:y_end, x_start:x_end]

        DICE = dice(output_a == target, output_b == target)
        coef.append(DICE)

        if vis:
            if index % vis_skip == 0:
                plt.figure(figsize=fig_size)
                plt.subplot(1, 3, 1)
                plt.imshow(output_1)
                plt.subplot(1, 3, 2)
                plt.imshow(output_2)
                plt.subplot(1, 3, 3)
                plt.imshow(mask[0])
                plt.show()
    return coef


def rise_aggregated(image, masks, coef, fig_size, fig_name=None, vis=True):
    aggregated_mask = np.zeros(masks[0][0].shape)

    for i, j in zip(masks[:len(coef)], coef):
        aggregated_mask += i[0].detach().cpu().numpy() * j.item()

    max_, min_ = aggregated_mask.max(), aggregated_mask.min()
    aggregated_mask = np.uint8(255 * (aggregated_mask - min_) / (max_ - min_))
    explanation_map = show_cam_on_image(image / 255, aggregated_mask / 255, use_rgb=True)
    return explanation_map


def vis_predict(image, model, fig_size, DEVICE='cpu', mask=None, box=None, fig_name=None, vis=True):
    input_tensor = preprocess(image)

    if mask is not None:
        input_tensor = input_tensor * mask
    output = model(input_tensor.unsqueeze(0).to(DEVICE))[0].detach().cpu()
    output = output.argmax(axis=0)

    if box is None:
        box = (0, 0, 0, 0)
    rect_image = image.copy()
    rect_image = cv2.rectangle(rect_image, (box[2], box[0]), (box[3], box[1]), 255, 10)

    plt.figure(figsize=fig_size)
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.subplot(1, 3, 2)
    plt.imshow(output)
    plt.title('To Explain')
    plt.subplot(1, 3, 3)
    plt.imshow(rect_image)
    if fig_name is not None:
        plt.savefig(fig_name, bbox_inches='tight')
    if vis is True:
        plt.show()
    plt.close()
    return image, output, rect_image


def visualize_algos(image, model, preprocess_transform=None, target=None,
                    target_layer=None, box=None, DEVICE='cpu',
                    method_indexes=[0],
                    vis_base=True, vis=True,
                    pool_sizes=[], pool_modes=[], reshape_transformer=False,
                    n_masks=0, input_size=None, p1=0.1,
                    initial_mask_size=(7, 7), image_vis=None, vis_skip=20,
                    vis_rise=False):
    im, pred, rect = vis_predict(image, model, preprocess_transform=preprocess_transform,
                                 DEVICE=DEVICE, mask=None, box=box, vis=vis)

    results = [rect, pred]
    results_masks = []
    for i in range(len(method_indexes)):
        if len(pool_sizes) - 1 < i:
            pool_size = None
        else:
            pool_size = pool_sizes[i]
        if len(pool_modes) - 1 < i:
            pool_mode = np.max
        else:
            pool_mode = pool_modes[i]
        mask, overlay = seg_grad_cam(image, model, preprocess_transform=preprocess_transform,
                                     target=target, target_layer=target_layer,
                                     box=box, DEVICE=DEVICE, vis_base=vis_base,
                                     vis=vis, method_index=method_indexes[i],
                                     reshape_transformer=reshape_transformer,
                                     pool_size=pool_size, pool_mode=pool_mode)
        results.append(overlay)
        results_masks.append(mask)
    if n_masks != 0:
        masks = generate_masks(n_masks=n_masks, input_size=input_size, p1=p1, initial_mask_size=initial_mask_size)
        coef = rise_segmentation(masks, image, model, preprocess_transform=preprocess_transform,
                                 target=target, box=box, DEVICE=DEVICE, vis=vis_rise, vis_skip=vis_skip)
        mask, overlay = rise_aggregated(image_vis, masks, coef, vis=vis_rise)
        results.append(overlay)
        results_masks.append(mask)
    return results, results_masks


def rise_explainer(input_model, input_image):
    np_input_image = np.array(input_image)  # np.array image
    n_masks = 2000  # Number of generated masks
    p1 = 0.1
    # window_size = (7, 7)
    input_size = np_input_image.shape
    # pool_sizes, pool_modes, reshape_transformer = [0, 1, 2], [None, np.mean, np.mean], False
    fig_size = (30, 50)
    vis, vis_base, vis_rise, grid = True, True, True, True
    initial_mask_size = (7, 7)
    vis_skip = 100
    target = 2
    box = None
    masks = generate_masks(n_masks=n_masks, input_size=input_size, p1=p1, initial_mask_size=initial_mask_size)
    coef = rise_segmentation(masks, np_input_image, input_model, fig_size=fig_size,
                             target=target, box=box, DEVICE=DEVICE, vis=vis_rise, vis_skip=vis_skip)
    explanation_map = rise_aggregated(np_input_image, masks, coef, fig_size, vis=vis_rise)
    return explanation_map
