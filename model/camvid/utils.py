import os
import torch
from pytorch_grad_cam import *

ROOT_DIR = "data/sample_data/CamVid/"
x_train_dir = os.path.join(ROOT_DIR, 'train')
y_train_dir = os.path.join(ROOT_DIR, 'trainannot')

x_valid_dir = os.path.join(ROOT_DIR, 'val')
y_valid_dir = os.path.join(ROOT_DIR, 'valannot')

x_test_dir = os.path.join(ROOT_DIR, 'test')
y_test_dir = os.path.join(ROOT_DIR, 'testannot')

EPOCHS = 1000

CLASSES = ['sky', 'building', 'pole', 'road', 'pavement',
           'tree', 'signsymbol', 'fence', 'car',
           'pedestrian', 'bicyclist', 'unlabelled']

ENCODER = 'resnet101'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATIONS = 'softmax2d'

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps'  # or 'cpu'

MODEL_PATH = 'model/camvid/model_ResNet101.pth'

XAI_METHODS = [GradCAM, GradCAMPlusPlus, EigenCAM, EigenGradCAM, ScoreCAM, HiResCAM, AblationCAM, XGradCAM, LayerCAM,
               FullGrad]

ALIGNMENT_WEIGHT = 0.5