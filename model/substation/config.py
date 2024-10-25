import os
import torch
from pytorch_grad_cam import *

ROOT_DIR = "data/substation"
DATA_DIR = os.path.join(ROOT_DIR, "ds")
ANN_DIR = os.path.join(DATA_DIR, "ann")
IMG_DIR = os.path.join(DATA_DIR, "img")
TRAIN_DIR = os.path.join(ROOT_DIR, "train")
TEST_DIR = os.path.join(ROOT_DIR, "test")
INFERENCE_DIR = os.path.join(ROOT_DIR, "inference")
x_train_dir = os.path.join(TRAIN_DIR, 'img')
y_train_dir = os.path.join(TRAIN_DIR, 'ann')
x_valid_dir = os.path.join(TEST_DIR, 'img')
y_valid_dir = os.path.join(TEST_DIR, 'ann')

EPOCHS = 1000

CLASSES = ['breaker', 'closed_blade_disconnect_switch', 'closed_tandem_disconnect_switch', 'current_transformer',
           'fuse_disconnect_switch', 'glass_disc_insulator', 'lightning_arrester', 'muffle',
           'open_blade_disconnect_switch', 'open_tandem_disconnect_switch', 'porcelain_pin_insulator',
           'potential_transformer', 'power_transformer', 'recloser', 'tripolar_disconnect_switch']

ENCODER = 'resnet101'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATIONS = 'softmax2d'


MODEL_PATH = 'model/substation/model_ResNet101.pth'

XAI_METHODS = [GradCAM, GradCAMPlusPlus, EigenCAM, EigenGradCAM, ScoreCAM, HiResCAM, AblationCAM, XGradCAM, LayerCAM,
               FullGrad]
