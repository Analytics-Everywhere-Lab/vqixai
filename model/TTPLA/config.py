import os
import torch

ROOT_DIR = "data/TTPLA/"
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

CLASSES = ['__background__', 'cable', 'tower_lattice', 'tower_tucohy', 'tower_wooden']

ENCODER = 'resnet101'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATIONS = 'softmax2d'

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps'  # or 'cpu'
