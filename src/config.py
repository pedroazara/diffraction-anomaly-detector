from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
IMAGES_DIR = RAW_DIR / "reflex_img_1024_inter_nearest"
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"

LABELS_TRAIN_CSV = RAW_DIR / "labels_train.csv"
LABELS_VAL_CSV = RAW_DIR / "labels_val.csv"
LABELS_TEST_CSV = RAW_DIR / "labels_test.csv"

CLASSES = [
    "ice_ring",
    "diffuse_scattering",
    "background_ring",
    "non_uniform_detector",
    "loop_scattering",
    "strong_background",
    "artifact",
]

IMAGE_SIZE = 512
BATCH_SIZE = 16
NUM_WORKERS = 4
NUM_EPOCHS = 30
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
