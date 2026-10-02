import torch
from torchvision.transforms import v2

from src.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

AUGMENTATIONS = ("rotate10", "dihedral")


class RandomRot90(torch.nn.Module):
    def forward(self, img):
        return torch.rot90(img, k=int(torch.randint(4, (1,))), dims=(-2, -1))


def build_train_transform(aug: str = "rotate10"):
    if aug == "rotate10":
        geometric = [v2.RandomHorizontalFlip(), v2.RandomVerticalFlip(), v2.RandomRotation(degrees=10)]
        tensor_geometric = []
    elif aug == "dihedral":
        # flips + 90° rotations: the 8 symmetries of the square, no interpolation or black corners
        geometric = [v2.RandomHorizontalFlip()]
        tensor_geometric = [RandomRot90()]
    else:
        raise ValueError(f"unknown augmentation: {aug} (options: {AUGMENTATIONS})")

    return v2.Compose([
        v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        *geometric,
        v2.ColorJitter(brightness=0.1, contrast=0.1),
        v2.ToImage(),
        *tensor_geometric,
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


train_transform = build_train_transform("rotate10")

eval_transform = v2.Compose([
    v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])
