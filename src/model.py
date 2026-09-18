import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

from src.config import CLASSES


def build_model(pretrained: bool = True, freeze_backbone: bool = True) -> nn.Module:
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    return model
