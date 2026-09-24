import argparse
from pathlib import Path

import matplotlib
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import (
    CHECKPOINTS_DIR,
    CLASSES,
    DEVICE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    LABELS_TEST_CSV,
    ROOT_DIR,
)
from src.dataset import RefleXDataset
from src.model import build_model
from src.transforms import eval_transform

FIGURES_DIR = ROOT_DIR / "reports" / "figures"
DEFAULT_CLASSES = ["ice_ring", "loop_scattering", "non_uniform_detector", "strong_background"]

PALETTE = {"bg": "#1a2332", "text": "#e8edf4", "muted": "#8a97ab", "accent": "#5b9bd5"}

plt.rcParams.update({
    "figure.facecolor": PALETTE["bg"], "axes.facecolor": PALETTE["bg"],
    "savefig.facecolor": PALETTE["bg"], "text.color": PALETTE["text"], "font.size": 10,
})


def denormalize(tensor):
    mean = torch.tensor(IMAGENET_MEAN)[:, None, None]
    std = torch.tensor(IMAGENET_STD)[:, None, None]
    return (tensor.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


@torch.no_grad()
def predict_probs(model, dataset, batch_size=32):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    probs, targets = [], []
    for images, labels in loader:
        probs.append(torch.sigmoid(model(images.to(DEVICE))).cpu())
        targets.append(labels)
    return torch.cat(probs).numpy(), torch.cat(targets).numpy().astype(int)


def main():
    parser = argparse.ArgumentParser(description="Gera mapas Grad-CAM para exemplos do conjunto de teste.")
    parser.add_argument("--checkpoint", default="baseline_resnet50_frozen.pt")
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    parser.add_argument("--out", default="gradcam.png")
    args = parser.parse_args()
    figures_dir = FIGURES_DIR / Path(args.checkpoint).stem

    model = build_model(pretrained=False, freeze_backbone=False).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINTS_DIR / args.checkpoint, map_location=DEVICE))
    model.eval()

    dataset = RefleXDataset(LABELS_TEST_CSV, transform=eval_transform)
    probs, targets = predict_probs(model, dataset)

    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])

    n = len(args.classes)
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, 4, figsize=(3.2 * 4, 3.9 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")

    for k, cls in enumerate(args.classes):
        j = CLASSES.index(cls)
        candidates = np.where(targets[:, j] == 1)[0]
        if len(candidates) == 0:
            continue
        idx = int(candidates[np.argmax(probs[candidates, j])])

        image, _ = dataset[idx]
        rgb = denormalize(image)
        grayscale_cam = cam(
            input_tensor=image.unsqueeze(0).to(DEVICE),
            targets=[ClassifierOutputTarget(j)],
        )[0]
        overlay = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)

        r, c = k // 2, (k % 2) * 2
        axes[r][c].imshow(rgb)
        axes[r][c].set_title(f"{cls}\noriginal", fontsize=10, color=PALETTE["muted"])
        axes[r][c + 1].imshow(overlay)
        axes[r][c + 1].set_title(f"Grad-CAM  p={probs[idx, j]:.2f}", fontsize=10, color=PALETTE["accent"])

    fig.suptitle("Onde o modelo olha: Grad-CAM sobre a última camada convolucional", y=1.0, fontsize=13)
    fig.tight_layout(h_pad=3.0)
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / args.out
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"Figura salva em {out_path}")


if __name__ == "__main__":
    main()
