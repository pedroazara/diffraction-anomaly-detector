import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    fbeta_score,
    jaccard_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import (
    CHECKPOINTS_DIR,
    CLASSES,
    DEVICE,
    LABELS_TEST_CSV,
    LABELS_TRAIN_CSV,
    LABELS_VAL_CSV,
    ROOT_DIR,
)
from src.dataset import RefleXDataset
from src.model import build_model
from src.transforms import eval_transform

FIGURES_DIR = ROOT_DIR / "reports" / "figures"

PALETTE = {"bg": "#1a2332", "grid": "#2d3b52", "accent": "#5b9bd5", "text": "#e8edf4", "muted": "#8a97ab"}

plt.rcParams.update({
    "figure.facecolor": PALETTE["bg"], "axes.facecolor": PALETTE["bg"],
    "savefig.facecolor": PALETTE["bg"], "axes.edgecolor": PALETTE["grid"],
    "axes.labelcolor": PALETTE["text"], "text.color": PALETTE["text"],
    "xtick.color": PALETTE["muted"], "ytick.color": PALETTE["muted"],
    "grid.color": PALETTE["grid"], "font.size": 10,
})


@torch.no_grad()
def predict_probs(model, dataset, batch_size=32):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    probs, targets = [], []
    for images, labels in loader:
        probs.append(torch.sigmoid(model(images.to(DEVICE))).cpu())
        targets.append(labels)
    return torch.cat(probs).numpy(), torch.cat(targets).numpy().astype(int)


def best_thresholds(probs, targets):
    """Per-class threshold maximizing F1. Must be computed on validation only."""
    ths = []
    for j in range(probs.shape[1]):
        if targets[:, j].sum() == 0:
            ths.append(0.5)
            continue
        prec, rec, th = precision_recall_curve(targets[:, j], probs[:, j])
        f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-9, None)
        ths.append(float(th[np.argmax(f1)]))
    return np.array(ths)


def report(probs, targets, ths):
    rows = []
    for j, cls in enumerate(CLASSES):
        has_pos = targets[:, j].sum() > 0
        rows.append(dict(
            classe=cls,
            positivos=int(targets[:, j].sum()),
            AP=average_precision_score(targets[:, j], probs[:, j]) if has_pos else np.nan,
            F1_0_5=f1_score(targets[:, j], probs[:, j] >= 0.5, zero_division=0),
            limiar=ths[j],
            F1_limiar=f1_score(targets[:, j], probs[:, j] >= ths[j], zero_division=0),
        ))
    df = pd.DataFrame(rows).set_index("classe")
    df.loc["MEDIA"] = df.mean(numeric_only=True)
    return df.round(3)


def paper_metrics(probs, targets, ths):
    preds = (probs >= ths).astype(int)

    rows = []
    for j, cls in enumerate(CLASSES):
        y, p, s = targets[:, j], preds[:, j], probs[:, j]
        rows.append(dict(
            classe=cls,
            Acc=accuracy_score(y, p),
            MCC=matthews_corrcoef(y, p),
            JI=jaccard_score(y, p, zero_division=0),
            Prec=precision_score(y, p, zero_division=0),
            Rec=recall_score(y, p, zero_division=0),
            F1=f1_score(y, p, zero_division=0),
            AUC=roc_auc_score(y, s),
            AUC_bin=roc_auc_score(y, p),  # como no artigo: AUC de predições 0/1 = acurácia balanceada
        ))
    per_class = pd.DataFrame(rows).set_index("classe").round(3)

    aggregate = pd.Series(dict(
        Prec_M=precision_score(targets, preds, average="macro", zero_division=0),
        Rec_M=recall_score(targets, preds, average="macro", zero_division=0),
        F1_M=f1_score(targets, preds, average="macro", zero_division=0),
        F2_M=fbeta_score(targets, preds, beta=2, average="macro", zero_division=0),
        MCC_M=np.mean([matthews_corrcoef(targets[:, j], preds[:, j]) for j in range(len(CLASSES))]),
        # imagem sem anomalia prevista sem anomalia conta como acerto (1), não como 0
        JI=jaccard_score(targets, preds, average="samples", zero_division=1),
        EMR=accuracy_score(targets, preds),
    )).round(3)
    return per_class, aggregate


def plot_class_frequency(path: Path):
    train_df = pd.read_csv(LABELS_TRAIN_CSV)
    freq = (train_df[CLASSES].mean() * 100).sort_values()
    counts = train_df[CLASSES].sum().astype(int)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(np.arange(len(freq)), freq.values, color=PALETTE["accent"], height=0.6)
    ax.set_yticks(np.arange(len(freq)), freq.index)
    ax.set_xlabel("% de imagens positivas (train)")
    ax.set_title("Frequência de cada anomalia (desbalanceamento)")
    for i, cls in enumerate(freq.index):
        ax.text(freq[cls] + 1, i, f"{freq[cls]:.1f}%  (n={counts[cls]})", va="center", fontsize=8.5,
                color=PALETTE["text"])
    ax.set_xlim(0, 75)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cooccurrence(path: Path):
    train_df = pd.read_csv(LABELS_TRAIN_CSV)
    Y = train_df[CLASSES].values
    co = Y.T @ Y
    cond = co / np.diag(co)[:, None]

    fig, ax = plt.subplots(figsize=(6.5, 5.8))
    im = ax.imshow(cond, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{cond[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if cond[i, j] > 0.5 else PALETTE["text"])
    ax.set_title("P(coluna | linha) — co-ocorrência de anomalias")
    plt.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pr_curves(probs, targets, path: Path):
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
    axes = axes.ravel()
    for j, cls in enumerate(CLASSES):
        ax = axes[j]
        ap = float("nan")
        if targets[:, j].sum():
            prec, rec, _ = precision_recall_curve(targets[:, j], probs[:, j])
            ax.plot(rec, prec, color=PALETTE["accent"], linewidth=2)
            ap = average_precision_score(targets[:, j], probs[:, j])
        ax.set_title(f"{cls}\nAP={ap:.2f}", fontsize=10)
        ax.set_xlabel("recall")
        ax.set_ylabel("precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
    axes[-1].axis("off")
    fig.suptitle("Curvas Precision-Recall por classe — conjunto de TESTE (held-out)", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_error_leak(probs, targets, ths, path: Path):
    preds = (probs >= ths).astype(int)
    leak = np.zeros((len(CLASSES), len(CLASSES)))
    for i in range(len(CLASSES)):
        fn = (targets[:, i] == 1) & (preds[:, i] == 0)
        if fn.sum():
            leak[i] = (preds[fn] * (targets[fn] == 0)).sum(0) / fn.sum()

    fig, ax = plt.subplots(figsize=(6.5, 5.8))
    im = ax.imshow(leak, cmap="Reds", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("predita indevidamente")
    ax.set_ylabel("verdadeira (perdida)")
    ax.set_title("Falsos negativos da linha -> falsos positivos da coluna (teste)")
    plt.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Avalia um checkpoint na validação e no teste.")
    parser.add_argument("--checkpoint", default="baseline_resnet50_frozen.pt")
    parser.add_argument("--figures", action="store_true",
                        help="também exporta as figuras em reports/figures/<checkpoint>/")
    args = parser.parse_args()
    figures_dir = FIGURES_DIR / Path(args.checkpoint).stem

    model = build_model(pretrained=False, freeze_backbone=False).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINTS_DIR / args.checkpoint, map_location=DEVICE))
    model.eval()

    val_ds = RefleXDataset(LABELS_VAL_CSV, transform=eval_transform)
    test_ds = RefleXDataset(LABELS_TEST_CSV, transform=eval_transform)

    p_val, y_val = predict_probs(model, val_ds)
    p_test, y_test = predict_probs(model, test_ds)

    thresholds = best_thresholds(p_val, y_val)
    print("\n=== VALIDAÇÃO ===")
    print(report(p_val, y_val, thresholds))
    print("\n=== TESTE (held-out, limiares fixados na validação) ===")
    print(report(p_test, y_test, thresholds))

    per_class, aggregate = paper_metrics(p_test, y_test, thresholds)
    print("\n=== TESTE — métricas do artigo (Tabela 5: por classe, Tabela 3: agregadas) ===")
    print(per_class)
    print()
    print(aggregate.to_string())

    if args.figures:
        figures_dir.mkdir(parents=True, exist_ok=True)
        plot_class_frequency(figures_dir / "class_freq.png")
        plot_cooccurrence(figures_dir / "cooccurrence.png")
        plot_pr_curves(p_test, y_test, figures_dir / "pr_curves_test.png")
        plot_error_leak(p_test, y_test, thresholds, figures_dir / "error_confusion_test.png")
        print(f"\nFiguras salvas em {figures_dir}")


if __name__ == "__main__":
    main()
