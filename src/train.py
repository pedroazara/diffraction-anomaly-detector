import argparse
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import MultilabelAveragePrecision, MultilabelF1Score
from tqdm import tqdm

from src.config import (
    BATCH_SIZE,
    CHECKPOINTS_DIR,
    CLASSES,
    DEVICE,
    LABELS_TRAIN_CSV,
    LABELS_VAL_CSV,
    LEARNING_RATE,
    NUM_EPOCHS,
    NUM_WORKERS,
    ROOT_DIR,
    WEIGHT_DECAY,
)
from src.dataset import RefleXDataset
from src.model import build_model
from src.transforms import eval_transform, train_transform


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_pos_weight(dataset: RefleXDataset) -> torch.Tensor:
    labels = torch.tensor(dataset.df[CLASSES].values.astype("float32"))
    positives = labels.sum(dim=0)
    negatives = len(dataset) - positives
    return (negatives / positives.clamp(min=1)).to(DEVICE)


def run_epoch(model, loader, criterion, optimizer=None, amp=False):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    ap_metric = MultilabelAveragePrecision(num_labels=len(CLASSES), average=None).to(DEVICE)
    f1_metric = MultilabelF1Score(num_labels=len(CLASSES), average=None).to(DEVICE)

    with torch.set_grad_enabled(is_train):
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logits = model(images)
                loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            ap_metric.update(logits.float(), labels.long())
            f1_metric.update(logits.float(), labels.long())

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, ap_metric.compute(), f1_metric.compute()


def build_optimizer(model, lr, backbone_lr, weight_decay):
    head_params = list(model.fc.parameters())
    if backbone_lr is None:
        return torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay
        )

    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    return torch.optim.AdamW(
        [{"params": backbone_params, "lr": backbone_lr}, {"params": head_params, "lr": lr}],
        lr=lr,
        weight_decay=weight_decay,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Treina o classificador multirrótulo de anomalias.")
    parser.add_argument("--name", default="best_model", help="nome do checkpoint salvo em checkpoints/")
    parser.add_argument("--unfreeze", action="store_true", help="treina o backbone inteiro (fine-tuning)")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="learning rate da cabeça")
    parser.add_argument("--backbone-lr", type=float, default=None,
                        help="learning rate do backbone; só faz sentido com --unfreeze")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--amp", action="store_true", help="autocast bfloat16 (mais rápido em GPU Ada)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    train_ds = RefleXDataset(LABELS_TRAIN_CSV, transform=train_transform)
    val_ds = RefleXDataset(LABELS_VAL_CSV, transform=eval_transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    model = build_model(pretrained=True, freeze_backbone=not args.unfreeze).to(DEVICE)

    pos_weight = compute_pos_weight(train_ds)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = build_optimizer(model, args.lr, args.backbone_lr, args.weight_decay)

    print(f"Experimento '{args.name}': backbone {'treinável' if args.unfreeze else 'congelado'}, "
          f"lr={args.lr}, backbone_lr={args.backbone_lr}, epochs={args.epochs}, amp={args.amp}, "
          f"seed={args.seed}")

    best_val_ap = 0.0
    history = []
    CHECKPOINTS_DIR.mkdir(exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_ap, train_f1 = run_epoch(model, train_loader, criterion, optimizer, amp=args.amp)
        val_loss, val_ap, val_f1 = run_epoch(model, val_loader, criterion, amp=args.amp)

        mean_val_ap = val_ap.mean().item()
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_mAP={mean_val_ap:.4f}")
        for cls, ap, f1 in zip(CLASSES, val_ap.tolist(), val_f1.tolist()):
            print(f"    {cls:22s} AP={ap:.3f}  F1={f1:.3f}")

        history.append(dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                            train_mAP=train_ap.mean().item(), val_mAP=mean_val_ap,
                            **{f"AP_{c}": v for c, v in zip(CLASSES, val_ap.tolist())}))

        if mean_val_ap > best_val_ap:
            best_val_ap = mean_val_ap
            torch.save(model.state_dict(), CHECKPOINTS_DIR / f"{args.name}.pt")
            print(f"  -> saved new best model (mAP={best_val_ap:.4f})")

    history_path = ROOT_DIR / "reports" / f"history_{args.name}.csv"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_path, index=False)
    print(f"\nMelhor val_mAP: {best_val_ap:.4f}. Histórico em {history_path}")


if __name__ == "__main__":
    main()
