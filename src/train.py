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
    WEIGHT_DECAY,
)
from src.dataset import RefleXDataset
from src.model import build_model
from src.transforms import eval_transform, train_transform


def compute_pos_weight(dataset: RefleXDataset) -> torch.Tensor:
    labels = torch.tensor(dataset.df[CLASSES].values.astype("float32"))
    positives = labels.sum(dim=0)
    negatives = len(dataset) - positives
    return (negatives / positives.clamp(min=1)).to(DEVICE)


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    ap_metric = MultilabelAveragePrecision(num_labels=len(CLASSES), average=None).to(DEVICE)
    f1_metric = MultilabelF1Score(num_labels=len(CLASSES), average=None).to(DEVICE)

    with torch.set_grad_enabled(is_train):
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            ap_metric.update(logits, labels.long())
            f1_metric.update(logits, labels.long())

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, ap_metric.compute(), f1_metric.compute()


def main():
    train_ds = RefleXDataset(LABELS_TRAIN_CSV, transform=train_transform)
    val_ds = RefleXDataset(LABELS_VAL_CSV, transform=eval_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = build_model(pretrained=True, freeze_backbone=True).to(DEVICE)

    pos_weight = compute_pos_weight(train_ds)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_ap = 0.0
    CHECKPOINTS_DIR.mkdir(exist_ok=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_ap, train_f1 = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_ap, val_f1 = run_epoch(model, val_loader, criterion)

        mean_val_ap = val_ap.mean().item()
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_mAP={mean_val_ap:.4f}")
        for cls, ap, f1 in zip(CLASSES, val_ap.tolist(), val_f1.tolist()):
            print(f"    {cls:22s} AP={ap:.3f}  F1={f1:.3f}")

        if mean_val_ap > best_val_ap:
            best_val_ap = mean_val_ap
            torch.save(model.state_dict(), CHECKPOINTS_DIR / "best_model.pt")
            print(f"  -> saved new best model (mAP={best_val_ap:.4f})")


if __name__ == "__main__":
    main()
