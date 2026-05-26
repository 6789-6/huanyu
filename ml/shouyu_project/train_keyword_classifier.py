"""Train a CE-CSL keyword detector from the existing keypoint dataset.

This is the short-deadline path: train a small multi-label classifier over
high-frequency gloss tokens instead of a 3846-class CTC sequence model.
"""
import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_keypoint import KP_DIM
from dataset_keyword import KeywordKeypointDataset, collate_keyword_batch
from model_keyword import KeywordClassifier


SEED = 42
BATCH_SIZE = 4
NUM_EPOCHS = 40
LR = 5e-4
WEIGHT_DECAY = 1e-4
NUM_FRAMES = 64
TOP_K = 80
THROTTLE_SECONDS = 0.08
NUM_THREADS = 4
PATIENCE = 8
DEVICE = torch.device("cpu")
SHOUYU_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = SHOUYU_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def compute_pos_weight(dataset: KeywordKeypointDataset) -> torch.Tensor:
    counts = torch.zeros(len(dataset.token_to_idx), dtype=torch.float32)
    for sample in dataset.samples:
        for idx in sample["label_indices"]:
            counts[idx] += 1.0
    total = float(len(dataset))
    negatives = torch.clamp(torch.full_like(counts, total) - counts, min=1.0)
    positives = counts.clamp_min(1.0)
    return (negatives / positives).clamp(max=20.0)


def limit_dataset(dataset: KeywordKeypointDataset, max_samples: int) -> KeywordKeypointDataset:
    if max_samples > 0 and len(dataset) > max_samples:
        dataset.samples = dataset.samples[:max_samples]
        dataset.base_indices = dataset.base_indices[:max_samples]
    return dataset


def keyword_metrics(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    pred = torch.sigmoid(logits) >= threshold
    target = labels.bool()
    tp = (pred & target).sum().item()
    fp = (pred & ~target).sum().item()
    fn = (~pred & target).sum().item()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    ker = (fp + fn) / max(int(target.sum().item()), 1)
    exact = (pred == target).all(dim=1).float().mean().item() if labels.numel() else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "ker": ker, "exact": exact}


def sweep_threshold_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    thresholds: list[float] | None = None,
) -> tuple[float, dict[str, float]]:
    if thresholds is None:
        thresholds = [round(0.1 + i * 0.05, 2) for i in range(17)]
    best_threshold = thresholds[0]
    best_metrics = keyword_metrics(logits, labels, threshold=best_threshold)
    for threshold in thresholds[1:]:
        metrics = keyword_metrics(logits, labels, threshold=threshold)
        candidate = (metrics["f1"], metrics["exact"], metrics["precision"])
        current = (best_metrics["f1"], best_metrics["exact"], best_metrics["precision"])
        if candidate > current:
            best_threshold = threshold
            best_metrics = metrics
    best_metrics = dict(best_metrics)
    best_metrics["threshold"] = best_threshold
    return best_threshold, best_metrics


@torch.no_grad()
def evaluate(
    model: KeywordClassifier,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    threshold: float,
    sweep_threshold: bool = True,
) -> tuple[float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_rows = 0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    for batch in dataloader:
        keypoints = batch["keypoints"].to(DEVICE)
        valid_mask = batch["valid_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        logits = model(keypoints, valid_mask)
        loss = criterion(logits, labels)
        total_loss += float(loss.item()) * labels.size(0)
        total_rows += labels.size(0)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    if not all_logits:
        return 0.0, {"precision": 0.0, "recall": 0.0, "f1": 0.0, "exact": 0.0}
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    if sweep_threshold:
        _, metrics = sweep_threshold_metrics(logits, labels)
    else:
        metrics = keyword_metrics(logits, labels, threshold=threshold)
        metrics["threshold"] = threshold
    return total_loss / max(total_rows, 1), metrics


def save_keyword_checkpoint(
    path: str | Path,
    model: KeywordClassifier,
    token_to_idx: dict[str, int],
    epoch: int,
    best_f1: float,
    best_threshold: float = 0.5,
) -> None:
    idx_to_token = {idx: token for token, idx in token_to_idx.items()}
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model.config(),
            "token_to_idx": dict(token_to_idx),
            "idx_to_token": idx_to_token,
            "epoch": epoch,
            "best_f1": best_f1,
            "best_threshold": best_threshold,
        },
        str(path),
    )


def load_keyword_checkpoint(path: str | Path) -> tuple[KeywordClassifier, dict]:
    state = torch.load(str(path), map_location="cpu")
    model = KeywordClassifier(**state["model_config"])
    model.load_state_dict(state["model_state_dict"], strict=True)
    meta = {
        "token_to_idx": state["token_to_idx"],
        "idx_to_token": state["idx_to_token"],
        "epoch": state.get("epoch", 0),
        "best_f1": state.get("best_f1", 0.0),
        "best_threshold": state.get("best_threshold", 0.5),
    }
    return model, meta


def train(args) -> Path:
    global DEVICE
    DEVICE = resolve_device(args.device)
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
    seed_everything(SEED)
    frames = None if args.frames <= 0 else args.frames

    train_ds = KeywordKeypointDataset(
        split="train",
        top_k=args.top_k,
        min_freq=args.min_freq,
        augment=args.augment,
        num_frames=frames,
        drop_empty=True,
    )
    limit_dataset(train_ds, args.max_train_samples)
    dev_ds = KeywordKeypointDataset(
        split="dev",
        token_to_idx=train_ds.token_to_idx,
        augment=False,
        num_frames=frames,
        drop_empty=True,
    )
    limit_dataset(dev_ds, args.max_dev_samples)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_keyword_batch,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda",
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_keyword_batch,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda",
    )

    model = KeywordClassifier(
        kp_dim=KP_DIM,
        num_keywords=len(train_ds.token_to_idx),
        d_model=args.d_model,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
    ).to(DEVICE)
    pos_weight = compute_pos_weight(train_ds).to(DEVICE)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    best_f1 = -1.0
    epochs_without_improvement = 0
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE} (torch_threads={torch.get_num_threads()})")
    print(
        f"Keyword dataset: train={len(train_ds)} dev={len(dev_ds)} "
        f"keywords={len(train_ds.token_to_idx)} frames={frames or 'full'}"
    )
    print(f"Top keywords: {list(train_ds.token_to_idx)[:20]}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        all_logits: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            keypoints = batch["keypoints"].to(DEVICE)
            valid_mask = batch["valid_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            logits = model(keypoints, valid_mask)
            loss = criterion(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += float(loss.item()) * labels.size(0)
            total_rows += labels.size(0)
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())
            metrics = keyword_metrics(torch.cat(all_logits), torch.cat(all_labels), threshold=args.threshold)
            pbar.set_postfix(
                loss=f"{total_loss / max(total_rows, 1):.4f}",
                f1=f"{metrics['f1']:.2%}",
            )
            if args.throttle_seconds > 0:
                time.sleep(args.throttle_seconds)

        scheduler.step()
        train_metrics = keyword_metrics(torch.cat(all_logits), torch.cat(all_labels), threshold=args.threshold)
        dev_loss, dev_metrics = evaluate(
            model,
            dev_loader,
            criterion,
            threshold=args.threshold,
            sweep_threshold=not args.no_threshold_sweep,
        )
        print(
            f"Epoch {epoch:03d} | train_loss={total_loss / max(total_rows, 1):.4f} "
            f"train_f1={train_metrics['f1']:.2%} train_ker={train_metrics['ker']:.2%} "
            f"train_recall={train_metrics['recall']:.2%} | "
            f"dev_loss={dev_loss:.4f} dev_f1={dev_metrics['f1']:.2%} "
            f"dev_ker={dev_metrics['ker']:.2%} dev_p={dev_metrics['precision']:.2%} "
            f"dev_r={dev_metrics['recall']:.2%} "
            f"thr={dev_metrics['threshold']:.2f}"
        )

        if dev_metrics["f1"] > best_f1:
            best_f1 = dev_metrics["f1"]
            epochs_without_improvement = 0
            save_keyword_checkpoint(
                out_path,
                model,
                train_ds.token_to_idx,
                epoch,
                best_f1,
                best_threshold=dev_metrics["threshold"],
            )
            print(f"  saved {out_path}")
        else:
            epochs_without_improvement += 1
            if args.patience > 0 and epochs_without_improvement >= args.patience:
                print(f"Early stopping after {args.patience} epochs without dev F1 improvement.")
                break

    return out_path


def parse_args():
    parser = argparse.ArgumentParser(description="Train CE-CSL keyword multi-label classifier")
    parser.add_argument("--output", default=str(OUTPUT_DIR / "best_keyword_classifier.pt"))
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-dev-samples", type=int, default=0)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--frames", type=int, default=NUM_FRAMES, help="0 keeps full sequences")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--no-threshold-sweep", action="store_true")
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--throttle-seconds", type=float, default=THROTTLE_SECONDS)
    parser.add_argument("--num-threads", type=int, default=NUM_THREADS)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--augment", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
