"""Two-stage fine-tuning, checkpoint loading, held-out evaluation, and inference."""

from __future__ import annotations

import json
import logging
import math
import os
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.transforms import v2

from .data import ImageDataset, Manifest, build_manifest, image_paths, read_rgb
from .metrics import classification_metrics, fit_temperature
from .models import (
    Classifier,
    Preprocessing,
    build_model,
    make_transform,
    optimizer_groups,
    preprocessing_for,
)

LOGGER = logging.getLogger(__name__)


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def select_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        requested = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError("Device must be auto, cpu, cuda, or mps")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA is unavailable; install the matching PyTorch build or use --device cpu"
        )
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable on this machine; use --device cpu")
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


@dataclass
class TrainConfig:
    data: Path
    output: Path
    model: str = "efficientnet_v2_s"
    epochs: int = 12
    warmup_epochs: int = 2
    batch_size: int = 16
    image_size: int | None = None
    learning_rate: float = 3e-4
    backbone_lr_ratio: float = 0.1
    weight_decay: float = 0.01
    label_smoothing: float = 0.1
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 42
    workers: int = 0
    device: str = "auto"
    amp: bool = True
    pretrained: bool = True
    augment: bool = True
    hflip: bool = True
    mixup: bool = True
    balance_classes: bool = False
    calibrate: bool = True
    patience: int = 5

    def validate(self) -> None:
        if self.epochs < 1 or not 0 <= self.warmup_epochs < self.epochs:
            raise ValueError("epochs must be positive and warmup_epochs must be in [0, epochs)")
        if self.batch_size < 1 or self.workers < 0 or self.patience < 1:
            raise ValueError("batch_size/patience must be positive; workers cannot be negative")
        if not 0 <= self.seed < 2**32:
            raise ValueError("seed must be in [0, 2**32)")
        for name in ("learning_rate", "backbone_lr_ratio"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if not 0 <= self.label_smoothing < 1:
            raise ValueError("label_smoothing must be in [0, 1)")
        if self.output.resolve().is_relative_to(self.data.resolve()):
            raise ValueError("Put the run output outside the dataset directory")
        if self.output.exists() and (not self.output.is_dir() or any(self.output.iterdir())):
            raise ValueError("Output is not empty; choose a new run directory")


def make_loader(
    dataset: ImageDataset,
    config: TrainConfig,
    device: torch.device,
    *,
    training: bool = False,
) -> DataLoader:
    generator = torch.Generator().manual_seed(config.seed)
    sampler = None
    if training and config.balance_classes:
        sampler = WeightedRandomSampler(
            dataset.sampling_weights(), len(dataset), replacement=True, generator=generator
        )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=training and sampler is None,
        sampler=sampler,
        num_workers=config.workers,
        worker_init_fn=seed_worker,
        generator=generator,
        pin_memory=device.type == "cuda",
        persistent_workers=config.workers > 0,
        timeout=60 if config.workers > 0 else 0,
        # Forking after PyTorch initialized thread pools can deadlock on Linux.
        # Spawn also works with CUDA, macOS, and Windows.
        multiprocessing_context="spawn" if config.workers > 0 else None,
    )


@torch.inference_mode()
def collect_logits(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits, targets = [], []
    for images, labels in loader:
        logits.append(model(images.to(device)).float().cpu())
        targets.append(labels.cpu())
    if not logits:
        raise ValueError("Cannot evaluate an empty dataset")
    return torch.cat(logits), torch.cat(targets)


def _save_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path, device: torch.device) -> tuple[Classifier, dict, Preprocessing]:
    # Loading never downloads ImageNet weights and never enables arbitrary pickle objects.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint version")
    classes = checkpoint["classes"]
    temperature = checkpoint["temperature"]
    if len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError("Checkpoint needs at least two distinct class names")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Invalid checkpoint temperature")
    spec = Preprocessing(**checkpoint["preprocessing"])
    model = build_model(checkpoint["architecture"], len(classes), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint, spec


def train(config: TrainConfig) -> dict:
    config.validate()
    device = select_device(config.device)
    spec = preprocessing_for(config.model, config.image_size)
    seed_everything(config.seed)
    manifest = build_manifest(
        config.data,
        seed=config.seed,
        val_fraction=config.val_fraction,
        test_fraction=config.test_fraction,
    )
    model = build_model(config.model, len(manifest.classes), pretrained=config.pretrained).to(
        device
    )
    datasets = {
        split: ImageDataset(
            config.data,
            manifest,
            split,
            make_transform(
                spec, training=split == "train", augment=config.augment, hflip=config.hflip
            ),
        )
        for split in ("train", "val", "test")
    }
    loaders = {
        split: make_loader(dataset, config, device, training=split == "train")
        for split, dataset in datasets.items()
    }
    optimizer = torch.optim.AdamW(
        optimizer_groups(
            model,
            learning_rate=config.learning_rate,
            backbone_lr_ratio=config.backbone_lr_ratio,
            weight_decay=config.weight_decay,
        )
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    amp_enabled = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    mixer = v2.RandomChoice(
        [
            v2.MixUp(alpha=0.2, num_classes=len(manifest.classes)),
            v2.CutMix(alpha=1.0, num_classes=len(manifest.classes)),
        ]
    )
    config.output.mkdir(parents=True, exist_ok=True)
    manifest.save(config.output / "splits.json")
    metadata = asdict(config)
    metadata.update(data=str(config.data), output=str(config.output))
    save_json(config.output / "config.json", metadata)
    best_score, best_loss, stalled = -1.0, float("inf"), 0
    checkpoint_path = config.output / "best.pt"
    LOGGER.info("Device: %s; split counts: %s", device, manifest.counts())

    for epoch in range(config.epochs):
        model.set_backbone_trainable(epoch >= config.warmup_epochs)
        model.train()
        total_loss, seen = 0.0, 0
        for images, labels in loaders["train"]:
            images, labels = images.to(device), labels.to(device)
            if config.mixup and images.shape[0] > 1:
                images, labels = mixer(images, labels)
            optimizer.zero_grad(set_to_none=True)
            context = torch.autocast("cuda", dtype=torch.float16) if amp_enabled else nullcontext()
            with context:
                loss = criterion(model(images), labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss; check data and learning rate")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # CUDA's scaler must be allowed to skip overflowing half-precision updates.
            nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0, error_if_nonfinite=not amp_enabled
            )
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * images.shape[0]
            seen += images.shape[0]
        val_logits, val_labels = collect_logits(model, loaders["val"], device)
        validation = classification_metrics(val_logits, val_labels, manifest.classes)
        score, val_loss = validation["macro_f1"], validation["negative_log_likelihood"]
        row = {
            "epoch": epoch + 1,
            "stage": "head" if epoch < config.warmup_epochs else "fine_tune",
            "train_loss": total_loss / seen,
            "validation": validation,
            "learning_rates": [group["lr"] for group in optimizer.param_groups],
        }
        with (config.output / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        LOGGER.info(
            "Epoch %d/%d: train loss %.4f, val macro F1 %.4f, val NLL %.4f",
            epoch + 1,
            config.epochs,
            total_loss / seen,
            score,
            val_loss,
        )
        if score > best_score or (score == best_score and val_loss < best_loss):
            best_score, best_loss, stalled = score, val_loss, 0
            _save_checkpoint(
                checkpoint_path,
                {
                    "schema_version": 1,
                    "architecture": config.model,
                    "classes": manifest.classes,
                    "preprocessing": asdict(spec),
                    "state_dict": {
                        name: value.detach().cpu() for name, value in model.state_dict().items()
                    },
                    "temperature": 1.0,
                    "epoch": epoch + 1,
                    "manifest_fingerprint": manifest.fingerprint,
                    "config": metadata,
                    "versions": {
                        "torch": str(torch.__version__),
                        "torchvision": torchvision.__version__,
                    },
                },
            )
        else:
            stalled += 1
        scheduler.step()
        # Give the unfrozen backbone a full patience window before early stopping.
        if epoch + 1 == config.warmup_epochs:
            stalled = 0
        if stalled >= config.patience and epoch >= config.warmup_epochs:
            LOGGER.info("Early stopping after %d epochs", epoch + 1)
            break

    model, checkpoint, _ = load_checkpoint(checkpoint_path, device)
    val_logits, val_labels = collect_logits(model, loaders["val"], device)
    # clone() here produces ordinary tensors usable by autograd, outside inference_mode.
    temperature = (
        fit_temperature(val_logits.clone(), val_labels.clone()) if config.calibrate else 1.0
    )
    checkpoint["temperature"] = temperature
    _save_checkpoint(checkpoint_path, checkpoint)
    test_logits, test_labels = collect_logits(model, loaders["test"], device)
    report = {
        "architecture": config.model,
        "best_epoch": checkpoint["epoch"],
        "device": str(device),
        "temperature": temperature,
        "split_counts": manifest.counts(),
        "manifest_fingerprint": manifest.fingerprint,
        "versions": checkpoint["versions"],
        "validation": classification_metrics(val_logits, val_labels, manifest.classes),
        "validation_calibrated": classification_metrics(
            val_logits, val_labels, manifest.classes, temperature=temperature
        ),
        "test": classification_metrics(test_logits, test_labels, manifest.classes),
        "test_calibrated": classification_metrics(
            test_logits, test_labels, manifest.classes, temperature=temperature
        ),
    }
    save_json(config.output / "metrics.json", report)
    return report


def evaluate(
    checkpoint_path: Path,
    data: Path,
    *,
    split: str = "test",
    manifest_path: Path | None = None,
    device: str = "auto",
    batch_size: int = 16,
) -> dict:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    target = select_device(device)
    model, checkpoint, spec = load_checkpoint(checkpoint_path, target)
    manifest = Manifest.load(manifest_path or checkpoint_path.parent / "splits.json")
    if (
        manifest.fingerprint != checkpoint["manifest_fingerprint"]
        or manifest.classes != checkpoint["classes"]
    ):
        raise ValueError("Manifest does not match this checkpoint's dataset and class mapping")
    manifest.verify_files(data, split)
    dataset = ImageDataset(data, manifest, split, make_transform(spec))
    logits, labels = collect_logits(model, DataLoader(dataset, batch_size=batch_size), target)
    return {
        "split": split,
        "temperature": checkpoint["temperature"],
        "raw": classification_metrics(logits, labels, manifest.classes),
        "calibrated": classification_metrics(
            logits, labels, manifest.classes, temperature=checkpoint["temperature"]
        ),
    }


@torch.inference_mode()
def predict(
    checkpoint_path: Path,
    source: Path,
    *,
    top_k: int = 3,
    device: str = "auto",
    batch_size: int = 16,
) -> dict:
    if top_k < 1 or batch_size < 1:
        raise ValueError("top_k and batch_size must be positive")
    target = select_device(device)
    model, checkpoint, spec = load_checkpoint(checkpoint_path, target)
    paths = image_paths(source) if source.is_dir() else [source]
    if not paths or not all(path.is_file() for path in paths):
        raise ValueError(f"No images found at {source}")
    transform = make_transform(spec)
    classes = checkpoint["classes"]
    predictions = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        batch = torch.stack([transform(read_rgb(path)) for path in batch_paths]).to(target)
        probabilities = (model(batch).float() / checkpoint["temperature"]).softmax(dim=1)
        scores, indices = probabilities.topk(min(top_k, len(classes)), dim=1)
        for path, values, labels in zip(
            batch_paths, scores.cpu().tolist(), indices.cpu().tolist(), strict=True
        ):
            ranked = [
                {"class": classes[label], "probability": value}
                for label, value in zip(labels, values, strict=True)
            ]
            predictions.append(
                {
                    "image": str(path),
                    "predicted_class": ranked[0]["class"],
                    "top_k": ranked,
                }
            )
    return {"temperature": checkpoint["temperature"], "predictions": predictions}
